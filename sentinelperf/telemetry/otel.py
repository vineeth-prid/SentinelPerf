"""OpenTelemetry telemetry source for SentinelPerf"""

import json
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

import httpx

from sentinelperf.telemetry.base import (
    TelemetrySource,
    TelemetryData,
    EndpointMetrics,
    TrafficPattern,
)
from sentinelperf.config.schema import TelemetrySourceConfig


@dataclass
class OTELSpan:
    """Parsed OTEL span data"""
    trace_id: str
    span_id: str
    service_name: str
    operation_name: str
    start_time: datetime
    end_time: datetime
    duration_ms: float
    status_code: int
    http_method: str
    http_url: str
    http_route: str
    http_status_code: int
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_error(self) -> bool:
        """Check if span represents an error"""
        return self.http_status_code >= 400 or self.status_code != 0
    
    @property
    def endpoint(self) -> str:
        """Get normalized endpoint path"""
        # Prefer route over URL for cleaner grouping
        if self.http_route:
            return self.http_route
        # Extract path from URL
        if self.http_url:
            from urllib.parse import urlparse
            parsed = urlparse(self.http_url)
            return parsed.path or "/"
        return self.operation_name or "/"


class OpenTelemetrySource(TelemetrySource):
    """
    OpenTelemetry-based telemetry source.
    
    Supports multiple ingestion modes:
    1. OTLP HTTP endpoint (collector query API)
    2. Jaeger/Tempo query API
    3. OTEL JSON export file (for offline/testing)
    
    Extracts from spans:
    - service name
    - endpoint / operation
    - duration
    - status
    - timestamp
    """
    
    def __init__(self, config: TelemetrySourceConfig):
        self.config = config
        self.endpoint = config.endpoint or "http://localhost:4318"
        self.file_path = config.path  # Optional: OTEL JSON export file
        self._connected = False
        self._spans: List[OTELSpan] = []
    
    @property
    def source_name(self) -> str:
        return "otel"
    
    async def connect(self) -> bool:
        """
        Verify connection to OTEL source.
        
        Returns:
            True if source is accessible
        """
        # If file path is provided, check file exists
        if self.file_path:
            path = Path(self.file_path)
            if path.exists():
                self._connected = True
                return True
            return False
        
        # Try OTLP HTTP endpoint health check
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                # Try common health endpoints
                for health_path in ["/health", "/api/health", "/ready", "/"]:
                    try:
                        response = await client.get(f"{self.endpoint}{health_path}")
                        if response.status_code < 500:
                            self._connected = True
                            return True
                    except httpx.RequestError:
                        continue
        except Exception:
            pass
        
        # Even if health check fails, we can still try to use the endpoint
        self._connected = True
        return True
    
    async def fetch_data(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        endpoints: Optional[List[str]] = None,
    ) -> TelemetryData:
        """
        Fetch and parse telemetry data from OpenTelemetry source.
        
        Supports:
        - OTEL JSON export files
        - Jaeger API queries
        - Tempo API queries
        """
        if not self._connected:
            await self.connect()
        
        end_time = end_time or datetime.utcnow()
        start_time = start_time or (end_time - timedelta(hours=1))
        
        # Load spans from source
        if self.file_path:
            self._spans = await self._load_from_file(self.file_path)
        else:
            self._spans = await self._fetch_from_endpoint(start_time, end_time)
        
        # Filter by time range
        self._spans = [
            s for s in self._spans
            if start_time <= s.start_time <= end_time
        ]
        
        # Filter by endpoints if specified
        if endpoints:
            self._spans = [
                s for s in self._spans
                if s.endpoint in endpoints or any(ep in s.endpoint for ep in endpoints)
            ]
        
        # Aggregate into endpoint metrics
        endpoint_metrics = self._aggregate_endpoint_metrics(self._spans)
        
        return TelemetryData(
            source=self.source_name,
            collection_start=start_time,
            collection_end=end_time,
            endpoints=endpoint_metrics,
            traffic_patterns=[],
            raw_metrics={
                "total_spans": len(self._spans),
                "unique_services": len(set(s.service_name for s in self._spans)),
                "unique_endpoints": len(set(s.endpoint for s in self._spans)),
            },
        )
    
    async def _load_from_file(self, file_path: str) -> List[OTELSpan]:
        """Load spans from OTEL JSON export file"""
        path = Path(file_path)
        if not path.exists():
            return []
        
        spans = []
        
        with open(path, "r") as f:
            # Support both single JSON and newline-delimited JSON
            content = f.read().strip()
            
            if content.startswith("["):
                # Array of spans
                data = json.loads(content)
                for item in data:
                    span = self._parse_span(item)
                    if span:
                        spans.append(span)
            else:
                # Newline-delimited JSON or OTEL export format
                for line in content.split("\n"):
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                        # Handle OTEL export format with resourceSpans
                        if "resourceSpans" in item:
                            spans.extend(self._parse_otel_export(item))
                        else:
                            span = self._parse_span(item)
                            if span:
                                spans.append(span)
                    except json.JSONDecodeError:
                        continue
        
        return spans
    
    def _parse_otel_export(self, data: Dict[str, Any]) -> List[OTELSpan]:
        """Parse OTEL export format (resourceSpans)"""
        spans = []
        
        for resource_span in data.get("resourceSpans", []):
            # Extract service name from resource attributes
            service_name = "unknown"
            resource = resource_span.get("resource", {})
            for attr in resource.get("attributes", []):
                if attr.get("key") == "service.name":
                    service_name = attr.get("value", {}).get("stringValue", "unknown")
                    break
            
            # Process each scope
            for scope_span in resource_span.get("scopeSpans", []):
                for span_data in scope_span.get("spans", []):
                    span = self._parse_otel_span(span_data, service_name)
                    if span:
                        spans.append(span)
        
        return spans
    
    def _parse_otel_span(self, span_data: Dict[str, Any], service_name: str) -> Optional[OTELSpan]:
        """Parse a single OTEL format span"""
        try:
            # Extract timestamps (nanoseconds to datetime)
            start_ns = int(span_data.get("startTimeUnixNano", 0))
            end_ns = int(span_data.get("endTimeUnixNano", 0))
            
            if start_ns == 0:
                return None
            
            start_time = datetime.utcfromtimestamp(start_ns / 1e9)
            end_time = datetime.utcfromtimestamp(end_ns / 1e9) if end_ns else start_time
            duration_ms = (end_ns - start_ns) / 1e6
            
            # Extract attributes
            attributes = {}
            http_method = "GET"
            http_url = ""
            http_route = ""
            http_status_code = 200
            
            for attr in span_data.get("attributes", []):
                key = attr.get("key", "")
                value = attr.get("value", {})
                
                # Get the actual value from the value dict
                actual_value = (
                    value.get("stringValue") or
                    value.get("intValue") or
                    value.get("doubleValue") or
                    value.get("boolValue")
                )
                attributes[key] = actual_value
                
                # Extract HTTP attributes
                if key == "http.method":
                    http_method = actual_value or "GET"
                elif key == "http.url":
                    http_url = actual_value or ""
                elif key == "http.route":
                    http_route = actual_value or ""
                elif key == "http.target":
                    http_route = http_route or actual_value or ""
                elif key == "http.status_code":
                    http_status_code = int(actual_value) if actual_value else 200
                elif key == "url.path":
                    http_route = http_route or actual_value or ""
            
            # Extract status
            status = span_data.get("status", {})
            status_code = 0 if status.get("code") == "STATUS_CODE_OK" else 1
            
            return OTELSpan(
                trace_id=span_data.get("traceId", ""),
                span_id=span_data.get("spanId", ""),
                service_name=service_name,
                operation_name=span_data.get("name", ""),
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                status_code=status_code,
                http_method=http_method,
                http_url=http_url,
                http_route=http_route,
                http_status_code=http_status_code,
                attributes=attributes,
            )
        except Exception:
            return None
    
    def _parse_span(self, data: Dict[str, Any]) -> Optional[OTELSpan]:
        """Parse a generic span format (Jaeger, Zipkin, etc.)"""
        try:
            # Try different timestamp formats
            start_time = None
            duration_ms = 0
            
            # Jaeger format
            if "startTime" in data:
                start_us = data["startTime"]
                start_time = datetime.utcfromtimestamp(start_us / 1e6)
                duration_ms = data.get("duration", 0) / 1000  # Jaeger uses microseconds
            
            # Generic format
            elif "timestamp" in data:
                ts = data["timestamp"]
                if isinstance(ts, (int, float)):
                    start_time = datetime.utcfromtimestamp(ts / 1000 if ts > 1e12 else ts)
                else:
                    start_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                duration_ms = data.get("duration_ms", data.get("duration", 0))
            
            if not start_time:
                return None
            
            end_time = start_time + timedelta(milliseconds=duration_ms)
            
            # Extract HTTP info from tags/attributes
            tags = data.get("tags", data.get("attributes", {}))
            if isinstance(tags, list):
                tags = {t.get("key"): t.get("value") for t in tags}
            
            return OTELSpan(
                trace_id=data.get("traceID", data.get("trace_id", "")),
                span_id=data.get("spanID", data.get("span_id", "")),
                service_name=data.get("serviceName", data.get("service_name", "unknown")),
                operation_name=data.get("operationName", data.get("name", "")),
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                status_code=0 if tags.get("status.code", "OK") == "OK" else 1,
                http_method=tags.get("http.method", "GET"),
                http_url=tags.get("http.url", ""),
                http_route=tags.get("http.route", tags.get("http.target", "")),
                http_status_code=int(tags.get("http.status_code", 200)),
                attributes=tags,
            )
        except Exception:
            return None
    
    async def _fetch_from_endpoint(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> List[OTELSpan]:
        """Fetch spans from OTEL collector/Jaeger/Tempo API"""
        spans = []
        
        # Try Jaeger API format
        try:
            spans = await self._fetch_jaeger_spans(start_time, end_time)
            if spans:
                return spans
        except Exception:
            pass
        
        # Try Tempo API format
        try:
            spans = await self._fetch_tempo_spans(start_time, end_time)
            if spans:
                return spans
        except Exception:
            pass
        
        return spans
    
    async def _fetch_jaeger_spans(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> List[OTELSpan]:
        """Fetch from Jaeger query API"""
        spans = []
        
        async with httpx.AsyncClient(timeout=30) as client:
            # First, get list of services
            services_url = f"{self.endpoint}/api/services"
            try:
                resp = await client.get(services_url)
                if resp.status_code != 200:
                    return []
                services = resp.json().get("data", [])
            except Exception:
                return []
            
            # Fetch traces for each service
            for service in services[:10]:  # Limit to 10 services
                traces_url = f"{self.endpoint}/api/traces"
                params = {
                    "service": service,
                    "start": int(start_time.timestamp() * 1e6),
                    "end": int(end_time.timestamp() * 1e6),
                    "limit": 1000,
                }
                
                try:
                    resp = await client.get(traces_url, params=params)
                    if resp.status_code == 200:
                        data = resp.json()
                        for trace in data.get("data", []):
                            for span_data in trace.get("spans", []):
                                span_data["serviceName"] = service
                                span = self._parse_span(span_data)
                                if span:
                                    spans.append(span)
                except Exception:
                    continue
        
        return spans
    
    async def _fetch_tempo_spans(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> List[OTELSpan]:
        """Fetch from Grafana Tempo API"""
        spans = []
        
        async with httpx.AsyncClient(timeout=30) as client:
            # Tempo search API
            search_url = f"{self.endpoint}/api/search"
            params = {
                "start": int(start_time.timestamp()),
                "end": int(end_time.timestamp()),
                "limit": 1000,
            }
            
            try:
                resp = await client.get(search_url, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    for trace_meta in data.get("traces", []):
                        trace_id = trace_meta.get("traceID")
                        if trace_id:
                            # Fetch full trace
                            trace_url = f"{self.endpoint}/api/traces/{trace_id}"
                            trace_resp = await client.get(trace_url)
                            if trace_resp.status_code == 200:
                                trace_data = trace_resp.json()
                                spans.extend(self._parse_otel_export(trace_data))
            except Exception:
                pass
        
        return spans
    
    def _aggregate_endpoint_metrics(self, spans: List[OTELSpan]) -> List[EndpointMetrics]:
        """Aggregate spans into endpoint metrics"""
        
        # Group by endpoint + method
        endpoint_data: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(lambda: {
            "latencies": [],
            "errors": 0,
            "total": 0,
            "timestamps": [],
        })
        
        for span in spans:
            key = (span.endpoint, span.http_method)
            endpoint_data[key]["latencies"].append(span.duration_ms)
            endpoint_data[key]["total"] += 1
            endpoint_data[key]["timestamps"].append(span.start_time)
            if span.is_error:
                endpoint_data[key]["errors"] += 1
        
        # Convert to EndpointMetrics
        metrics = []
        for (path, method), data in endpoint_data.items():
            latencies = data["latencies"]
            timestamps = data["timestamps"]
            
            if not latencies:
                continue
            
            # Calculate latency percentiles
            sorted_latencies = sorted(latencies)
            n = len(sorted_latencies)
            
            # Calculate throughput (requests per minute)
            if len(timestamps) >= 2:
                time_range = (max(timestamps) - min(timestamps)).total_seconds()
                throughput_rpm = (data["total"] / time_range * 60) if time_range > 0 else 0
            else:
                throughput_rpm = data["total"]  # Single minute assumption
            
            metrics.append(EndpointMetrics(
                path=path,
                method=method,
                request_count=data["total"],
                error_count=data["errors"],
                latency_avg_ms=statistics.mean(latencies),
                latency_p50_ms=sorted_latencies[int(n * 0.50)] if n > 0 else 0,
                latency_p95_ms=sorted_latencies[int(n * 0.95)] if n > 0 else 0,
                latency_p99_ms=sorted_latencies[int(n * 0.99)] if n > 0 else 0,
                throughput_rpm=throughput_rpm,
            ))
        
        # Sort by request count
        return sorted(metrics, key=lambda x: x.request_count, reverse=True)
    
    async def infer_traffic_patterns(self, data: TelemetryData) -> List[TrafficPattern]:
        """
        Analyze telemetry data to infer traffic patterns.
        
        Rules-based inference (no LLM):
        - Steady: Low variance in RPS over time
        - Bursty: High variance in RPS
        - Periodic: Clear hourly/daily patterns
        """
        patterns = []
        
        if not self._spans:
            return patterns
        
        # Group spans by hour
        hourly_counts: Dict[int, int] = defaultdict(int)
        for span in self._spans:
            hour = span.start_time.hour
            hourly_counts[hour] += 1
        
        if not hourly_counts:
            return patterns
        
        counts = list(hourly_counts.values())
        avg_count = statistics.mean(counts) if counts else 0
        
        # Calculate coefficient of variation
        if avg_count > 0 and len(counts) > 1:
            std_dev = statistics.stdev(counts)
            cv = std_dev / avg_count
        else:
            cv = 0
        
        # Determine pattern type
        if cv < 0.3:
            pattern_type = "steady"
            description = "Consistent traffic with low variance"
        elif cv < 0.7:
            pattern_type = "moderate"
            description = "Moderate traffic variance throughout the day"
        else:
            pattern_type = "bursty"
            description = "High variance traffic with significant bursts"
        
        # Find peak hours
        if hourly_counts:
            threshold = avg_count * 1.5
            peak_hours = [h for h, c in hourly_counts.items() if c >= threshold]
        else:
            peak_hours = []
        
        # Calculate RPS
        if self._spans:
            time_range = (max(s.start_time for s in self._spans) - 
                         min(s.start_time for s in self._spans)).total_seconds()
            avg_rps = len(self._spans) / time_range if time_range > 0 else 0
            peak_rps = max(counts) / 3600 if counts else 0  # Assuming hourly buckets
        else:
            avg_rps = 0
            peak_rps = 0
        
        patterns.append(TrafficPattern(
            pattern_type=pattern_type,
            peak_rps=peak_rps,
            avg_rps=avg_rps,
            peak_hours=sorted(peak_hours),
            description=description,
        ))
        
        return patterns
    
    async def close(self) -> None:
        """Close OTEL connection"""
        self._connected = False
        self._spans = []
