"""Baseline behavior inference for SentinelPerf"""

import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from enum import Enum

from sentinelperf.telemetry.base import TelemetryData, EndpointMetrics, TrafficPattern


class DataQualityFlag(str, Enum):
    """Data quality warning flags for baseline confidence"""
    LOW_SAMPLE_SIZE = "LOW_SAMPLE_SIZE"  # < 100 requests
    VERY_LOW_SAMPLE_SIZE = "VERY_LOW_SAMPLE_SIZE"  # < 30 requests
    SHORT_TIME_WINDOW = "SHORT_TIME_WINDOW"  # < 5 minutes
    VERY_SHORT_TIME_WINDOW = "VERY_SHORT_TIME_WINDOW"  # < 1 minute
    SKEWED_ENDPOINT_DISTRIBUTION = "SKEWED_ENDPOINT_DISTRIBUTION"  # Top endpoint > 50%
    SINGLE_ENDPOINT = "SINGLE_ENDPOINT"  # Only 1 endpoint
    HIGH_ERROR_RATE = "HIGH_ERROR_RATE"  # > 10% baseline errors
    NO_TRAFFIC_PATTERNS = "NO_TRAFFIC_PATTERNS"  # Could not infer patterns
    MISSING_LATENCY_DATA = "MISSING_LATENCY_DATA"  # No latency metrics


@dataclass
class BaselineConfidence:
    """Confidence assessment for baseline data"""
    score: float  # 0.0 to 1.0
    level: str  # "HIGH", "MEDIUM", "LOW", "VERY_LOW"
    flags: Set[DataQualityFlag] = field(default_factory=set)
    
    # Thresholds used for assessment
    sample_count: int = 0
    time_window_minutes: float = 0.0
    endpoint_count: int = 0
    top_endpoint_share: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "level": self.level,
            "flags": [f.value for f in self.flags],
            "sample_count": self.sample_count,
            "time_window_minutes": self.time_window_minutes,
            "endpoint_count": self.endpoint_count,
            "top_endpoint_share": self.top_endpoint_share,
        }
    
    @property
    def is_reliable(self) -> bool:
        """Check if baseline is reliable enough for load testing"""
        return self.score >= 0.5 and DataQualityFlag.VERY_LOW_SAMPLE_SIZE not in self.flags


@dataclass
class EndpointBaseline:
    """Baseline metrics for a single endpoint"""
    path: str
    method: str
    
    # Traffic volume
    request_count: int
    requests_per_minute: float
    traffic_share: float  # Percentage of total traffic
    
    # Latency baseline (ms)
    latency_avg: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    
    # Error baseline
    error_rate: float
    error_count: int
    
    # Weight for load testing (based on traffic share)
    load_weight: int = 1


@dataclass
class BaselineBehavior:
    """
    Inferred baseline behavior from telemetry.
    
    This is the foundation for load test generation.
    No load has been applied yet - this represents normal operating conditions.
    """
    # Time range analyzed
    analysis_start: datetime
    analysis_end: datetime
    analysis_duration_minutes: float
    
    # Global baseline metrics
    total_requests: int
    total_errors: int
    global_error_rate: float
    global_avg_rps: float
    global_peak_rps: float
    
    # Latency distribution (all endpoints)
    global_latency_p50: float
    global_latency_p95: float
    global_latency_p99: float
    
    # Top N endpoints by traffic
    top_endpoints: List[EndpointBaseline] = field(default_factory=list)
    
    # Traffic patterns
    traffic_patterns: List[TrafficPattern] = field(default_factory=list)
    
    # Service information
    services: List[str] = field(default_factory=list)
    
    # Raw data reference
    telemetry_source: str = ""
    
    # Confidence assessment
    confidence: Optional[BaselineConfidence] = None
    
    def to_load_plan_input(self) -> Dict[str, Any]:
        """
        Convert baseline to load plan input format.
        
        This is the contract between telemetry analysis and test generation.
        """
        return {
            "baseline": {
                "avg_rps": self.global_avg_rps,
                "peak_rps": self.global_peak_rps,
                "error_rate": self.global_error_rate,
                "latency_p50_ms": self.global_latency_p50,
                "latency_p95_ms": self.global_latency_p95,
                "latency_p99_ms": self.global_latency_p99,
            },
            "endpoints": [
                {
                    "path": ep.path,
                    "method": ep.method,
                    "weight": ep.load_weight,
                    "baseline_latency_p95": ep.latency_p95,
                    "baseline_error_rate": ep.error_rate,
                }
                for ep in self.top_endpoints
            ],
            "patterns": [
                {
                    "type": p.pattern_type,
                    "avg_rps": p.avg_rps,
                    "peak_rps": p.peak_rps,
                    "peak_hours": p.peak_hours,
                }
                for p in self.traffic_patterns
            ],
            "confidence": self.confidence.to_dict() if self.confidence else None,
            "recommended_initial_vus": self._calculate_initial_vus(),
            "recommended_max_vus": self._calculate_max_vus(),
        }
    
    def _calculate_initial_vus(self) -> int:
        """Calculate recommended initial VUs based on baseline"""
        # Start at ~10% of baseline RPS, minimum 1
        initial = max(1, int(self.global_avg_rps * 0.1))
        return min(initial, 10)  # Cap at 10 for safety
    
    def _calculate_max_vus(self) -> int:
        """Calculate recommended max VUs for breaking point discovery"""
        # Target 10x baseline RPS for stress testing
        if self.global_peak_rps > 0:
            max_vus = int(self.global_peak_rps * 10)
        else:
            max_vus = 100  # Default
        
        return max(10, min(max_vus, 500))  # Between 10 and 500
    
    def get_confidence_summary(self) -> str:
        """Get human-readable confidence summary"""
        if not self.confidence:
            return "Confidence: UNKNOWN (not assessed)"
        
        c = self.confidence
        flags_str = ", ".join(f.value for f in c.flags) if c.flags else "none"
        
        return (
            f"Baseline confidence: {c.level} "
            f"({c.sample_count} spans, {c.time_window_minutes:.1f} min window) "
            f"[flags: {flags_str}]"
        )


class BaselineInference:
    """
    Rules-based baseline behavior inference.
    
    No LLM involvement - pure statistical analysis of telemetry data.
    
    Infers:
    - Top N endpoints by traffic
    - Baseline latency distribution (p50/p95/p99)
    - Approximate baseline RPS / throughput
    """
    
    def __init__(self, top_n: int = 10):
        self.top_n = top_n
    
    def infer(self, telemetry: TelemetryData) -> BaselineBehavior:
        """
        Infer baseline behavior from telemetry data.
        
        Args:
            telemetry: Aggregated telemetry data
            
        Returns:
            BaselineBehavior with inferred metrics
        """
        # Calculate time range
        duration_seconds = (telemetry.collection_end - telemetry.collection_start).total_seconds()
        duration_minutes = duration_seconds / 60 if duration_seconds > 0 else 1
        
        # Aggregate global metrics
        total_requests = sum(ep.request_count for ep in telemetry.endpoints)
        total_errors = sum(ep.error_count for ep in telemetry.endpoints)
        global_error_rate = total_errors / total_requests if total_requests > 0 else 0
        
        # Calculate global RPS
        global_avg_rps = total_requests / duration_seconds if duration_seconds > 0 else 0
        
        # Estimate peak RPS from throughput_rpm
        if telemetry.endpoints:
            global_peak_rps = max(ep.throughput_rpm / 60 for ep in telemetry.endpoints)
        else:
            global_peak_rps = global_avg_rps
        
        # Calculate global latency percentiles
        all_latencies = self._collect_latency_estimates(telemetry.endpoints)
        global_latency_p50, global_latency_p95, global_latency_p99 = self._calculate_percentiles(all_latencies)
        
        # Get top N endpoints
        top_endpoints = self._get_top_endpoints(telemetry.endpoints, total_requests)
        
        return BaselineBehavior(
            analysis_start=telemetry.collection_start,
            analysis_end=telemetry.collection_end,
            analysis_duration_minutes=duration_minutes,
            total_requests=total_requests,
            total_errors=total_errors,
            global_error_rate=global_error_rate,
            global_avg_rps=global_avg_rps,
            global_peak_rps=global_peak_rps,
            global_latency_p50=global_latency_p50,
            global_latency_p95=global_latency_p95,
            global_latency_p99=global_latency_p99,
            top_endpoints=top_endpoints,
            traffic_patterns=telemetry.traffic_patterns,
            telemetry_source=telemetry.source,
        )
    
    def _collect_latency_estimates(self, endpoints: List[EndpointMetrics]) -> List[float]:
        """
        Collect latency estimates from endpoint metrics.
        
        Since we don't have raw latency values, we use weighted estimates
        based on endpoint request counts and their p50 values.
        """
        latencies = []
        
        for ep in endpoints:
            # Add weighted latency samples based on request count
            # This is an approximation since we don't have raw values
            weight = min(ep.request_count, 100)  # Cap weight for estimation
            
            # Add p50 values (most common)
            latencies.extend([ep.latency_p50_ms] * int(weight * 0.5))
            # Add values between p50 and p95
            mid_95 = (ep.latency_p50_ms + ep.latency_p95_ms) / 2
            latencies.extend([mid_95] * int(weight * 0.4))
            # Add p95 values
            latencies.extend([ep.latency_p95_ms] * int(weight * 0.09))
            # Add p99 values
            latencies.extend([ep.latency_p99_ms] * int(weight * 0.01))
        
        return latencies
    
    def _calculate_percentiles(self, latencies: List[float]) -> tuple:
        """Calculate p50, p95, p99 from latency list"""
        if not latencies:
            return 0.0, 0.0, 0.0
        
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        
        p50 = sorted_latencies[int(n * 0.50)]
        p95 = sorted_latencies[int(n * 0.95)] if n > 1 else sorted_latencies[-1]
        p99 = sorted_latencies[int(n * 0.99)] if n > 1 else sorted_latencies[-1]
        
        return p50, p95, p99
    
    def _get_top_endpoints(
        self,
        endpoints: List[EndpointMetrics],
        total_requests: int,
    ) -> List[EndpointBaseline]:
        """Get top N endpoints with baseline metrics"""
        
        # Sort by request count
        sorted_endpoints = sorted(endpoints, key=lambda x: x.request_count, reverse=True)
        top = sorted_endpoints[:self.top_n]
        
        baselines = []
        for ep in top:
            # Calculate traffic share
            traffic_share = ep.request_count / total_requests if total_requests > 0 else 0
            
            # Calculate error rate
            error_rate = ep.error_count / ep.request_count if ep.request_count > 0 else 0
            
            # Calculate load weight (1-10 scale based on traffic share)
            load_weight = max(1, min(10, int(traffic_share * 100)))
            
            baselines.append(EndpointBaseline(
                path=ep.path,
                method=ep.method,
                request_count=ep.request_count,
                requests_per_minute=ep.throughput_rpm,
                traffic_share=traffic_share,
                latency_avg=ep.latency_avg_ms,
                latency_p50=ep.latency_p50_ms,
                latency_p95=ep.latency_p95_ms,
                latency_p99=ep.latency_p99_ms,
                error_rate=error_rate,
                error_count=ep.error_count,
                load_weight=load_weight,
            ))
        
        return baselines
    
    def print_summary(self, baseline: BaselineBehavior) -> str:
        """Generate human-readable summary of baseline"""
        lines = [
            "=" * 60,
            "BASELINE BEHAVIOR SUMMARY",
            "=" * 60,
            f"Analysis Period: {baseline.analysis_start.strftime('%Y-%m-%d %H:%M')} to {baseline.analysis_end.strftime('%Y-%m-%d %H:%M')}",
            f"Duration: {baseline.analysis_duration_minutes:.1f} minutes",
            f"Source: {baseline.telemetry_source}",
            "",
            "GLOBAL METRICS:",
            f"  Total Requests: {baseline.total_requests:,}",
            f"  Average RPS: {baseline.global_avg_rps:.2f}",
            f"  Peak RPS: {baseline.global_peak_rps:.2f}",
            f"  Error Rate: {baseline.global_error_rate:.2%}",
            "",
            "LATENCY DISTRIBUTION:",
            f"  P50: {baseline.global_latency_p50:.1f}ms",
            f"  P95: {baseline.global_latency_p95:.1f}ms",
            f"  P99: {baseline.global_latency_p99:.1f}ms",
            "",
            f"TOP {len(baseline.top_endpoints)} ENDPOINTS BY TRAFFIC:",
        ]
        
        for i, ep in enumerate(baseline.top_endpoints, 1):
            lines.append(
                f"  {i}. {ep.method} {ep.path} "
                f"({ep.traffic_share:.1%} traffic, {ep.request_count:,} reqs, "
                f"P95: {ep.latency_p95:.0f}ms)"
            )
        
        if baseline.traffic_patterns:
            lines.append("")
            lines.append("TRAFFIC PATTERNS:")
            for pattern in baseline.traffic_patterns:
                lines.append(f"  - {pattern.pattern_type}: {pattern.description}")
                if pattern.peak_hours:
                    lines.append(f"    Peak hours: {pattern.peak_hours}")
        
        lines.append("")
        lines.append("LOAD TEST RECOMMENDATIONS:")
        load_plan = baseline.to_load_plan_input()
        lines.append(f"  Initial VUs: {load_plan['recommended_initial_vus']}")
        lines.append(f"  Max VUs: {load_plan['recommended_max_vus']}")
        lines.append("=" * 60)
        
        return "\n".join(lines)
