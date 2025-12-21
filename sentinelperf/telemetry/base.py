"""Base telemetry interface for SentinelPerf"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional


@dataclass
class EndpointMetrics:
    """Metrics for a single endpoint"""
    path: str
    method: str = "GET"
    request_count: int = 0
    error_count: int = 0
    latency_avg_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    throughput_rpm: float = 0.0  # Requests per minute


@dataclass
class TrafficPattern:
    """Inferred traffic pattern"""
    pattern_type: str  # steady, bursty, periodic
    peak_rps: float
    avg_rps: float
    peak_hours: List[int] = field(default_factory=list)
    description: str = ""


@dataclass
class TelemetryData:
    """Aggregated telemetry data from any source"""
    source: str  # otel, logs, prometheus
    collection_start: datetime
    collection_end: datetime
    endpoints: List[EndpointMetrics] = field(default_factory=list)
    traffic_patterns: List[TrafficPattern] = field(default_factory=list)
    raw_metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_requests(self) -> int:
        return sum(ep.request_count for ep in self.endpoints)
    
    @property
    def top_endpoints(self) -> List[EndpointMetrics]:
        """Get top 10 endpoints by request count"""
        return sorted(self.endpoints, key=lambda x: x.request_count, reverse=True)[:10]


class TelemetrySource(ABC):
    """
    Abstract base class for telemetry sources.
    
    Implementations must provide methods to:
    1. Connect to the telemetry source
    2. Fetch and parse telemetry data
    3. Convert to standardized TelemetryData format
    """
    
    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the source identifier (otel, logs, prometheus)"""
        pass
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to telemetry source.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def fetch_data(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        endpoints: Optional[List[str]] = None,
    ) -> TelemetryData:
        """
        Fetch telemetry data from source.
        
        Args:
            start_time: Start of time range (default: last 1 hour)
            end_time: End of time range (default: now)
            endpoints: Specific endpoints to filter (default: all)
            
        Returns:
            Aggregated TelemetryData
        """
        pass
    
    @abstractmethod
    async def infer_traffic_patterns(self, data: TelemetryData) -> List[TrafficPattern]:
        """
        Analyze telemetry data to infer traffic patterns.
        
        Args:
            data: Previously fetched telemetry data
            
        Returns:
            List of inferred traffic patterns
        """
        pass
    
    async def close(self) -> None:
        """Close connection to telemetry source"""
        pass
