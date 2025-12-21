"""Agent state management for LangGraph orchestration"""

from typing import Optional, List, Dict, Any, Literal
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class AgentPhase(str, Enum):
    """Execution phases of the agent"""
    INIT = "init"
    TELEMETRY_ANALYSIS = "telemetry_analysis"
    TEST_GENERATION = "test_generation"
    LOAD_EXECUTION = "load_execution"
    RESULTS_COLLECTION = "results_collection"
    BREAKING_POINT_DETECTION = "breaking_point_detection"
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"
    REPORT_GENERATION = "report_generation"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class TelemetryInsight:
    """Insights derived from telemetry data"""
    source: str  # otel, logs, prometheus
    endpoints: List[Dict[str, Any]] = field(default_factory=list)
    traffic_patterns: Dict[str, Any] = field(default_factory=dict)
    baseline_metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class LoadTestResult:
    """Results from a single load test execution"""
    test_type: str  # load, stress, spike
    vus: int
    duration: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    error_rate: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    throughput_rps: float = 0.0
    raw_output: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BreakingPoint:
    """Detected breaking point information"""
    vus_at_break: int
    rps_at_break: float
    failure_type: str  # error_rate, latency, timeout
    threshold_exceeded: str
    observed_value: float
    threshold_value: float
    confidence: float  # 0.0 to 1.0
    signals: List[str] = field(default_factory=list)


@dataclass
class RootCauseAnalysis:
    """Root cause analysis results"""
    primary_cause: str
    confidence: float  # 0.0 to 1.0
    supporting_evidence: List[str] = field(default_factory=list)
    reasoning_steps: List[str] = field(default_factory=list)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    llm_mode: str = "unknown"  # ollama, rules, mock


@dataclass
class AgentState:
    """
    Central state object for LangGraph agent orchestration.
    
    This state is passed between nodes and updated as the agent progresses.
    """
    # Current phase
    phase: AgentPhase = AgentPhase.INIT
    
    # Configuration reference
    environment: str = ""
    target_url: str = ""
    
    # Telemetry analysis results
    telemetry_source: Optional[str] = None
    telemetry_insights: Optional[TelemetryInsight] = None
    
    # Generated test configurations
    generated_tests: List[Dict[str, Any]] = field(default_factory=list)
    
    # Load test execution results
    load_results: List[LoadTestResult] = field(default_factory=list)
    
    # Breaking point analysis
    breaking_point: Optional[BreakingPoint] = None
    
    # Root cause analysis
    root_cause: Optional[RootCauseAnalysis] = None
    
    # Error tracking
    errors: List[str] = field(default_factory=list)
    
    # Execution metadata
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    def add_error(self, error: str) -> None:
        """Add an error to the state"""
        self.errors.append(error)
        self.phase = AgentPhase.ERROR
    
    def mark_complete(self) -> None:
        """Mark execution as complete"""
        self.completed_at = datetime.utcnow()
        self.phase = AgentPhase.COMPLETE


@dataclass
class ExecutionResult:
    """Final execution result returned to CLI"""
    success: bool
    state: AgentState
    summary: str
    markdown_report_path: Optional[str] = None
    json_report_path: Optional[str] = None
    
    @property
    def console_summary(self) -> str:
        """Generate max 5-line console summary"""
        lines = []
        
        if self.success:
            lines.append(f"✓ SentinelPerf analysis complete for {self.state.target_url}")
        else:
            lines.append(f"✗ SentinelPerf analysis failed for {self.state.target_url}")
        
        if self.state.breaking_point:
            bp = self.state.breaking_point
            lines.append(f"Breaking point: {bp.vus_at_break} VUs @ {bp.rps_at_break:.1f} RPS ({bp.failure_type})")
        
        if self.state.root_cause:
            rc = self.state.root_cause
            lines.append(f"Root cause: {rc.primary_cause} (confidence: {rc.confidence:.0%})")
        
        if self.state.errors:
            lines.append(f"Errors: {len(self.state.errors)}")
        
        if self.markdown_report_path:
            lines.append(f"Report: {self.markdown_report_path}")
        
        return "\n".join(lines[:5])  # Max 5 lines
