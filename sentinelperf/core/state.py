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
    RECOMMENDATIONS = "recommendations"
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
class DeterministicFailureAnalysis:
    """
    Deterministic failure analysis results (rules-based).
    
    This is NOT root cause analysis - it's purely rule-based classification
    and pattern detection. Root cause analysis requires LLM reasoning.
    """
    failure_category: str  # From FailureCategory enum
    category_confidence: float  # 0.0 to 1.0
    classification_rationale: List[str] = field(default_factory=list)
    supporting_evidence: List[str] = field(default_factory=list)


@dataclass
class RootCauseAnalysis:
    """
    LLM-assisted root cause analysis results.
    
    This is the output from the LLM reasoning layer which explains
    WHY the failure occurred based on the deterministic analysis.
    
    The LLM is constrained to:
    - Explain why the classification makes sense
    - Connect timeline events causally
    - Translate signals into human reasoning
    - Assign explanation confidence
    
    The LLM is NOT allowed to:
    - Change the classification
    - Invent new metrics
    - Override breaking point
    - Suggest fixes (that's Phase 6)
    """
    root_cause_summary: str
    primary_cause: str
    contributing_factors: List[str] = field(default_factory=list)
    confidence: float = 0.0  # 0.0 to 1.0
    assumptions: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    failure_pattern: str = ""  # Detected failure pattern
    pattern_explanation: str = ""  # Explanation for the pattern
    llm_mode: str = "rules"  # ollama, rules, mock
    llm_model: str = ""
    llm_latency_ms: float = 0.0


@dataclass
class AgentState:
    """
    Central state object for LangGraph agent orchestration.
    
    This state is passed between nodes and updated as the agent progresses.
    """
    # Execution identity (MUST be set at run start)
    execution_id: str = ""  # UUID generated at run start
    config_file_path: str = ""  # Path to config file used
    autoscaling_enabled: bool = False  # Whether autoscaling was enabled
    
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
    
    # Load execution tracking (configured vs achieved)
    configured_max_vus: int = 0  # What was configured in config
    achieved_max_vus: int = 0    # Highest VUs actually executed
    early_stop_reason: Optional[str] = None  # Why we stopped before max
    planned_vus_stages: List[int] = field(default_factory=list)  # Planned VU levels
    executed_vus_stages: List[int] = field(default_factory=list)  # Actually executed
    
    # Breaking point analysis
    breaking_point: Optional[BreakingPoint] = None
    failure_timeline: List[Dict[str, Any]] = field(default_factory=list)
    failure_category: Optional[str] = None
    
    # Deterministic failure analysis (rules-based)
    failure_analysis: Optional[DeterministicFailureAnalysis] = None
    
    # Root cause analysis (LLM-assisted)
    root_cause: Optional[RootCauseAnalysis] = None
    
    # Recommendations
    recommendations: Optional[Dict[str, Any]] = None  # RecommendationResult.to_dict()
    
    # Infrastructure saturation
    infra_saturation: Optional[Dict[str, Any]] = None
    
    # Error tracking
    errors: List[str] = field(default_factory=list)
    
    # Execution metadata - REAL timestamps
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # Report generation tracking
    report_generated: bool = False
    
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
