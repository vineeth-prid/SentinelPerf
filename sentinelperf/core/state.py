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


class ExecutionStatus(str, Enum):
    """
    Clear execution status for CLI output.
    
    SUCCESS: Tests ran, reports generated, no issues
    SUCCESS_WITH_WARNINGS: Tests ran, reports generated, but confidence reduced
    FAILED_TO_EXECUTE: Tests could not run or reports could not be generated
    """
    SUCCESS = "SUCCESS"
    SUCCESS_WITH_WARNINGS = "SUCCESS_WITH_WARNINGS"
    FAILED_TO_EXECUTE = "FAILED_TO_EXECUTE"


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
    
    # Execution status tracking
    execution_status: Optional[str] = None  # ExecutionStatus value
    execution_stop_reason: Optional[str] = None  # Human-readable stop reason
    
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
    
    def get_execution_status(self) -> ExecutionStatus:
        """
        Determine execution status based on state.
        
        SUCCESS: Tests ran, reports generated, no confidence reduction
        SUCCESS_WITH_WARNINGS: Tests ran, reports generated, but confidence reduced
        FAILED_TO_EXECUTE: Tests could not run or reports could not be generated
        """
        # If report wasn't generated, execution failed
        if not self.state.report_generated:
            return ExecutionStatus.FAILED_TO_EXECUTE
        
        # If phase is ERROR, execution failed
        if self.state.phase == AgentPhase.ERROR:
            return ExecutionStatus.FAILED_TO_EXECUTE
        
        # Check for warnings that reduce confidence
        has_warnings = False
        
        # Infrastructure saturation reduces confidence
        if self.state.infra_saturation:
            if self.state.infra_saturation.get("confidence_penalty", 0) > 0:
                has_warnings = True
            if self.state.infra_saturation.get("saturated_at_break", False):
                has_warnings = True
        
        # Root cause analysis confidence below 0.5 is a warning
        if self.state.root_cause and self.state.root_cause.confidence < 0.5:
            has_warnings = True
        
        # LLM mode fallback to rules is a warning
        if self.state.root_cause and self.state.root_cause.llm_mode == "rules":
            has_warnings = True
        
        # Errors during execution (but still completed) are warnings
        if self.state.errors:
            has_warnings = True
        
        return ExecutionStatus.SUCCESS_WITH_WARNINGS if has_warnings else ExecutionStatus.SUCCESS
    
    def get_test_case_count(self) -> int:
        """Get total number of test cases executed"""
        return len(self.state.load_results)
    
    def get_max_vus_reached(self) -> int:
        """Get the maximum VUs actually reached during execution"""
        if self.state.achieved_max_vus > 0:
            return self.state.achieved_max_vus
        if self.state.load_results:
            return max((r.vus for r in self.state.load_results), default=0)
        return 0
    
    def get_stop_reason(self) -> str:
        """Get human-readable reason why execution stopped"""
        # Use explicit stop reason if set
        if self.state.execution_stop_reason:
            return self.state.execution_stop_reason
        
        # Derive from early_stop_reason
        if self.state.early_stop_reason:
            reason_map = {
                "error_rate_exceeded": "Breaking point detected (error rate threshold exceeded)",
                "latency_exceeded": "Breaking point detected (latency threshold exceeded)",
                "throughput_degradation": "Breaking point detected (throughput degradation)",
                "max_limit_reached": "Configured maximum VUs reached",
            }
            return reason_map.get(self.state.early_stop_reason, self.state.early_stop_reason)
        
        # Check if max VUs was reached
        if self.state.configured_max_vus > 0:
            if self.get_max_vus_reached() >= self.state.configured_max_vus:
                return "Configured maximum VUs reached"
        
        # Default: completed all planned tests
        return "All planned test stages completed"
    
    @property
    def console_summary(self) -> str:
        """Generate max 5-line console summary"""
        lines = []
        
        status = self.get_execution_status()
        
        if status == ExecutionStatus.SUCCESS:
            lines.append(f"✓ SentinelPerf analysis complete for {self.state.target_url}")
        elif status == ExecutionStatus.SUCCESS_WITH_WARNINGS:
            lines.append(f"⚠ SentinelPerf analysis complete (with warnings) for {self.state.target_url}")
        else:
            lines.append(f"✗ SentinelPerf execution failed for {self.state.target_url}")
        
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
