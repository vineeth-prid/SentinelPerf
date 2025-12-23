"""LangGraph-based agent orchestration for SentinelPerf"""

import asyncio
from pathlib import Path
from typing import Dict, Any, Literal, Optional, List, TypedDict
from datetime import datetime

from langgraph.graph import StateGraph, END

from sentinelperf.config.schema import SentinelPerfConfig
from sentinelperf.core.state import (
    AgentState, 
    AgentPhase, 
    ExecutionResult,
    TelemetryInsight,
    LoadTestResult,
    BreakingPoint,
    DeterministicFailureAnalysis,
    RootCauseAnalysis,
)
from sentinelperf.telemetry.otel import OpenTelemetrySource
from sentinelperf.telemetry.logs import AccessLogSource
from sentinelperf.telemetry.prometheus import PrometheusSource
from sentinelperf.telemetry.baseline import BaselineInference, BaselineBehavior
from sentinelperf.load.generator import TestGenerator, TestScript
from sentinelperf.load.k6_executor import K6Executor, K6Result
from sentinelperf.analysis.breaking_point import BreakingPointDetector, BreakingPointResult, FailureCategory


# TypedDict for LangGraph state (must use dict-like state)
class GraphState(TypedDict, total=False):
    """State schema for LangGraph workflow"""
    phase: str
    environment: str
    target_url: str
    telemetry_source: Optional[str]
    telemetry_insights: Optional[TelemetryInsight]
    generated_tests: List[Dict[str, Any]]
    load_results: List[LoadTestResult]
    breaking_point: Optional[BreakingPoint]
    breaking_point_result: Optional[Dict[str, Any]]  # Full analysis result
    failure_timeline: Optional[List[Dict[str, Any]]]
    failure_category: Optional[str]
    failure_analysis: Optional[DeterministicFailureAnalysis]  # Rules-based analysis
    root_cause: Optional[RootCauseAnalysis]  # LLM-assisted analysis
    recommendations: Optional[Dict[str, Any]]  # RecommendationResult
    errors: List[str]
    started_at: str
    completed_at: Optional[str]


class SentinelPerfAgent:
    """
    Main agent orchestrator using LangGraph.
    
    Coordinates the autonomous performance analysis workflow:
    1. Telemetry Analysis - Infer traffic patterns from OTEL/logs/prometheus
    2. Test Generation - Generate load/stress/spike tests based on baseline
    3. Load Execution - Execute tests via k6
    4. Results Collection - Aggregate metrics
    5. Breaking Point Detection - Find failure threshold
    6. Root Cause Analysis - Explain why with confidence
    7. Report Generation - Output markdown, JSON, console
    """
    
    def __init__(
        self,
        config: SentinelPerfConfig,
        llm_mode: Literal["ollama", "rules", "mock"] = "ollama",
        output_dir: Path = Path("./sentinelperf-reports"),
        verbose: bool = False
    ):
        self.config = config
        self.llm_mode = llm_mode
        self.output_dir = output_dir
        self.verbose = verbose
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.baseline_inference = BaselineInference(top_n=10)
        self._baseline: Optional[BaselineBehavior] = None
        self._generated_scripts: List[TestScript] = []
        
        # Initialize k6 executor
        self.k6_executor = K6Executor(
            output_dir=self.output_dir / "k6-scripts",
            use_docker=False,  # Default to local k6
        )
        
        # Build the agent graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine"""
        
        # Define the graph with GraphState (TypedDict) as state schema
        workflow = StateGraph(GraphState)
        
        # Add nodes for each phase
        workflow.add_node("telemetry_analysis", self._node_telemetry_analysis)
        workflow.add_node("test_generation", self._node_test_generation)
        workflow.add_node("load_execution", self._node_load_execution)
        workflow.add_node("results_collection", self._node_results_collection)
        workflow.add_node("breaking_point_detection", self._node_breaking_point_detection)
        workflow.add_node("root_cause_analysis", self._node_root_cause_analysis)
        workflow.add_node("recommendations", self._node_recommendations)
        workflow.add_node("report_generation", self._node_report_generation)
        
        # Define edges (workflow sequence)
        workflow.set_entry_point("telemetry_analysis")
        workflow.add_edge("telemetry_analysis", "test_generation")
        workflow.add_edge("test_generation", "load_execution")
        workflow.add_edge("load_execution", "results_collection")
        workflow.add_edge("results_collection", "breaking_point_detection")
        workflow.add_edge("breaking_point_detection", "root_cause_analysis")
        workflow.add_edge("root_cause_analysis", "recommendations")
        workflow.add_edge("recommendations", "report_generation")
        workflow.add_edge("report_generation", END)
        
        return workflow.compile()
    
    def run(self) -> ExecutionResult:
        """Execute the full agent workflow"""
        
        # Initialize state as dict for LangGraph
        initial_state = {
            "phase": AgentPhase.INIT.value,
            "environment": self.config._active_env or "unknown",
            "target_url": self.config.target.base_url,
            "telemetry_source": None,
            "telemetry_insights": None,
            "generated_tests": [],
            "load_results": [],
            "breaking_point": None,
            "root_cause": None,
            "errors": [],
            "started_at": datetime.utcnow().isoformat(),
            "completed_at": None,
        }
        
        if self.verbose:
            print("Starting SentinelPerf analysis...")
            print(f"Target: {initial_state['target_url']}")
            print(f"LLM Mode: {self.llm_mode}")
        
        try:
            # Run the graph
            final_state_dict = self.graph.invoke(initial_state)
            
            # Convert dict back to AgentState
            final_state = self._dict_to_agent_state(final_state_dict)
            
            # Build result
            return ExecutionResult(
                success=final_state.phase == AgentPhase.COMPLETE,
                state=final_state,
                summary=self._generate_summary(final_state),
                markdown_report_path=str(self.output_dir / "report.md"),
                json_report_path=str(self.output_dir / "summary.json"),
            )
            
        except Exception as e:
            # Handle execution errors
            error_state = AgentState(
                phase=AgentPhase.ERROR,
                environment=self.config._active_env or "unknown",
                target_url=self.config.target.base_url,
            )
            error_state.errors.append(str(e))
            
            return ExecutionResult(
                success=False,
                state=error_state,
                summary=f"Execution failed: {e}",
            )
    
    def _dict_to_agent_state(self, d: Dict[str, Any]) -> AgentState:
        """Convert LangGraph dict output to AgentState"""
        state = AgentState(
            phase=AgentPhase(d.get("phase", "error")),
            environment=d.get("environment", "unknown"),
            target_url=d.get("target_url", ""),
            telemetry_source=d.get("telemetry_source"),
            telemetry_insights=d.get("telemetry_insights"),
            generated_tests=d.get("generated_tests", []),
            load_results=d.get("load_results", []),
            breaking_point=d.get("breaking_point"),
            failure_timeline=d.get("failure_timeline", []),
            failure_category=d.get("failure_category"),
            failure_analysis=d.get("failure_analysis"),
            root_cause=d.get("root_cause"),
            recommendations=d.get("recommendations"),
            errors=d.get("errors", []),
        )
        
        if d.get("started_at"):
            if isinstance(d["started_at"], str):
                state.started_at = datetime.fromisoformat(d["started_at"])
            else:
                state.started_at = d["started_at"]
        
        if d.get("completed_at"):
            if isinstance(d["completed_at"], str):
                state.completed_at = datetime.fromisoformat(d["completed_at"])
            else:
                state.completed_at = d["completed_at"]
        
        return state
    
    # ===========================================
    # Node Implementations (work with dict state)
    # ===========================================
    
    def _node_telemetry_analysis(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze telemetry data to infer traffic patterns.
        
        1. Connect to telemetry source (OTEL > logs > prometheus)
        2. Fetch spans/metrics
        3. Infer baseline behavior (rules-only)
        """
        state["phase"] = AgentPhase.TELEMETRY_ANALYSIS.value
        
        if self.verbose:
            print("[1/8] Analyzing telemetry...")
        
        # Determine active telemetry source
        telemetry_config = self.config.telemetry
        active_source = telemetry_config.get_active_source()
        
        if not active_source:
            state["errors"].append("No telemetry source configured or enabled")
            state["phase"] = AgentPhase.ERROR.value
            return state
        
        state["telemetry_source"] = active_source
        
        if self.verbose:
            print(f"  Using telemetry source: {active_source}")
        
        # Run async telemetry fetch
        try:
            telemetry_data, baseline = asyncio.get_event_loop().run_until_complete(
                self._fetch_and_analyze_telemetry(active_source)
            )
        except RuntimeError:
            # No event loop, create one
            telemetry_data, baseline = asyncio.run(
                self._fetch_and_analyze_telemetry(active_source)
            )
        
        if telemetry_data is None:
            state["errors"].append(f"Failed to fetch telemetry from {active_source}")
            state["phase"] = AgentPhase.ERROR.value
            return state
        
        # Store baseline for later phases
        self._baseline = baseline
        
        # Convert to TelemetryInsight for state
        state["telemetry_insights"] = TelemetryInsight(
            source=active_source,
            endpoints=[
                {
                    "path": ep.path,
                    "method": ep.method,
                    "request_count": ep.request_count,
                    "latency_p95": ep.latency_p95,
                    "error_rate": ep.error_rate,
                    "weight": ep.load_weight,
                }
                for ep in baseline.top_endpoints
            ],
            traffic_patterns={
                "type": baseline.traffic_patterns[0].pattern_type if baseline.traffic_patterns else "unknown",
                "avg_rps": baseline.global_avg_rps,
                "peak_rps": baseline.global_peak_rps,
            },
            baseline_metrics={
                "total_requests": baseline.total_requests,
                "global_error_rate": baseline.global_error_rate,
                "latency_p50": baseline.global_latency_p50,
                "latency_p95": baseline.global_latency_p95,
                "latency_p99": baseline.global_latency_p99,
            },
        )
        
        if self.verbose:
            print(f"  Analyzed {baseline.total_requests} requests")
            print(f"  Found {len(baseline.top_endpoints)} top endpoints")
            print(f"  Baseline RPS: {baseline.global_avg_rps:.2f} avg, {baseline.global_peak_rps:.2f} peak")
            print(f"  Latency P95: {baseline.global_latency_p95:.1f}ms")
        
        return state
    
    async def _fetch_and_analyze_telemetry(self, source_type: str):
        """Fetch telemetry and infer baseline"""
        
        # Create appropriate source
        if source_type == "otel":
            source_config = self.config.telemetry.otel
            source = OpenTelemetrySource(source_config)
        elif source_type == "logs":
            source_config = self.config.telemetry.logs
            source = AccessLogSource(source_config)
        elif source_type == "prometheus":
            source_config = self.config.telemetry.prometheus
            source = PrometheusSource(source_config)
        else:
            return None, None
        
        try:
            # Connect and fetch
            connected = await source.connect()
            if not connected:
                return None, None
            
            # Fetch telemetry data
            telemetry_data = await source.fetch_data()
            
            if not telemetry_data.endpoints:
                if self.verbose:
                    print(f"  Warning: No endpoints found in {source_type} telemetry")
                # Return empty baseline instead of failure
                from sentinelperf.telemetry.base import TelemetryData
                telemetry_data = TelemetryData(
                    source=source_type,
                    collection_start=datetime.utcnow(),
                    collection_end=datetime.utcnow(),
                )
            
            # Infer traffic patterns
            patterns = await source.infer_traffic_patterns(telemetry_data)
            telemetry_data.traffic_patterns = patterns
            
            # Infer baseline behavior
            baseline = self.baseline_inference.infer(telemetry_data)
            
            if self.verbose and baseline.top_endpoints:
                print(self.baseline_inference.print_summary(baseline))
            
            await source.close()
            
            return telemetry_data, baseline
            
        except Exception as e:
            if self.verbose:
                print(f"  Error fetching telemetry: {e}")
            return None, None
    
    def _node_test_generation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate baseline, stress, and spike test configurations.
        
        Uses baseline behavior to inform test parameters.
        """
        state["phase"] = AgentPhase.TEST_GENERATION.value
        
        if self.verbose:
            print("[2/8] Generating test configurations...")
        
        # Get baseline data for load planning
        load_plan_input = self._get_load_plan_input()
        
        # Get auth headers from config
        auth_headers = self._get_auth_headers()
        
        # Initialize test generator
        generator = TestGenerator(
            base_url=self.config.target.base_url,
            auth_headers=auth_headers,
        )
        self._test_generator = generator  # Store for later use
        
        # Get endpoints for testing
        endpoints = load_plan_input.get("endpoints", [])
        if not endpoints:
            # Fallback to configured endpoints
            if self.config.target.endpoints:
                endpoints = [
                    {"path": ep, "method": "GET", "weight": 1}
                    for ep in self.config.target.endpoints
                ]
            else:
                # Use health endpoint as fallback
                endpoints = [
                    {"path": self.config.target.health_endpoint, "method": "GET", "weight": 1}
                ]
        
        # Get recommended VUs from baseline, but cap by config
        recommended_initial = load_plan_input.get("recommended_initial_vus", self.config.load.initial_vus)
        recommended_max = load_plan_input.get("recommended_max_vus", self.config.load.max_vus)
        
        initial_vus = min(recommended_initial, self.config.load.initial_vus) if self.config.load.initial_vus > 0 else recommended_initial
        max_vus = min(recommended_max, self.config.load.max_vus)  # Always respect config limit
        
        # Generate tests based on baseline
        baseline_metrics = load_plan_input.get("baseline", {})
        
        if self.verbose:
            print(f"  Endpoints to test: {len(endpoints)}")
            print(f"  Initial VUs: {initial_vus}, Max VUs: {max_vus}")
            print(f"  Baseline P95: {baseline_metrics.get('latency_p95_ms', 'N/A')}ms")
        
        # Clear previous scripts
        self._generated_scripts = []
        
        # Determine test durations - shorter for small VU counts
        baseline_duration = self.config.load.hold_duration
        stress_step_duration = "10s" if max_vus <= 10 else "20s"
        spike_duration = "10s" if max_vus <= 10 else "20s"
        
        # 1. Baseline test - validate current behavior
        baseline_test = generator.generate_baseline_test(
            endpoints=endpoints,
            target_vus=max(1, initial_vus),
            duration=baseline_duration,
            error_threshold=self.config.load.error_rate_threshold,
            p95_threshold_ms=self.config.load.p95_latency_threshold_ms,
        )
        self._generated_scripts.append(baseline_test)
        
        # 2. Stress test - incremental load to find limits
        stress_test = generator.generate_stress_test(
            endpoints=endpoints,
            start_vus=1,
            max_vus=max_vus,
            step_vus=self.config.load.adaptive_step,
            step_duration=stress_step_duration,
        )
        self._generated_scripts.append(stress_test)
        
        # 3. Spike test - sudden burst
        spike_test = generator.generate_spike_test(
            endpoints=endpoints,
            baseline_vus=max(1, initial_vus),
            spike_vus=max_vus,
            spike_duration=spike_duration,
        )
        self._generated_scripts.append(spike_test)
        
        # Store test metadata in state (without script objects for serialization)
        state["generated_tests"] = [
            {
                "type": baseline_test.test_type.value,
                "name": baseline_test.name,
                "vus": max(1, initial_vus),
                "stages": len(baseline_test.stages),
                "endpoints": len(endpoints),
            },
            {
                "type": stress_test.test_type.value,
                "name": stress_test.name,
                "vus": max_vus,
                "stages": len(stress_test.stages),
                "endpoints": len(endpoints),
            },
            {
                "type": spike_test.test_type.value,
                "name": spike_test.name,
                "vus": max_vus,
                "stages": len(spike_test.stages),
                "endpoints": len(endpoints),
            },
        ]
        
        if self.verbose:
            print(f"  Generated {len(self._generated_scripts)} test scripts:")
            for script in self._generated_scripts:
                print(f"    - {script.test_type.value}: {script.name} ({len(script.stages)} stages)")
        
        return state
    
    def _get_load_plan_input(self) -> Dict[str, Any]:
        """Get load plan input from baseline or defaults"""
        if self._baseline:
            return self._baseline.to_load_plan_input()
        
        # Return defaults if no baseline available
        return {
            "baseline": {
                "avg_rps": 0,
                "peak_rps": 0,
                "error_rate": 0,
                "latency_p50_ms": 0,
                "latency_p95_ms": 0,
                "latency_p99_ms": 0,
            },
            "endpoints": [],
            "patterns": [],
            "recommended_initial_vus": self.config.load.initial_vus,
            "recommended_max_vus": self.config.load.max_vus,
        }
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers from config"""
        auth = self.config.auth
        headers = {}
        
        if auth.method == "bearer" and auth.token:
            headers["Authorization"] = f"Bearer {auth.token}"
        elif auth.method == "header" and auth.header_name and auth.header_value:
            headers[auth.header_name] = auth.header_value
        
        return headers
    
    def _node_load_execution(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute load tests using k6.
        
        Runs actual k6 tests if available, otherwise creates mock results.
        """
        state["phase"] = AgentPhase.LOAD_EXECUTION.value
        
        if self.verbose:
            print("[3/8] Executing load tests...")
        
        # Check infrastructure saturation before tests
        from sentinelperf.telemetry.infra_monitor import check_infra_saturation
        pre_test_infra = check_infra_saturation()
        
        # Check if k6 is available
        k6_available = self.k6_executor.check_available()
        
        if k6_available and self._generated_scripts:
            if self.verbose:
                k6_version = self.k6_executor.get_version()
                print(f"  k6 available: {k6_version}")
            
            load_results = []
            
            # Check if adaptive mode is enabled
            if self.config.load.adaptive_enabled:
                if self.verbose:
                    print(f"  Adaptive mode enabled")
                
                # Use first script as template for adaptive execution
                template_script = self._generated_scripts[0]
                
                k6_results = self.k6_executor.execute_adaptive(
                    script=template_script,
                    initial_vus=self.config.load.initial_vus,
                    max_vus=min(self.config.load.max_vus, 1000),  # Hard cap
                    step=self.config.load.adaptive_step,
                    hold_seconds=self.config.load.adaptive_hold_seconds,
                    error_threshold=self.config.load.error_rate_threshold,
                    latency_slope_threshold=self.config.load.adaptive_latency_slope_threshold,
                    fine_step_divisor=self.config.load.adaptive_fine_step_divisor,
                    timeout_per_step=120,
                    verbose=self.verbose,
                )
                
                # Convert K6Result to LoadTestResult
                for k6_result in k6_results:
                    load_result = LoadTestResult(
                        test_type=k6_result.test_type,
                        vus=k6_result.metrics.vus_max,
                        duration=f"{k6_result.duration_seconds:.0f}s",
                        total_requests=k6_result.metrics.total_requests,
                        successful_requests=k6_result.metrics.total_requests - k6_result.metrics.failed_requests,
                        failed_requests=k6_result.metrics.failed_requests,
                        error_rate=k6_result.metrics.error_rate,
                        latency_p50_ms=k6_result.metrics.latency_p50,
                        latency_p95_ms=k6_result.metrics.latency_p95,
                        latency_p99_ms=k6_result.metrics.latency_p99,
                        throughput_rps=k6_result.metrics.requests_per_second,
                        raw_output=k6_result.raw_stdout[:1000] if k6_result.raw_stdout else "",
                    )
                    load_results.append(load_result)
            else:
                # Default behavior: execute predefined scripts
                if self.verbose:
                    print(f"  Running {len(self._generated_scripts)} tests...")
                
                k6_results = self.k6_executor.execute_all(
                    self._generated_scripts,
                    timeout_per_test=300,
                    verbose=self.verbose,
                )
                
                # Convert K6Result to LoadTestResult
                for k6_result in k6_results:
                    load_result = LoadTestResult(
                        test_type=k6_result.test_type,
                        vus=k6_result.metrics.vus_max,
                        duration=f"{k6_result.duration_seconds:.0f}s",
                        total_requests=k6_result.metrics.total_requests,
                        successful_requests=k6_result.metrics.total_requests - k6_result.metrics.failed_requests,
                        failed_requests=k6_result.metrics.failed_requests,
                        error_rate=k6_result.metrics.error_rate,
                        latency_p50_ms=k6_result.metrics.latency_p50,
                        latency_p95_ms=k6_result.metrics.latency_p95,
                        latency_p99_ms=k6_result.metrics.latency_p99,
                        throughput_rps=k6_result.metrics.requests_per_second,
                        raw_output=k6_result.raw_stdout[:1000] if k6_result.raw_stdout else "",
                    )
                    load_results.append(load_result)
            
            state["load_results"] = load_results
            
            if self.verbose:
                print(f"  Completed {len(load_results)} tests")
        else:
            # k6 not available - create mock results
            if self.verbose:
                if not k6_available:
                    print("  ⚠ k6 not available - using mock results")
                    print("  Install k6: https://k6.io/docs/getting-started/installation/")
                else:
                    print("  ⚠ No test scripts generated - using mock results")
            
            load_results = []
            for test in state.get("generated_tests", []):
                # Mock result based on baseline
                vus = test.get("vus", 10)
                result = LoadTestResult(
                    test_type=test.get("type", "unknown"),
                    vus=vus,
                    duration="30s",
                    total_requests=vus * 100,
                    successful_requests=vus * 95,
                    failed_requests=vus * 5,
                    error_rate=0.05,
                    latency_p50_ms=self._baseline.global_latency_p50 if self._baseline else 50,
                    latency_p95_ms=self._baseline.global_latency_p95 if self._baseline else 200,
                    latency_p99_ms=self._baseline.global_latency_p99 if self._baseline else 500,
                    throughput_rps=vus * 10,
                    raw_output="[MOCK] k6 not available",
                )
                load_results.append(result)
            
            state["load_results"] = load_results
        
        # Check infrastructure saturation after tests
        post_test_infra = check_infra_saturation()
        
        # Collect infra warnings
        infra_warnings = []
        if pre_test_infra.is_saturated:
            infra_warnings.extend(pre_test_infra.warnings)
        if post_test_infra.is_saturated:
            for w in post_test_infra.warnings:
                if w not in infra_warnings:
                    infra_warnings.append(w)
        
        # Store infra saturation data in state
        state["infra_saturation"] = {
            "pre_test": {
                "cpu_percent": pre_test_infra.cpu_percent,
                "memory_percent": pre_test_infra.memory_percent,
                "saturated": pre_test_infra.is_saturated,
            },
            "post_test": {
                "cpu_percent": post_test_infra.cpu_percent,
                "memory_percent": post_test_infra.memory_percent,
                "saturated": post_test_infra.is_saturated,
            },
            "warnings": infra_warnings,
            "confidence_penalty": max(pre_test_infra.confidence_penalty, post_test_infra.confidence_penalty),
        }
        
        if self.verbose and infra_warnings:
            print(f"  ⚠ Infrastructure warnings: {len(infra_warnings)}")
            for w in infra_warnings[:2]:
                print(f"    {w}")
        
        return state
    
    def _node_results_collection(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Collect and aggregate test results"""
        state["phase"] = AgentPhase.RESULTS_COLLECTION.value
        
        if self.verbose:
            print("[4/8] Collecting results...")
        
        # Results are already collected in load_execution
        # This phase would aggregate across multiple test runs
        
        return state
    
    def _node_breaking_point_detection(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect the first breaking point using rules-only analysis.
        
        Uses:
        - Error rate threshold breach (sustained)
        - Latency p95/p99 degradation (slope-based)
        - Throughput plateau or drop despite rising VUs
        - Saturation indicators
        """
        state["phase"] = AgentPhase.BREAKING_POINT_DETECTION.value
        
        if self.verbose:
            print("[5/8] Detecting breaking point...")
        
        load_results = state.get("load_results", [])
        
        if not load_results:
            if self.verbose:
                print("  No load results to analyze")
            state["breaking_point"] = None
            state["failure_category"] = FailureCategory.NO_FAILURE.value
            state["failure_timeline"] = []
            return state
        
        # Initialize detector
        detector = BreakingPointDetector(self.config.load)
        
        # Run detection
        result: BreakingPointResult = detector.detect(load_results)
        
        # Store results in state
        state["breaking_point"] = result.breaking_point
        state["breaking_point_result"] = result.to_dict()
        state["failure_timeline"] = result.timeline.to_list()
        state["failure_category"] = result.primary_category.value
        
        if self.verbose:
            if result.detected:
                bp = result.breaking_point
                print(f"  ✗ Breaking point DETECTED at {bp.vus_at_break} VUs")
                print(f"    Type: {bp.failure_type}")
                print(f"    Category: {result.primary_category.value}")
                print(f"    Confidence: {result.category_confidence:.0%}")
                print(f"  Timeline ({len(result.timeline.events)} events):")
                for event in result.timeline.events[:5]:
                    print(f"    {event.timestamp}: {event.description}")
                if len(result.timeline.events) > 5:
                    print(f"    ... and {len(result.timeline.events) - 5} more events")
            else:
                print("  ✓ No breaking point detected")
                print(f"    Category: {result.primary_category.value}")
                print(f"    Violations: {len(result.violations)}")
        
        # Run destructive tests if enabled and breaking point detected
        if self.config.load.destructive_enabled:
            self._run_destructive_tests(state, result)
        
        return state
    
    def _run_destructive_tests(
        self,
        state: Dict[str, Any],
        bp_result,
    ) -> None:
        """Run optional destructive test scenarios"""
        if not self.k6_executor.check_available():
            if self.verbose:
                print("  ⚠ k6 not available - skipping destructive tests")
            return
        
        if self.verbose:
            print("  Running destructive tests...")
        
        load_results = state.get("load_results", [])
        endpoints = self._get_test_endpoints()
        
        # Sustained test
        sustained_vus = self.config.load.sustained_test_vus
        if sustained_vus == 0:
            # Auto-calculate from baseline (50% of breaking point or initial)
            if bp_result.detected:
                sustained_vus = max(1, bp_result.breaking_point.vus_at_break // 2)
            else:
                sustained_vus = self.config.load.initial_vus * 2
        
        if self.verbose:
            print(f"    Sustained test: {sustained_vus} VUs for {self.config.load.sustained_test_duration}")
        
        sustained_script = self._test_generator.generate_sustained_test(
            endpoints=endpoints,
            target_vus=sustained_vus,
            duration=self.config.load.sustained_test_duration,
        )
        
        sustained_results = self.k6_executor.execute_all(
            [sustained_script],
            timeout_per_test=600,
            verbose=False,
        )
        
        for k6_result in sustained_results:
            if not k6_result.success:
                if self.verbose:
                    print(f"      ⚠ Sustained test failed: {k6_result.raw_stderr[:100] if k6_result.raw_stderr else 'unknown error'}")
                continue
            load_result = LoadTestResult(
                test_type=k6_result.test_type,
                vus=k6_result.metrics.vus_max,
                duration=f"{k6_result.duration_seconds:.0f}s",
                total_requests=k6_result.metrics.total_requests,
                successful_requests=k6_result.metrics.total_requests - k6_result.metrics.failed_requests,
                failed_requests=k6_result.metrics.failed_requests,
                error_rate=k6_result.metrics.error_rate,
                latency_p50_ms=k6_result.metrics.latency_p50,
                latency_p95_ms=k6_result.metrics.latency_p95,
                latency_p99_ms=k6_result.metrics.latency_p99,
                throughput_rps=k6_result.metrics.requests_per_second,
                raw_output="",
            )
            load_results.append(load_result)
            if self.verbose:
                print(f"      ✓ Sustained: {load_result.error_rate:.1%} errors, P95: {load_result.latency_p95_ms:.0f}ms")
        
        # Recovery test (only if breaking point detected and enabled)
        if self.config.load.recovery_test_enabled and bp_result.detected:
            breaking_vus = bp_result.breaking_point.vus_at_break
            
            if self.verbose:
                print(f"    Recovery test: push to {int(breaking_vus * 1.2)} VUs, then recover")
            
            recovery_script = self._test_generator.generate_recovery_test(
                endpoints=endpoints,
                breaking_vus=breaking_vus,
                baseline_vus=self.config.load.initial_vus,
            )
            
            recovery_results = self.k6_executor.execute_all(
                [recovery_script],
                timeout_per_test=300,
                verbose=False,
            )
            
            for k6_result in recovery_results:
                if not k6_result.success:
                    if self.verbose:
                        print(f"      ⚠ Recovery test failed: {k6_result.raw_stderr[:100] if k6_result.raw_stderr else 'unknown error'}")
                    continue
                load_result = LoadTestResult(
                    test_type=k6_result.test_type,
                    vus=k6_result.metrics.vus_max,
                    duration=f"{k6_result.duration_seconds:.0f}s",
                    total_requests=k6_result.metrics.total_requests,
                    successful_requests=k6_result.metrics.total_requests - k6_result.metrics.failed_requests,
                    failed_requests=k6_result.metrics.failed_requests,
                    error_rate=k6_result.metrics.error_rate,
                    latency_p50_ms=k6_result.metrics.latency_p50,
                    latency_p95_ms=k6_result.metrics.latency_p95,
                    latency_p99_ms=k6_result.metrics.latency_p99,
                    throughput_rps=k6_result.metrics.requests_per_second,
                    raw_output="",
                )
                load_results.append(load_result)
                if self.verbose:
                    print(f"      ✓ Recovery: {load_result.error_rate:.1%} errors, P95: {load_result.latency_p95_ms:.0f}ms")
        
        state["load_results"] = load_results
    
    def _get_test_endpoints(self) -> List[Dict[str, Any]]:
        """Get endpoints for testing"""
        if self._baseline and self._baseline.top_endpoints:
            return [
                {"path": ep.path, "method": ep.method, "weight": ep.request_count}
                for ep in self._baseline.top_endpoints
            ]
        elif self.config.target.endpoints:
            return [
                {"path": ep, "method": "GET", "weight": 1}
                for ep in self.config.target.endpoints
            ]
        return [{"path": "/", "method": "GET", "weight": 1}]
    
    def _node_root_cause_analysis(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform LLM-assisted root cause analysis.
        
        Phase 5 Scope:
        - EXPLAIN why the classification makes sense
        - CONNECT timeline events causally
        - TRANSLATE signals into human reasoning
        - ASSIGN explanation confidence
        
        NOT allowed:
        - Change classification
        - Invent metrics
        - Override breaking point
        - Suggest fixes (Phase 6)
        """
        state["phase"] = AgentPhase.ROOT_CAUSE_ANALYSIS.value
        
        if self.verbose:
            print("[6/8] Analyzing root cause...")
        
        # Get deterministic analysis data
        breaking_point = state.get("breaking_point")
        failure_category = state.get("failure_category", FailureCategory.NO_FAILURE.value)
        timeline = state.get("failure_timeline", [])
        bp_result = state.get("breaking_point_result", {})
        load_results = state.get("load_results", [])
        
        # Store deterministic failure analysis first
        evidence = self._collect_evidence(state)
        state["failure_analysis"] = DeterministicFailureAnalysis(
            failure_category=failure_category,
            category_confidence=bp_result.get("category_confidence", 0.5),
            classification_rationale=bp_result.get("classification_rationale", []),
            supporting_evidence=evidence,
        )
        
        # Build baseline summary for LLM input
        baseline_summary = {}
        baseline_confidence = 0.0
        if self._baseline:
            baseline_summary = {
                "total_requests": self._baseline.total_requests,
                "avg_rps": self._baseline.global_avg_rps,
                "peak_rps": self._baseline.global_peak_rps,
                "error_rate": self._baseline.global_error_rate,
                "latency_p50_ms": self._baseline.global_latency_p50,
                "latency_p95_ms": self._baseline.global_latency_p95,
                "top_endpoints_count": len(self._baseline.top_endpoints),
            }
            if self._baseline.confidence:
                baseline_confidence = self._baseline.confidence.score
        
        # Build breaking point dict for LLM input
        breaking_point_dict = None
        if breaking_point:
            breaking_point_dict = {
                "vus_at_break": breaking_point.vus_at_break,
                "rps_at_break": breaking_point.rps_at_break,
                "failure_type": breaking_point.failure_type,
                "threshold_exceeded": breaking_point.threshold_exceeded,
                "observed_value": breaking_point.observed_value,
                "threshold_value": breaking_point.threshold_value,
                "confidence": breaking_point.confidence,
            }
        
        # Build observed metrics for LLM input
        observed_metrics = {}
        if load_results:
            observed_metrics = {
                "total_requests": sum(r.total_requests for r in load_results),
                "total_errors": sum(r.failed_requests for r in load_results),
                "avg_error_rate": sum(r.error_rate for r in load_results) / len(load_results),
                "max_p95_latency_ms": max(r.latency_p95_ms for r in load_results),
                "max_vus": max(r.vus for r in load_results),
                "test_types": list(set(r.test_type for r in load_results)),
            }
        
        # Run root cause analysis (LLM or rules-based)
        from sentinelperf.analysis.root_cause import run_root_cause_analysis
        
        llm_output, mode, model, latency_ms = run_root_cause_analysis(
            config=self.config.llm,
            baseline_summary=baseline_summary,
            baseline_confidence=baseline_confidence,
            breaking_point=breaking_point_dict,
            failure_timeline=timeline,
            failure_classification=failure_category,
            classification_rationale=bp_result.get("classification_rationale", []),
            observed_metrics=observed_metrics,
            verbose=self.verbose,
        )
        
        # Store root cause analysis result
        # Apply infra saturation penalty to confidence
        infra_saturation = state.get("infra_saturation", {})
        confidence_penalty = infra_saturation.get("confidence_penalty", 0.0)
        adjusted_confidence = max(0.1, llm_output.confidence - confidence_penalty)
        
        # Add infra warnings to limitations
        limitations = list(llm_output.limitations)
        if infra_saturation.get("warnings"):
            limitations.append("Results may be infra-limited (high resource usage detected)")
        
        state["root_cause"] = RootCauseAnalysis(
            root_cause_summary=llm_output.root_cause_summary,
            primary_cause=llm_output.primary_cause,
            contributing_factors=llm_output.contributing_factors,
            confidence=adjusted_confidence,
            assumptions=llm_output.assumptions,
            limitations=limitations,
            failure_pattern=llm_output.failure_pattern,
            pattern_explanation=llm_output.pattern_explanation,
            llm_mode=mode,
            llm_model=model,
            llm_latency_ms=latency_ms,
        )
        
        if self.verbose:
            print(f"  Mode: {mode}" + (f" ({model})" if model else ""))
            print(f"  Primary cause: {llm_output.primary_cause}")
            print(f"  Pattern: {llm_output.failure_pattern}")
            print(f"  Confidence: {llm_output.confidence:.0%}")
            if llm_output.limitations:
                print(f"  Limitations: {len(llm_output.limitations)}")
        
        return state
    
    def _collect_evidence(self, state: Dict[str, Any]) -> List[str]:
        """Collect supporting evidence from state"""
        evidence = []
        
        # From baseline
        if self._baseline:
            evidence.append(f"Baseline: {self._baseline.total_requests} requests analyzed")
            evidence.append(f"Baseline P95: {self._baseline.global_latency_p95:.1f}ms")
            if self._baseline.confidence:
                evidence.append(f"Baseline confidence: {self._baseline.confidence.level}")
        
        # From load results
        load_results = state.get("load_results", [])
        if load_results:
            total_reqs = sum(r.total_requests for r in load_results)
            avg_error = sum(r.error_rate for r in load_results) / len(load_results)
            max_p95 = max(r.latency_p95_ms for r in load_results)
            evidence.append(f"Load tests: {total_reqs} total requests")
            evidence.append(f"Average error rate: {avg_error:.1%}")
            evidence.append(f"Max P95 latency: {max_p95:.1f}ms")
        
        # From breaking point
        breaking_point = state.get("breaking_point")
        if breaking_point:
            evidence.extend(breaking_point.signals[:3])  # Top 3 signals
        
        return evidence
    
    def _node_recommendations(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate recommendations based on failure classification.
        
        Phase 6 Scope:
        - Deterministic recommendations from templates
        - Optional LLM polishing (rephrasing only)
        
        NOT allowed:
        - Auto-fix
        - Infra changes
        - Scaling actions
        - Code modification
        """
        state["phase"] = AgentPhase.RECOMMENDATIONS.value
        
        if self.verbose:
            print("[7/8] Generating recommendations...")
        
        # Check if recommendations are enabled
        rec_config = self.config.recommendations
        if not rec_config.enabled:
            if self.verbose:
                print("  Recommendations disabled in config")
            state["recommendations"] = {
                "recommendations": [],
                "limitations": ["Recommendations disabled in configuration"],
                "polished_by_llm": False,
            }
            return state
        
        # Get failure classification
        failure_category = state.get("failure_category", "no_failure")
        
        # Get breaking point for context
        breaking_point = state.get("breaking_point")
        breaking_point_dict = None
        if breaking_point:
            breaking_point_dict = {
                "vus_at_break": breaking_point.vus_at_break,
                "rps_at_break": breaking_point.rps_at_break,
                "failure_type": breaking_point.failure_type,
                "confidence": breaking_point.confidence,
            }
        
        # Get root cause confidence
        root_cause = state.get("root_cause")
        root_cause_confidence = root_cause.confidence if root_cause else 0.5
        
        # Generate recommendations
        from sentinelperf.analysis.recommendations import generate_recommendations
        
        result = generate_recommendations(
            failure_classification=failure_category,
            breaking_point=breaking_point_dict,
            root_cause_confidence=root_cause_confidence,
            llm_config=self.config.llm if rec_config.polish_with_llm else None,
            polish_with_llm=rec_config.polish_with_llm,
            verbose=self.verbose,
        )
        
        # Limit number of recommendations
        result.recommendations = result.recommendations[:rec_config.max_recommendations]
        
        # Store result
        state["recommendations"] = result.to_dict()
        
        if self.verbose:
            print(f"  Generated {len(result.recommendations)} recommendations")
            if result.polished_by_llm:
                print(f"  Polished by LLM ({result.llm_model})")
            for i, rec in enumerate(result.recommendations, 1):
                print(f"    {i}. {rec.action[:60]}...")
        
        return state
    
    def _node_report_generation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate output reports"""
        state["phase"] = AgentPhase.REPORT_GENERATION.value
        
        if self.verbose:
            print("[8/8] Generating reports...")
        
        # Convert to AgentState for report generation
        agent_state = self._dict_to_agent_state(state)
        
        # Generate reports using report modules
        from sentinelperf.reports.markdown import MarkdownReporter
        from sentinelperf.reports.json_report import JSONReporter
        
        # Create intermediate result for report generation
        intermediate_result = ExecutionResult(
            success=True,
            state=agent_state,
            summary="",
        )
        
        # Generate markdown report
        md_reporter = MarkdownReporter(self.output_dir)
        md_path = md_reporter.generate(intermediate_result)
        
        # Generate JSON report
        json_reporter = JSONReporter(self.output_dir)
        json_path = json_reporter.generate(intermediate_result)
        
        if self.verbose:
            print(f"  Markdown report: {md_path}")
            print(f"  JSON report: {json_path}")
        
        # Mark complete
        state["phase"] = AgentPhase.COMPLETE.value
        state["completed_at"] = datetime.utcnow().isoformat()
        
        return state
    
    def _generate_summary(self, state: AgentState) -> str:
        """Generate execution summary"""
        if state.phase == AgentPhase.COMPLETE:
            summary_parts = [f"Analysis complete for {state.target_url}"]
            
            if self._baseline:
                summary_parts.append(
                    f"Baseline: {self._baseline.global_avg_rps:.1f} RPS, "
                    f"P95: {self._baseline.global_latency_p95:.0f}ms"
                )
            
            return " | ".join(summary_parts)
            
        elif state.phase == AgentPhase.ERROR:
            return f"Analysis failed: {', '.join(state.errors)}"
        else:
            return f"Analysis interrupted at phase: {state.phase.value}"
