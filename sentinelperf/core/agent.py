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
    RootCauseAnalysis,
)
from sentinelperf.telemetry.otel import OpenTelemetrySource
from sentinelperf.telemetry.logs import AccessLogSource
from sentinelperf.telemetry.prometheus import PrometheusSource
from sentinelperf.telemetry.baseline import BaselineInference, BaselineBehavior
from sentinelperf.load.generator import TestGenerator, TestScript
from sentinelperf.load.k6_executor import K6Executor, K6Result


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
    root_cause: Optional[RootCauseAnalysis]
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
        workflow.add_node("report_generation", self._node_report_generation)
        
        # Define edges (workflow sequence)
        workflow.set_entry_point("telemetry_analysis")
        workflow.add_edge("telemetry_analysis", "test_generation")
        workflow.add_edge("test_generation", "load_execution")
        workflow.add_edge("load_execution", "results_collection")
        workflow.add_edge("results_collection", "breaking_point_detection")
        workflow.add_edge("breaking_point_detection", "root_cause_analysis")
        workflow.add_edge("root_cause_analysis", "report_generation")
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
            root_cause=d.get("root_cause"),
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
            print("[1/7] Analyzing telemetry...")
        
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
        Generate load, stress, and spike test configurations.
        
        Uses baseline behavior to inform test parameters.
        """
        state["phase"] = AgentPhase.TEST_GENERATION.value
        
        if self.verbose:
            print("[2/7] Generating test configurations...")
        
        # Get baseline data for load planning
        load_plan_input = self._get_load_plan_input()
        
        # Get auth headers from config
        auth_headers = self._get_auth_headers()
        
        # Initialize test generator
        generator = TestGenerator(
            base_url=self.config.target.base_url,
            auth_headers=auth_headers,
        )
        
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
        
        # Get recommended VUs from baseline
        initial_vus = load_plan_input.get("recommended_initial_vus", self.config.load.initial_vus)
        max_vus = load_plan_input.get("recommended_max_vus", self.config.load.max_vus)
        
        # Generate tests based on baseline
        baseline_metrics = load_plan_input.get("baseline", {})
        
        if self.verbose:
            print(f"  Endpoints to test: {len(endpoints)}")
            print(f"  Initial VUs: {initial_vus}, Max VUs: {max_vus}")
            print(f"  Baseline P95: {baseline_metrics.get('latency_p95_ms', 'N/A')}ms")
        
        # Generate load test (baseline capacity)
        load_test = generator.generate_load_test(
            endpoints=endpoints,
            initial_vus=1,
            target_vus=initial_vus,
            ramp_duration=self.config.load.ramp_duration,
            hold_duration=self.config.load.hold_duration,
            error_threshold=self.config.load.error_rate_threshold,
            p95_threshold_ms=self.config.load.p95_latency_threshold_ms,
        )
        
        # Generate stress test (breaking point discovery)
        stress_test = generator.generate_stress_test(
            endpoints=endpoints,
            max_vus=max_vus,
            step_duration="30s",
            steps=5,
        )
        
        # Generate spike test (burst handling)
        spike_test = generator.generate_spike_test(
            endpoints=endpoints,
            baseline_vus=initial_vus,
            spike_vus=max_vus,
            spike_duration="30s",
        )
        
        # Generate adaptive test (fine-grained breaking point)
        adaptive_test = generator.generate_adaptive_test(
            endpoints=endpoints,
            initial_vus=1,
            max_vus=max_vus,
            step_vus=self.config.load.adaptive_step,
            step_duration="60s",
        )
        
        # Store generated tests
        state["generated_tests"] = [
            {
                "type": "load",
                "name": load_test.name,
                "vus": initial_vus,
                "duration": self.config.load.hold_duration,
                "script": load_test,
                "endpoints": endpoints,
                "baseline_driven": True,
            },
            {
                "type": "stress",
                "name": stress_test.name,
                "vus": max_vus,
                "duration": "180s",  # 5 steps * 30s + hold + ramp
                "script": stress_test,
                "endpoints": endpoints,
                "baseline_driven": True,
            },
            {
                "type": "spike",
                "name": spike_test.name,
                "vus": max_vus,
                "duration": "120s",
                "script": spike_test,
                "endpoints": endpoints,
                "baseline_driven": True,
            },
            {
                "type": "adaptive",
                "name": adaptive_test.name,
                "vus": max_vus,
                "duration": f"{(max_vus // self.config.load.adaptive_step) * 60}s",
                "script": adaptive_test,
                "endpoints": endpoints,
                "baseline_driven": True,
            },
        ]
        
        if self.verbose:
            print(f"  Generated {len(state['generated_tests'])} test configurations")
            for test in state["generated_tests"]:
                print(f"    - {test['type']}: {test['vus']} VUs, {test['duration']}")
        
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
        
        Currently mocked - will be implemented in next phase.
        """
        state["phase"] = AgentPhase.LOAD_EXECUTION.value
        
        if self.verbose:
            print("[3/7] Executing load tests...")
            print("  (Load execution is mocked - k6 integration pending)")
        
        # Create mocked results based on generated tests
        load_results = []
        for test in state["generated_tests"]:
            # Mock result that simulates data flow
            result = LoadTestResult(
                test_type=test["type"],
                vus=test["vus"],
                duration=test["duration"],
                # Mock metrics based on baseline if available
                total_requests=test["vus"] * 100,  # Simulated
                successful_requests=test["vus"] * 95,
                failed_requests=test["vus"] * 5,
                error_rate=0.05,
                latency_p50_ms=self._baseline.global_latency_p50 if self._baseline else 50,
                latency_p95_ms=self._baseline.global_latency_p95 if self._baseline else 200,
                latency_p99_ms=self._baseline.global_latency_p99 if self._baseline else 500,
                throughput_rps=test["vus"] * 10,  # Simulated
            )
            load_results.append(result)
        
        state["load_results"] = load_results
        
        if self.verbose:
            print(f"  Created {len(load_results)} mocked test results")
        
        return state
    
    def _node_results_collection(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Collect and aggregate test results"""
        state["phase"] = AgentPhase.RESULTS_COLLECTION.value
        
        if self.verbose:
            print("[4/7] Collecting results...")
        
        # Results are already collected in load_execution
        # This phase would aggregate across multiple test runs
        
        return state
    
    def _node_breaking_point_detection(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Detect the first breaking point"""
        state["phase"] = AgentPhase.BREAKING_POINT_DETECTION.value
        
        if self.verbose:
            print("[5/7] Detecting breaking point...")
        
        # Since load execution is mocked, create placeholder breaking point
        # In real implementation, this would analyze actual test results
        state["breaking_point"] = BreakingPoint(
            vus_at_break=0,
            rps_at_break=0.0,
            failure_type="none",
            threshold_exceeded="none",
            observed_value=0.0,
            threshold_value=0.0,
            confidence=0.0,
            signals=["Load execution mocked - no real breaking point detected"],
        )
        
        return state
    
    def _node_root_cause_analysis(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze root cause of failures using LLM"""
        state["phase"] = AgentPhase.ROOT_CAUSE_ANALYSIS.value
        
        if self.verbose:
            print("[6/7] Analyzing root cause...")
        
        # Since no real breaking point, provide baseline summary
        state["root_cause"] = RootCauseAnalysis(
            primary_cause="No breaking point detected (load execution mocked)",
            confidence=0.0,
            supporting_evidence=[
                f"Baseline analyzed: {self._baseline.total_requests if self._baseline else 0} requests",
                f"Top endpoints: {len(self._baseline.top_endpoints) if self._baseline else 0}",
            ],
            reasoning_steps=[
                "Step 1: Telemetry data ingested and analyzed",
                "Step 2: Baseline behavior inferred from historical data",
                "Step 3: Load tests generated based on baseline",
                "Step 4: Load execution pending - results mocked",
            ],
            recommendations=[
                {
                    "action": "Run with real k6 execution to detect actual breaking point",
                    "confidence": 1.0,
                    "priority": 1,
                    "estimated_impact": "High - enables actual performance analysis",
                }
            ],
            llm_mode=self.llm_mode,
        )
        
        return state
    
    def _node_report_generation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate output reports"""
        state["phase"] = AgentPhase.REPORT_GENERATION.value
        
        if self.verbose:
            print("[7/7] Generating reports...")
        
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
