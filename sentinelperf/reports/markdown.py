"""Markdown report generator for SentinelPerf"""

from pathlib import Path
from datetime import datetime
from typing import Optional

from sentinelperf.core.state import ExecutionResult, AgentState


class MarkdownReporter:
    """
    Generates authoritative Markdown report.
    
    This is the primary output format for SentinelPerf analysis.
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(self, result: ExecutionResult) -> Path:
        """Generate Markdown report and return file path"""
        
        state = result.state
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"sentinelperf_report_{timestamp}.md"
        filepath = self.output_dir / filename
        
        content = self._build_report(state, result)
        
        with open(filepath, "w") as f:
            f.write(content)
        
        return filepath
    
    def _build_report(self, state: AgentState, result: ExecutionResult) -> str:
        """Build complete Markdown report"""
        
        # Report order per spec:
        # 1. Executive Summary
        # 2. Test Case Summary ← new
        # 3. Test Case Coverage Summary ← fixed
        # 4. API & Backend Trigger Summary ← new
        # 5. Breaking Point Analysis
        # 6. Failure Timeline (included in breaking_point_section)
        # 7. Root Cause Analysis
        # 8. Recommendations
        # 9. Load Test Results
        # 10. Infrastructure Saturation (if exists)
        # 11. Telemetry Analysis
        # 12. Methodology
        # 13. Appendix
        sections = [
            self._header(state),
            self._executive_summary(state, result),
            self._test_case_summary_section(state),
            self._test_case_coverage_summary_section(state),
            self._api_trigger_summary_section(state),
            self._breaking_point_section(state),
            self._root_cause_section(state),
            self._recommendations_section(state),
            self._load_test_results_section(state),
            self._infra_saturation_section(state),
            self._telemetry_section(state),
            self._methodology_section(),
            self._footer(state),
        ]
        
        return "\n\n".join(filter(None, sections))
    
    def _header(self, state: AgentState) -> str:
        """Report header"""
        return f"""# SentinelPerf Analysis Report

**Target:** {state.target_url}  
**Environment:** {state.environment}  
**Generated:** {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}  
**Status:** {'✅ Complete' if state.phase.value == 'complete' else '⚠️ ' + state.phase.value}

---"""

    def _executive_summary(self, state: AgentState, result: ExecutionResult) -> str:
        """Executive summary section"""
        summary_lines = ["## Executive Summary", ""]
        
        if state.breaking_point and state.breaking_point.vus_at_break > 0:
            bp = state.breaking_point
            summary_lines.append(
                f"The system reached its **breaking point at {bp.vus_at_break} virtual users** "
                f"({bp.rps_at_break:.1f} requests/second), where {bp.failure_type} exceeded "
                f"acceptable thresholds."
            )
        else:
            summary_lines.append(
                "No breaking point was detected within the tested load range. "
                "The system handled all test scenarios within acceptable thresholds."
            )
        
        summary_lines.append("")
        
        if state.root_cause and state.root_cause.primary_cause:
            rc = state.root_cause
            summary_lines.append(
                f"**Primary Root Cause:** {rc.primary_cause} "
                f"(confidence: {rc.confidence:.0%})"
            )
        
        return "\n".join(summary_lines)
    
    def _infra_saturation_section(self, state: AgentState) -> str:
        """Infrastructure Saturation section - only rendered if data exists"""
        if not state.infra_saturation:
            return ""
        
        infra = state.infra_saturation
        pre = infra.get("pre_test", {})
        post = infra.get("post_test", {})
        warnings = infra.get("warnings", [])
        confidence_penalty = infra.get("confidence_penalty", 0)
        
        lines = [
            "## Infrastructure Saturation",
            "",
            "### Pre-test vs Post-test Resource Usage",
            "",
            "| Phase | CPU | Memory |",
            "|-------|-----|--------|",
            f"| Pre-test | {pre.get('cpu_percent', 0):.1f}% | {pre.get('memory_percent', 0):.1f}% |",
            f"| Post-test | {post.get('cpu_percent', 0):.1f}% | {post.get('memory_percent', 0):.1f}% |",
        ]
        
        if warnings:
            lines.extend([
                "",
                "### Warnings",
                "",
            ])
            for w in warnings:
                lines.append(f"- ⚠️ {w}")
        
        if confidence_penalty > 0:
            lines.append("")
            lines.append(f"**Confidence penalty applied:** -{confidence_penalty*100:.0f}%")
        
        return "\n".join(lines)
    
    def _breaking_point_section(self, state: AgentState) -> str:
        """Breaking point details with classification and timeline"""
        sections = []
        
        # Classification header
        category = state.failure_category or "no_failure"
        category_display = category.replace("_", " ").title()
        
        if not state.breaking_point or state.breaking_point.vus_at_break == 0:
            sections.append(f"""## Breaking Point Analysis

**Classification:** {category_display}

No breaking point detected within tested parameters.""")
            return "\n".join(sections)
        
        bp = state.breaking_point
        
        # Main breaking point section
        sections.append(f"""## Breaking Point Analysis

**Classification:** {category_display}

| Metric | Value |
|--------|-------|
| Virtual Users at Break | {bp.vus_at_break} |
| Requests/Second at Break | {bp.rps_at_break:.1f} |
| Failure Type | {bp.failure_type} |
| Threshold Exceeded | {bp.threshold_exceeded} |
| Observed Value | {bp.observed_value:.4f} |
| Threshold Value | {bp.threshold_value:.4f} |
| Detection Confidence | {bp.confidence:.0%} |

### Observed Signals

{chr(10).join(['- ' + s for s in bp.signals]) if bp.signals else '- No additional signals recorded'}""")
        
        # Add failure timeline if available
        if state.failure_timeline:
            timeline_lines = ["", "### Failure Timeline", ""]
            timeline_lines.append("| Time | Event | Test Type | VUs | Description |")
            timeline_lines.append("|------|-------|-----------|-----|-------------|")
            
            for event in state.failure_timeline:
                event_type = event.get("event_type", "unknown")
                test_type = event.get("test_type", "-")
                vus = event.get("vus", "-")
                description = event.get("description", "-")
                timestamp = event.get("timestamp", "-")
                
                # Add emoji based on event type
                type_emoji = {
                    "load_change": "📈",
                    "error_rate_breach": "🔴",
                    "latency_degradation": "🟠",
                    "throughput_plateau": "🟡",
                    "saturation": "⚠️",
                }.get(event_type, "•")
                
                timeline_lines.append(
                    f"| {timestamp} | {type_emoji} {event_type} | {test_type} | {vus} | {description[:50]}{'...' if len(description) > 50 else ''} |"
                )
            
            sections.append("\n".join(timeline_lines))
        
        return "\n".join(sections)

    def _root_cause_section(self, state: AgentState) -> str:
        """Root cause analysis details"""
        if not state.root_cause:
            return """## Root Cause Analysis

Root cause analysis was not performed."""
        
        rc = state.root_cause
        
        # Contributing factors
        factors = "\n".join([f"- {f}" for f in rc.contributing_factors]) \
            if rc.contributing_factors else "- None identified"
        
        # Assumptions
        assumptions = "\n".join([f"- {a}" for a in rc.assumptions]) \
            if rc.assumptions else "- None"
        
        # Limitations
        limitations = "\n".join([f"- {lim}" for lim in rc.limitations]) \
            if rc.limitations else "- None"
        
        # Model info
        model_info = f"**Model:** {rc.llm_model}" if rc.llm_model else ""
        latency_info = f" ({rc.llm_latency_ms:.0f}ms)" if rc.llm_latency_ms > 0 else ""
        
        # Pattern section
        pattern_section = ""
        if rc.failure_pattern and rc.pattern_explanation:
            pattern_display = rc.failure_pattern.replace("_", " ").title()
            pattern_section = f"""

### Failure Pattern

**Detected Pattern:** {pattern_display}

{rc.pattern_explanation}"""
        
        return f"""## Root Cause Analysis

**Analysis Mode:** {rc.llm_mode}{latency_info}  
{model_info}
**Confidence:** {rc.confidence:.0%}

### Summary

{rc.root_cause_summary}

### Primary Cause

{rc.primary_cause}{pattern_section}

### Contributing Factors

{factors}

### Assumptions

{assumptions}

### Limitations

{limitations}"""

    def _recommendations_section(self, state: AgentState) -> str:
        """Recommendations section"""
        if not state.recommendations:
            return """## Recommendations

No recommendations generated."""
        
        recs_data = state.recommendations
        recs = recs_data.get("recommendations", [])
        limitations = recs_data.get("limitations", [])
        polished = recs_data.get("polished_by_llm", False)
        
        if not recs:
            return """## Recommendations

No recommendations available for this classification."""
        
        # Header
        polish_note = " *(polished by LLM)*" if polished else ""
        sections = [f"## Recommendations{polish_note}", ""]
        
        # Recommendations
        for i, rec in enumerate(recs, 1):
            risk = rec.get("risk", "MEDIUM")
            confidence = rec.get("confidence", 0)
            priority = rec.get("priority", 5)
            
            risk_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(risk, "⚪")
            priority_emoji = "🔴" if priority == 1 else "🟡" if priority <= 3 else "🟢"
            
            sections.append(f"""### {i}. {rec.get('action', 'Unknown action')}

**Priority:** {priority_emoji} P{priority}  
**Risk:** {risk_emoji} {risk}  
**Confidence:** {confidence:.0%}

**Rationale:** {rec.get('rationale', 'No rationale provided')}

**Expected Impact:** {rec.get('expected_impact', 'Unknown')}
""")
        
        # Limitations
        if limitations:
            sections.append("### Limitations")
            sections.append("")
            for lim in limitations:
                sections.append(f"- {lim}")
        
        return "\n".join(sections)

    def _load_test_results_section(self, state: AgentState) -> str:
        """Load test results table"""
        if not state.load_results:
            return """## Load Test Results

No load test results available."""
        
        header = """## Load Test Results

| Test Type | VUs | Duration | Requests | Error Rate | P95 Latency | Throughput |
|-----------|-----|----------|----------|------------|-------------|------------|"""
        
        rows = []
        for r in state.load_results:
            rows.append(
                f"| {r.test_type} | {r.vus} | {r.duration} | "
                f"{r.total_requests} | {r.error_rate:.2%} | "
                f"{r.latency_p95_ms:.0f}ms | {r.throughput_rps:.1f} RPS |"
            )
        
        return header + "\n" + "\n".join(rows)
    
    def _test_case_summary_section(self, state: AgentState) -> str:
        """Test Case Summary section"""
        if not state.load_results:
            return """## Test Case Summary

No test cases were executed."""
        
        # Define test type semantics
        test_semantics = {
            "baseline": ("Steady validation", "Constant load"),
            "stress": ("Incremental capacity test", "Ramping load"),
            "spike": ("Burst resilience test", "Sudden spike"),
            "adaptive": ("Adaptive capacity search", "Stepwise escalation"),
            "sustained": ("Long duration stability", "Sustained load"),
            "recovery": ("Recovery behavior test", "Overload then drop"),
        }
        
        lines = [
            "## Test Case Summary",
            "",
            "| Test Case | Purpose | Load Pattern | Max VUs | Duration |",
            "|-----------|---------|--------------|---------|----------|",
        ]
        
        # Group results by test type base name
        seen_types = set()
        for r in state.load_results:
            # Extract base test type (e.g., "adaptive_10vus" -> "adaptive")
            test_type_base = r.test_type.split("_")[0].lower()
            
            # Skip if we've already added this type
            if test_type_base in seen_types:
                # Update max VUs for this type
                continue
            
            # Find max VUs and duration for this test type
            type_results = [
                res for res in state.load_results 
                if res.test_type.split("_")[0].lower() == test_type_base
            ]
            max_vus = max(res.vus for res in type_results)
            
            # Get duration from first result of this type
            duration = type_results[0].duration if type_results else "N/A"
            
            # Get semantics
            purpose, pattern = test_semantics.get(test_type_base, ("Performance test", "Variable"))
            
            lines.append(f"| {test_type_base.capitalize()} | {purpose} | {pattern} | {max_vus} | {duration} |")
            seen_types.add(test_type_base)
        
        return "\n".join(lines)
    
    def _test_case_coverage_summary_section(self, state: AgentState) -> str:
        """Test Case Coverage Summary section"""
        lines = [
            "## Test Case Coverage Summary",
            "",
        ]
        
        # Determine which test types ran
        test_types_run = set()
        if state.load_results:
            for r in state.load_results:
                test_type_base = r.test_type.split("_")[0].lower()
                test_types_run.add(test_type_base)
        
        # A) Load Pattern Coverage
        lines.extend([
            "### A) Load Pattern Coverage",
            "",
            "| Pattern | Covered | Notes |",
            "|---------|---------|-------|",
        ])
        
        patterns = [
            ("Baseline (steady)", "baseline", "Validates normal operation"),
            ("Stress (incremental ramp)", "stress", "Tests gradual load increase"),
            ("Spike (sudden burst)", "spike", "Tests burst handling"),
            ("Soak (long duration)", "sustained", "Tests prolonged load stability"),
            ("Recovery after overload", "recovery", "Tests system resilience"),
        ]
        
        for pattern_name, test_key, note in patterns:
            covered = "Yes" if test_key in test_types_run else "No"
            notes = note if covered == "Yes" else "Not executed in this run"
            lines.append(f"| {pattern_name} | {covered} | {notes} |")
        
        # B) Failure Mode Coverage
        lines.extend([
            "",
            "### B) Failure Mode Coverage",
            "",
            "| Failure Mode | Observed | Notes |",
            "|--------------|----------|-------|",
        ])
        
        # Determine observed failures from timeline and breaking point
        observed_failures = set()
        if state.failure_timeline:
            for event in state.failure_timeline:
                event_type = event.get("event_type", "")
                if event_type == "error_rate_breach":
                    observed_failures.add("error_rate")
                elif event_type == "latency_degradation":
                    observed_failures.add("latency")
                elif event_type == "throughput_plateau":
                    observed_failures.add("throughput")
                elif event_type == "saturation":
                    observed_failures.add("saturation")
        
        # Check for burst instability
        if state.failure_category == "instability_under_burst":
            observed_failures.add("burst")
        
        failure_modes = [
            ("Error rate breach", "error_rate", "Error threshold exceeded"),
            ("Latency degradation", "latency", "P95 latency increased significantly"),
            ("Throughput plateau", "throughput", "RPS stopped scaling with VUs"),
            ("Infrastructure saturation", "saturation", "Resource limits detected"),
            ("Burst instability", "burst", "Spike test caused failures"),
        ]
        
        for mode_name, mode_key, note in failure_modes:
            observed = "Yes" if mode_key in observed_failures else "No"
            notes = note if observed == "Yes" else "Not observed"
            lines.append(f"| {mode_name} | {observed} | {notes} |")
        
        # C) Observability Coverage
        lines.extend([
            "",
            "### C) Observability Coverage",
            "",
            "| Signal Type | Available |",
            "|-------------|-----------|",
        ])
        
        app_telemetry = "Yes" if state.telemetry_insights else "No"
        infra_metrics = "Yes" if state.infra_saturation else "No"
        load_metrics = "Yes" if state.load_results else "No"
        
        lines.append(f"| Application telemetry | {app_telemetry} |")
        lines.append(f"| Infrastructure metrics | {infra_metrics} |")
        lines.append(f"| Load generator metrics | {load_metrics} |")
        
        return "\n".join(lines)
    
    def _api_trigger_summary_section(self, state: AgentState) -> str:
        """API & Backend Trigger Summary section"""
        lines = [
            "## API & Backend Trigger Summary",
            "",
        ]
        
        # Check if we have API-level telemetry
        has_api_telemetry = bool(
            state.telemetry_insights and 
            state.telemetry_insights.endpoints and 
            len(state.telemetry_insights.endpoints) > 0
        )
        
        # Get endpoints from telemetry or generated tests
        endpoints = []
        if has_api_telemetry:
            for ep in state.telemetry_insights.endpoints[:10]:
                if isinstance(ep, dict):
                    endpoints.append(ep.get("path", str(ep)))
                elif hasattr(ep, 'path'):
                    endpoints.append(ep.path)
                else:
                    endpoints.append(str(ep))
        elif state.generated_tests:
            for test in state.generated_tests:
                if "endpoints" in test:
                    for ep in test["endpoints"]:
                        path = ep.get("path", ep) if isinstance(ep, dict) else str(ep)
                        if path not in endpoints:
                            endpoints.append(path)
        
        # Add fallback note if no API-level telemetry
        if not has_api_telemetry:
            lines.append("*API-level telemetry not available; summary inferred from load execution.*")
            lines.append("")
        
        if not endpoints:
            endpoints = [state.target_url or "/ (default)"]
        
        # A) APIs Exercised
        lines.extend([
            "### A) APIs/Targets Exercised",
            "",
            "| API Endpoint | Test Phase | Load Applied | Observed Effect |",
            "|--------------|------------|--------------|-----------------|",
        ])
        
        # Determine test phases that ran
        test_phases = set()
        if state.load_results:
            for r in state.load_results:
                phase = r.test_type.split("_")[0].lower()
                test_phases.add(phase)
        
        # Determine overall observed effect
        overall_effect = "None"
        if state.breaking_point and state.breaking_point.vus_at_break > 0:
            if state.breaking_point.failure_type == "error_rate_breach":
                overall_effect = "Errors increased"
            elif state.breaking_point.failure_type == "latency_degradation":
                overall_effect = "Latency increased"
            else:
                overall_effect = "Degradation observed"
        elif state.failure_category == "no_failure":
            overall_effect = "None"
        
        # Generate load applied descriptions
        for endpoint in endpoints[:5]:  # Limit to 5 endpoints
            phases_str = ", ".join(sorted(test_phases)) if test_phases else "N/A"
            
            # Describe load applied
            max_vus = max((r.vus for r in state.load_results), default=0) if state.load_results else 0
            if "spike" in test_phases:
                load_desc = f"Spike to {max_vus} VUs"
            elif "stress" in test_phases or "adaptive" in test_phases:
                load_desc = f"Gradual ramp to {max_vus} VUs"
            elif "baseline" in test_phases:
                load_desc = f"Steady {max_vus} VUs"
            else:
                load_desc = f"Up to {max_vus} VUs"
            
            lines.append(f"| {endpoint} | {phases_str} | {load_desc} | {overall_effect} |")
        
        if len(endpoints) > 5:
            lines.append(f"| ... and {len(endpoints) - 5} more | - | - | - |")
        
        # B) APIs Contributing to Instability
        lines.extend([
            "",
            "### B) APIs Contributing to Instability",
            "",
        ])
        
        # Check if we have signals associated with specific endpoints
        instability_apis = []
        if state.breaking_point and state.breaking_point.signals:
            # Extract any endpoint-specific signals
            for signal in state.breaking_point.signals:
                if "/" in signal:  # Simple heuristic for endpoint reference
                    instability_apis.append(signal)
        
        if instability_apis:
            lines.extend([
                "| API Endpoint | Failure Signal | Phase |",
                "|--------------|----------------|-------|",
            ])
            for api_signal in instability_apis[:5]:
                phase = state.breaking_point.failure_type if state.breaking_point else "N/A"
                lines.append(f"| {api_signal} | {phase} | stress |")
        else:
            lines.append("No single API endpoint dominated failure behavior.")
        
        return "\n".join(lines)

    def _test_coverage_summary_section(self, state: AgentState) -> str:
        """Test coverage summary section"""
        if not state.load_results:
            return """## Test Coverage Summary

No test coverage data available."""
        
        results = state.load_results
        
        # Max VUs reached
        max_vus = max(r.vus for r in results)
        
        # Max sustained RPS
        max_rps = max(r.throughput_rps for r in results)
        
        # Total requests executed
        total_requests = sum(r.total_requests for r in results)
        
        # Longest continuous load duration
        durations = []
        for r in results:
            dur_str = r.duration.rstrip('s')
            try:
                durations.append(int(dur_str))
            except ValueError:
                durations.append(0)
        longest_duration = max(durations) if durations else 0
        
        # Spike severity description
        spike_results = [r for r in results if 'spike' in r.test_type.lower()]
        if spike_results:
            spike_error = max(r.error_rate for r in spike_results)
            if spike_error >= 0.5:
                spike_severity = "Severe (>50% errors)"
            elif spike_error >= 0.2:
                spike_severity = "Moderate (20-50% errors)"
            elif spike_error >= 0.05:
                spike_severity = "Mild (5-20% errors)"
            else:
                spike_severity = "Negligible (<5% errors)"
        else:
            spike_severity = "Not tested"
        
        # Recovery observed
        recovery_observed = "No"
        for i in range(1, len(results)):
            prev = results[i-1]
            curr = results[i]
            if prev.error_rate > 0.1 and curr.error_rate < 0.05:
                recovery_observed = "Yes"
                break
        
        return f"""## Test Coverage Summary

| Metric | Value |
|--------|-------|
| Max VUs Reached | {max_vus} |
| Max Sustained RPS | {max_rps:.1f} |
| Total Requests Executed | {total_requests:,} |
| Longest Load Duration | {longest_duration}s |
| Spike Severity | {spike_severity} |
| Recovery Observed | {recovery_observed} |"""

    def _telemetry_section(self, state: AgentState) -> str:
        """Telemetry data section"""
        if not state.telemetry_insights:
            return """## Telemetry Analysis

No telemetry data was analyzed."""
        
        ti = state.telemetry_insights
        
        return f"""## Telemetry Analysis

**Source:** {ti.source}  
**Collection Time:** {ti.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")}

### Endpoints Analyzed

{len(ti.endpoints)} endpoints were analyzed for traffic patterns."""

    def _methodology_section(self) -> str:
        """Methodology explanation"""
        return """## Methodology

This analysis was performed using SentinelPerf AI, an autonomous performance engineering agent.

### Analysis Pipeline

1. **Telemetry Analysis** - Inferred traffic patterns from available telemetry
2. **Test Generation** - Generated load, stress, and spike test configurations
3. **Load Execution** - Executed tests using k6 load testing framework
4. **Breaking Point Detection** - Identified first point of failure
5. **Root Cause Analysis** - Determined probable cause using observed signals only

### LLM Rules Enforced

- LLM may NOT invent metrics not present in observed data
- LLM may NOT infer causes without supporting signals
- All reasoning is step-by-step and reproducible
- Confidence scores are based on evidence strength only"""

    def _footer(self, state: AgentState) -> str:
        """Report footer"""
        duration = ""
        if state.completed_at and state.started_at:
            delta = state.completed_at - state.started_at
            duration = f"**Duration:** {delta.total_seconds():.1f} seconds  "
        
        return f"""---

## Appendix

{duration}
**Errors Encountered:** {len(state.errors)}

{chr(10).join(['- ' + e for e in state.errors]) if state.errors else 'No errors encountered.'}

---

*Generated by SentinelPerf AI v0.1.0*"""
