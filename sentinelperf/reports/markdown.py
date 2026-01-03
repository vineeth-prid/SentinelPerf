"""Markdown report generator for SentinelPerf"""

from pathlib import Path
from datetime import datetime, timezone
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
        
        # Use execution start time for filename (REAL runtime, not reused)
        if state.started_at:
            timestamp = state.started_at.strftime("%Y%m%d_%H%M%S")
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        
        # Include execution ID in filename for uniqueness guarantee
        exec_id_short = state.execution_id[:8] if state.execution_id else "unknown"
        filename = f"sentinelperf_report_{timestamp}_{exec_id_short}.md"
        filepath = self.output_dir / filename
        
        content = self._build_report(state, result)
        
        with open(filepath, "w") as f:
            f.write(content)
        
        return filepath
    
    def _build_report(self, state: AgentState, result: ExecutionResult) -> str:
        """Build complete Markdown report"""
        
        # Report order per spec:
        # 1. Executive Summary
        # 2. Execution Proof ← NEW (mandatory)
        # 3. Test Case Summary
        # 4. Test Case Coverage Summary
        # 5. API & Backend Trigger Summary
        # 6. Breaking Point Analysis
        # 7. Failure Timeline (included in breaking_point_section)
        # 8. Root Cause Analysis
        # 9. Recommendations
        # 10. Load Test Results
        # 11. Infrastructure Saturation (if exists)
        # 12. Telemetry Analysis
        # 13. Methodology
        # 14. Appendix
        sections = [
            self._header(state, result),
            self._executive_summary(state, result),
            self._execution_proof_section(state, result),
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
    
    def _header(self, state: AgentState, result: ExecutionResult = None) -> str:
        """Report header with execution timestamps and status"""
        started = state.started_at.strftime("%Y-%m-%d %H:%M:%S UTC") if state.started_at else "N/A"
        completed = state.completed_at.strftime("%Y-%m-%d %H:%M:%S UTC") if state.completed_at else "N/A"
        
        # Get execution status if result available
        if result:
            exec_status = result.get_execution_status()
            status_display = {
                "SUCCESS": "✅ SUCCESS",
                "SUCCESS_WITH_WARNINGS": "⚠️ SUCCESS_WITH_WARNINGS",
                "FAILED_TO_EXECUTE": "❌ FAILED_TO_EXECUTE",
            }.get(exec_status.value, exec_status.value)
        else:
            status_display = '✅ Complete' if state.phase.value == 'complete' else '⚠️ ' + state.phase.value
        
        return f"""# SentinelPerf Analysis Report

**Target:** {state.target_url}  
**Environment:** {state.environment}  
**Execution ID:** `{state.execution_id or 'N/A'}`  
**Started:** {started}  
**Completed:** {completed}  
**Status:** {status_display}

---"""
    
    def _execution_proof_section(self, state: AgentState, result: ExecutionResult = None) -> str:
        """Execution Proof section - proves this report is from a real execution"""
        started = state.started_at.isoformat() if state.started_at else "N/A"
        completed = state.completed_at.isoformat() if state.completed_at else "N/A"
        
        # Calculate actual max VUs from load results
        actual_max_vus = state.achieved_max_vus
        if actual_max_vus == 0 and state.load_results:
            actual_max_vus = max((r.vus for r in state.load_results), default=0)
        
        # Get execution metrics if result available
        if result:
            exec_status = result.get_execution_status().value
            test_count = result.get_test_case_count()
            stop_reason = result.get_stop_reason()
        else:
            exec_status = "UNKNOWN"
            test_count = len(state.load_results) if state.load_results else 0
            stop_reason = state.execution_stop_reason or state.autoscale_stop_reason or state.early_stop_reason or "N/A"
        
        # Autoscale execution proof
        planned_max = state.autoscale_planned_max_vus or state.configured_max_vus or 0
        stages_planned = state.autoscale_total_stages_planned or len(state.planned_vus_stages)
        stages_executed = state.autoscale_total_stages_executed or len(state.executed_vus_stages)
        autoscale_stop = state.autoscale_stop_reason or stop_reason
        
        return f"""## Execution Proof

| Property | Value |
|----------|-------|
| Execution ID | `{state.execution_id or 'N/A'}` |
| Execution Status | **{exec_status}** |
| Started At | {started} |
| Completed At | {completed} |
| Config File | `{state.config_file_path or 'N/A'}` |
| Environment | {state.environment} |
| Autoscaling Enabled | {state.autoscaling_enabled} |
| **Planned Max VUs** | **{planned_max}** |
| **Actual Max VUs Reached** | **{actual_max_vus}** |
| Total Autoscale Stages Planned | {stages_planned} |
| Total Autoscale Stages Executed | {stages_executed} |
| **Stop Reason** | **{autoscale_stop}** |"""

    def _executive_summary(self, state: AgentState, result: ExecutionResult) -> str:
        """Executive summary section with load execution transparency"""
        summary_lines = ["## Executive Summary", ""]
        
        # Load Execution Transparency
        configured_max = state.configured_max_vus
        achieved_max = state.achieved_max_vus
        
        # Calculate achieved from load results if not set
        if achieved_max == 0 and state.load_results:
            achieved_max = max((r.vus for r in state.load_results), default=0)
        
        # Format stop reason for display
        stop_reason_display = {
            "breaking_point_error": "error threshold exceeded",
            "breaking_point_latency": "latency threshold exceeded",
            "breaking_point": "breaking point detected",
            "max_limit_reached": "configured maximum reached",
            "execution_failure": "execution failure",
            "infra_saturation": "infrastructure saturation",
        }
        
        if configured_max > 0:
            if achieved_max >= configured_max:
                load_summary = f"**Load Execution:** Scaled to **{achieved_max} VUs** (configured maximum reached ✓)"
            elif achieved_max > 0:
                load_summary = f"**Load Execution:** Scaled to **{achieved_max} VUs** of {configured_max} configured"
                reason = state.early_stop_reason
                if reason:
                    reason_text = stop_reason_display.get(reason, reason.replace("_", " "))
                    load_summary += f" — stopped: {reason_text}"
                elif state.breaking_point and state.breaking_point.vus_at_break > 0:
                    load_summary += " — breaking point detected"
            else:
                load_summary = f"**Load Execution:** Configured for {configured_max} VUs (execution incomplete)"
            summary_lines.append(load_summary)
            summary_lines.append("")
        elif state.load_results:
            # Fallback: calculate from load results
            max_from_results = max((r.vus for r in state.load_results), default=0)
            if max_from_results > 0:
                summary_lines.append(f"**Load Execution:** Tested up to **{max_from_results} VUs**")
                summary_lines.append("")
        
        # Breaking point summary
        if state.breaking_point and state.breaking_point.vus_at_break > 0:
            bp = state.breaking_point
            summary_lines.append(
                f"**Breaking Point:** System reached its limit at **{bp.vus_at_break} VUs** "
                f"({bp.rps_at_break:.1f} RPS) — {bp.failure_type.replace('_', ' ')}"
            )
        else:
            max_tested = achieved_max or max((r.vus for r in state.load_results), default=0) if state.load_results else 0
            if max_tested > 0:
                summary_lines.append(
                    f"**Breaking Point:** Not detected within tested range (up to {max_tested} VUs)"
                )
            else:
                summary_lines.append(
                    "**Breaking Point:** Not detected — load tests may not have executed"
                )
        
        summary_lines.append("")
        
        if state.root_cause and state.root_cause.primary_cause:
            rc = state.root_cause
            # Use different label based on failure status
            is_failure = state.breaking_point and state.breaking_point.vus_at_break > 0
            if is_failure:
                summary_lines.append(
                    f"**Primary Cause:** {rc.primary_cause} "
                    f"(confidence: {rc.confidence:.0%})"
                )
            else:
                summary_lines.append(
                    f"**System Behavior:** {rc.primary_cause} "
                    f"(confidence: {rc.confidence:.0%})"
                )
        
        return "\n".join(summary_lines)
    
    def _infra_saturation_section(self, state: AgentState) -> str:
        """Infrastructure Saturation section - ALWAYS rendered"""
        lines = [
            "## Infrastructure Metrics",
            "",
        ]
        
        infra = state.infra_saturation
        
        # Check if data is available
        if not infra or not infra.get("data_available", False):
            # Check for legacy format (pre_test/post_test only)
            if infra and (infra.get("pre_test") or infra.get("post_test")):
                return self._infra_saturation_section_legacy(state)
            
            lines.extend([
                "*Infrastructure metrics not captured during this test run.*",
                "",
                "To enable infrastructure monitoring, ensure the test environment supports /proc filesystem access.",
            ])
            return "\n".join(lines)
        
        # Render timeline table
        snapshots = infra.get("snapshots", [])
        warnings = infra.get("warnings", [])
        confidence_penalty = infra.get("confidence_penalty", 0)
        saturated_at_break = infra.get("saturated_at_break", False)
        breaking_point_vus = infra.get("breaking_point_vus", 0)
        
        # Check if snapshots have RPS/latency (new format) or just phase (old format)
        has_load_metrics = snapshots and "rps" in snapshots[0]
        
        if has_load_metrics:
            # New format with VU correlation
            lines.extend([
                "### Resource Usage by VU Level",
                "",
                "| VUs | CPU% | Memory% | RPS | P95 Latency | Error Rate | Status |",
                "|-----|------|---------|-----|-------------|------------|--------|",
            ])
            
            for snap in snapshots:
                vus = snap.get("vus", 0)
                cpu = snap.get("cpu_percent", 0)
                mem = snap.get("memory_percent", 0)
                rps = snap.get("rps", 0)
                latency = snap.get("latency_p95_ms", 0)
                error_rate = snap.get("error_rate", 0)
                saturated = snap.get("saturated", False)
                
                # Determine status
                if saturated:
                    status = "⚠️ Saturated"
                elif error_rate >= 0.05:
                    status = "⚠️ High errors"
                elif latency >= 5000:
                    status = "⚠️ High latency"
                else:
                    status = "✓ Normal"
                
                lines.append(
                    f"| {vus} | {cpu:.1f}% | {mem:.1f}% | {rps:.1f} | {latency:.0f}ms | {error_rate:.1%} | {status} |"
                )
        else:
            # Old format with phase names
            lines.extend([
                "### Resource Usage by Load Phase",
                "",
                "| Load Phase | VUs | CPU% | Memory% | Notes |",
                "|------------|-----|------|---------|-------|",
            ])
            
            for snap in snapshots:
                phase = snap.get("phase", "unknown").replace("_", " ").title()
                vus = snap.get("vus", 0)
                cpu = snap.get("cpu_percent", 0)
                mem = snap.get("memory_percent", 0)
                notes = snap.get("notes", "")
                saturated = snap.get("saturated", False)
                
                if saturated:
                    notes = f"⚠️ {notes}" if notes else "⚠️ Saturated"
                
                lines.append(f"| {phase} | {vus} | {cpu:.1f}% | {mem:.1f}% | {notes} |")
        
        # Saturation at breaking point note
        if saturated_at_break:
            lines.extend([
                "",
                f"**⚠️ Infrastructure saturation detected at breaking point ({breaking_point_vus} VUs)**",
                "",
                "This may indicate that the breaking point was caused or influenced by infrastructure limits rather than application limits.",
            ])
        
        # Warnings section
        if warnings:
            lines.extend([
                "",
                "### Warnings",
                "",
            ])
            for w in warnings:
                lines.append(f"- {w}")
        
        # Confidence penalty
        if confidence_penalty > 0:
            lines.append("")
            lines.append(f"**Confidence penalty applied:** -{confidence_penalty*100:.0f}%")
        
        return "\n".join(lines)
    
    def _infra_saturation_section_legacy(self, state: AgentState) -> str:
        """Legacy format for backward compatibility (pre_test/post_test only)"""
        infra = state.infra_saturation
        pre = infra.get("pre_test", {})
        post = infra.get("post_test", {})
        warnings = infra.get("warnings", [])
        confidence_penalty = infra.get("confidence_penalty", 0)
        
        lines = [
            "## Infrastructure Metrics",
            "",
            "### Resource Usage by Load Phase",
            "",
            "| Load Phase | VUs | CPU% | Memory% | Notes |",
            "|------------|-----|------|---------|-------|",
            f"| Pre-test | 0 | {pre.get('cpu_percent', 0):.1f}% | {pre.get('memory_percent', 0):.1f}% | {'⚠️ Saturated' if pre.get('saturated') else 'Normal'} |",
            f"| Post-test | - | {post.get('cpu_percent', 0):.1f}% | {post.get('memory_percent', 0):.1f}% | {'⚠️ Saturated' if post.get('saturated') else 'Normal'} |",
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
        
        # Determine if this is a failure or no-failure case
        is_failure = state.breaking_point and state.breaking_point.vus_at_break > 0
        
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
        
        # Pattern section - different heading based on failure status
        pattern_section = ""
        if rc.failure_pattern and rc.pattern_explanation:
            if is_failure:
                pattern_display = rc.failure_pattern.replace("_", " ").title()
                pattern_section = f"""

### Failure Pattern

**Detected Pattern:** {pattern_display}

{rc.pattern_explanation}"""
            else:
                pattern_section = f"""

### Observed Behavior

{rc.pattern_explanation}"""
        
        # Use different heading based on failure status
        if is_failure:
            cause_heading = "### Primary Cause"
        else:
            cause_heading = "### System Behavior Explanation"
        
        # Different contributing factors heading
        if is_failure:
            factors_heading = "### Contributing Factors"
        else:
            factors_heading = "### Observations"
        
        return f"""## Root Cause Analysis

**Analysis Mode:** {rc.llm_mode}{latency_info}  
{model_info}
**Confidence:** {rc.confidence:.0%}

### Summary

{rc.root_cause_summary}

{cause_heading}

{rc.primary_cause}{pattern_section}

{factors_heading}

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
        """Test Case Summary section - ALWAYS rendered with VU transparency"""
        lines = [
            "## Test Case Summary",
            "",
        ]
        
        # Add configured vs achieved VUs transparency
        configured_max = state.configured_max_vus
        achieved_max = state.achieved_max_vus
        
        if configured_max > 0:
            lines.append("### Load Execution Summary")
            lines.append("")
            lines.append(f"- **Configured Max VUs:** {configured_max}")
            lines.append(f"- **Achieved Max VUs:** {achieved_max}")
            
            if achieved_max < configured_max:
                if state.early_stop_reason:
                    lines.append(f"- **Early Stop Reason:** {state.early_stop_reason}")
                elif state.breaking_point and state.breaking_point.vus_at_break > 0:
                    lines.append(f"- **Early Stop Reason:** Breaking point detected at {state.breaking_point.vus_at_break} VUs")
                else:
                    lines.append("- **Early Stop Reason:** Test did not reach configured maximum")
            elif achieved_max >= configured_max:
                lines.append("- **Status:** ✅ Full load range executed")
            
            # Show planned vs executed stages if available
            if state.planned_vus_stages and state.executed_vus_stages:
                planned = ", ".join(str(v) for v in state.planned_vus_stages)
                executed = ", ".join(str(v) for v in state.executed_vus_stages)
                lines.append(f"- **Planned Stages:** {planned} VUs")
                lines.append(f"- **Executed Stages:** {executed} VUs")
            
            lines.append("")
        
        if not state.load_results:
            lines.extend([
                "### Test Cases",
                "",
                "| Test Case | Purpose | Load Pattern | Max VUs | Duration |",
                "|-----------|---------|--------------|---------|----------|",
                "| *None* | *No tests executed* | *N/A* | *N/A* | *N/A* |",
            ])
            return "\n".join(lines)
        
        # Define test type semantics
        test_semantics = {
            "baseline": ("Steady validation", "Constant load"),
            "stress": ("Incremental capacity test", "Ramping load"),
            "spike": ("Burst resilience test", "Sudden spike"),
            "adaptive": ("Adaptive capacity search", "Stepwise escalation"),
            "autoscale": ("Auto-scaling ramp", "Staged escalation"),
            "sustained": ("Long duration stability", "Sustained load"),
            "recovery": ("Recovery behavior test", "Overload then drop"),
        }
        
        lines.extend([
            "### Test Cases",
            "",
            "| Test Case | Purpose | Load Pattern | Max VUs | Duration |",
            "|-----------|---------|--------------|---------|----------|",
        ])
        
        # Group results by test type base name
        seen_types = set()
        for r in state.load_results:
            # Extract base test type (e.g., "adaptive_10vus" -> "adaptive")
            test_type_base = r.test_type.split("_")[0].lower()
            
            # Skip if we've already added this type
            if test_type_base in seen_types:
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
        """API & Backend Trigger Summary section - ALWAYS rendered"""
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
        
        # Get endpoints from telemetry or generated tests or config
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
                test_endpoints = test.get("endpoints", [])
                # Guard: ensure endpoints is a list, not an int
                if not isinstance(test_endpoints, list):
                    continue
                for ep in test_endpoints:
                    if isinstance(ep, dict):
                        path = ep.get("path", str(ep))
                    elif isinstance(ep, str):
                        path = ep
                    else:
                        path = str(ep)
                    if path not in endpoints:
                        endpoints.append(path)
        
        # Always show telemetry status
        if has_api_telemetry:
            lines.append("*API-level telemetry available from application instrumentation.*")
        else:
            lines.append("*API-level telemetry not available; summary inferred from load execution.*")
        lines.append("")
        
        # Use target URL as fallback endpoint
        if not endpoints:
            endpoints = [state.target_url or "/ (default target)"]
        
        # A) APIs/Targets Exercised
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
        
        # Determine overall observed effect based on available data
        if not state.load_results:
            overall_effect = "Not tested"
        elif state.breaking_point and state.breaking_point.vus_at_break > 0:
            if state.breaking_point.failure_type == "error_rate_breach":
                overall_effect = "Errors increased"
            elif state.breaking_point.failure_type == "latency_degradation":
                overall_effect = "Latency increased"
            else:
                overall_effect = "Degradation observed"
        else:
            overall_effect = "No degradation detected"
        
        # Generate load applied descriptions
        if not state.load_results:
            # No load tests executed
            for endpoint in endpoints[:5]:
                lines.append(f"| {endpoint} | *Not executed* | *N/A* | *Not tested* |")
        else:
            max_vus = max((r.vus for r in state.load_results), default=0)
            for endpoint in endpoints[:5]:
                phases_str = ", ".join(sorted(test_phases)) if test_phases else "N/A"
                
                # Describe load applied
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
        elif not state.breaking_point or state.breaking_point.vus_at_break == 0:
            lines.append("No instability detected during testing.")
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
