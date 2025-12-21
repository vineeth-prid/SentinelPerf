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
        
        sections = [
            self._header(state),
            self._executive_summary(state, result),
            self._breaking_point_section(state),
            self._root_cause_section(state),
            self._recommendations_section(state),
            self._load_test_results_section(state),
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
        limitations = "\n".join([f"- {l}" for l in rc.limitations]) \
            if rc.limitations else "- None"
        
        # Model info
        model_info = f"**Model:** {rc.llm_model}" if rc.llm_model else ""
        latency_info = f" ({rc.llm_latency_ms:.0f}ms)" if rc.llm_latency_ms > 0 else ""
        
        return f"""## Root Cause Analysis

**Analysis Mode:** {rc.llm_mode}{latency_info}  
{model_info}
**Confidence:** {rc.confidence:.0%}

### Summary

{rc.root_cause_summary}

### Primary Cause

{rc.primary_cause}

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
