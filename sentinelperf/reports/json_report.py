"""JSON report generator for SentinelPerf"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from dataclasses import asdict

from sentinelperf.core.state import ExecutionResult, AgentState


class JSONReporter:
    """
    Generates CI/CD-friendly JSON summary.
    
    Output is designed for:
    - Pipeline integration
    - Automated threshold checks
    - Historical tracking
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(self, result: ExecutionResult) -> Path:
        """Generate JSON summary and return file path"""
        
        state = result.state
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"sentinelperf_summary_{timestamp}.json"
        filepath = self.output_dir / filename
        
        data = self._build_summary(state, result)
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        return filepath
    
    def _build_summary(self, state: AgentState, result: ExecutionResult) -> Dict[str, Any]:
        """Build JSON summary structure"""
        
        summary = {
            "version": "1.0",
            "generated_at": datetime.utcnow().isoformat(),
            "status": "success" if result.success else "failure",
            
            "target": {
                "url": state.target_url,
                "environment": state.environment,
            },
            
            "execution": {
                "started_at": state.started_at.isoformat() if state.started_at else None,
                "completed_at": state.completed_at.isoformat() if state.completed_at else None,
                "duration_seconds": (
                    (state.completed_at - state.started_at).total_seconds()
                    if state.completed_at and state.started_at else None
                ),
                "phase": state.phase.value,
                "errors": state.errors,
            },
            
            "telemetry": self._telemetry_summary(state),
            "breaking_point": self._breaking_point_summary(state),
            "root_cause": self._root_cause_summary(state),
            "recommendations": state.recommendations or {"recommendations": [], "limitations": []},
            "load_tests": self._load_tests_summary(state),
            "test_coverage": self._test_coverage_summary(state),
            
            "ci_cd": self._cicd_output(state, result),
        }
        
        return summary
    
    def _telemetry_summary(self, state: AgentState) -> Dict[str, Any]:
        """Telemetry section"""
        if not state.telemetry_insights:
            return {"available": False}
        
        ti = state.telemetry_insights
        return {
            "available": True,
            "source": ti.source,
            "endpoints_analyzed": len(ti.endpoints),
            "collection_timestamp": ti.timestamp.isoformat(),
        }
    
    def _breaking_point_summary(self, state: AgentState) -> Dict[str, Any]:
        """Breaking point section with classification and timeline"""
        result = {
            "detected": False,
            "classification": state.failure_category or "no_failure",
            "message": "No breaking point detected within tested parameters",
            "timeline": state.failure_timeline or [],
        }
        
        if state.breaking_point and state.breaking_point.vus_at_break > 0:
            bp = state.breaking_point
            result.update({
                "detected": True,
                "vus_at_break": bp.vus_at_break,
                "rps_at_break": bp.rps_at_break,
                "failure_type": bp.failure_type,
                "threshold_exceeded": bp.threshold_exceeded,
                "observed_value": bp.observed_value,
                "threshold_value": bp.threshold_value,
                "confidence": bp.confidence,
                "signals": bp.signals,
                "message": f"Breaking point at {bp.vus_at_break} VUs ({bp.failure_type})",
            })
        
        return result
    
    def _root_cause_summary(self, state: AgentState) -> Dict[str, Any]:
        """Root cause section"""
        if not state.root_cause:
            return {"analyzed": False}
        
        rc = state.root_cause
        return {
            "analyzed": True,
            "root_cause_summary": rc.root_cause_summary,
            "primary_cause": rc.primary_cause,
            "contributing_factors": rc.contributing_factors,
            "confidence": rc.confidence,
            "assumptions": rc.assumptions,
            "limitations": rc.limitations,
            "llm_mode": rc.llm_mode,
            "llm_model": rc.llm_model,
            "llm_latency_ms": rc.llm_latency_ms,
        }
    
    def _load_tests_summary(self, state: AgentState) -> list:
        """Load test results"""
        if not state.load_results:
            return []
        
        return [
            {
                "test_type": r.test_type,
                "vus": r.vus,
                "duration": r.duration,
                "total_requests": r.total_requests,
                "successful_requests": r.successful_requests,
                "failed_requests": r.failed_requests,
                "error_rate": r.error_rate,
                "latency": {
                    "p50_ms": r.latency_p50_ms,
                    "p95_ms": r.latency_p95_ms,
                    "p99_ms": r.latency_p99_ms,
                },
                "throughput_rps": r.throughput_rps,
                "timestamp": r.timestamp.isoformat(),
            }
            for r in state.load_results
        ]
    
    def _test_coverage_summary(self, state: AgentState) -> Dict[str, Any]:
        """Test coverage summary"""
        if not state.load_results:
            return {
                "max_vus_reached": 0,
                "max_sustained_rps": 0,
                "total_requests_executed": 0,
                "longest_load_duration_seconds": 0,
                "spike_severity": "not_tested",
                "recovery_observed": False,
            }
        
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
        
        # Spike severity
        spike_results = [r for r in results if 'spike' in r.test_type.lower()]
        if spike_results:
            spike_error = max(r.error_rate for r in spike_results)
            if spike_error >= 0.5:
                spike_severity = "severe"
            elif spike_error >= 0.2:
                spike_severity = "moderate"
            elif spike_error >= 0.05:
                spike_severity = "mild"
            else:
                spike_severity = "negligible"
        else:
            spike_severity = "not_tested"
        
        # Recovery observed
        recovery_observed = False
        for i in range(1, len(results)):
            prev = results[i-1]
            curr = results[i]
            if prev.error_rate > 0.1 and curr.error_rate < 0.05:
                recovery_observed = True
                break
        
        return {
            "max_vus_reached": max_vus,
            "max_sustained_rps": round(max_rps, 1),
            "total_requests_executed": total_requests,
            "longest_load_duration_seconds": longest_duration,
            "spike_severity": spike_severity,
            "recovery_observed": recovery_observed,
        }
    
    def _cicd_output(self, state: AgentState, result: ExecutionResult) -> Dict[str, Any]:
        """
        CI/CD-specific output for pipeline integration.
        
        Returns:
            - exit_code: 0 for success, 1 for failure
            - thresholds_passed: Boolean for threshold gates
            - summary: One-line summary for pipeline logs
        """
        
        thresholds_passed = True
        if state.breaking_point and state.breaking_point.vus_at_break > 0:
            thresholds_passed = False
        
        summary = ""
        if result.success and not state.breaking_point:
            summary = f"PASS: No breaking point detected for {state.target_url}"
        elif state.breaking_point:
            bp = state.breaking_point
            summary = f"FAIL: Breaking point at {bp.vus_at_break} VUs ({bp.failure_type})"
        else:
            summary = f"ERROR: Analysis incomplete - {state.phase.value}"
        
        return {
            "exit_code": 0 if result.success else 1,
            "thresholds_passed": thresholds_passed,
            "summary": summary,
            "breaking_point_vus": (
                state.breaking_point.vus_at_break
                if state.breaking_point else None
            ),
            "root_cause_confidence": (
                state.root_cause.confidence
                if state.root_cause else None
            ),
            "recommendation_count": (
                len(state.recommendations.get("recommendations", []))
                if state.recommendations else 0
            ),
        }
