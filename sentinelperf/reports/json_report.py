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
            "test_case_summary": self._test_case_summary(state),
            "test_case_coverage_summary": self._test_case_coverage_summary(state),
            "api_trigger_summary": self._api_trigger_summary(state),
            "test_coverage": self._test_coverage_summary(state),
            "infra_saturation": state.infra_saturation or {"warnings": [], "saturated": False},
            
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
        
        # Determine if this is a failure or no-failure case
        is_failure = state.breaking_point and state.breaking_point.vus_at_break > 0
        
        return {
            "analyzed": True,
            "is_failure": is_failure,
            "root_cause_summary": rc.root_cause_summary,
            # Use different field name based on failure status
            "primary_cause": rc.primary_cause if is_failure else None,
            "system_behavior_explanation": rc.primary_cause if not is_failure else None,
            "contributing_factors": rc.contributing_factors if is_failure else None,
            "observations": rc.contributing_factors if not is_failure else None,
            "confidence": rc.confidence,
            "assumptions": rc.assumptions,
            "limitations": rc.limitations,
            "failure_pattern": rc.failure_pattern if is_failure else None,
            "observed_behavior": rc.failure_pattern if not is_failure else None,
            "pattern_explanation": rc.pattern_explanation,
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
    
    def _test_case_summary(self, state: AgentState) -> Dict[str, Any]:
        """Test Case Summary - ALWAYS rendered, describes what tests were executed"""
        test_semantics = {
            "baseline": {"purpose": "Steady validation", "load_pattern": "Constant load"},
            "stress": {"purpose": "Incremental capacity test", "load_pattern": "Ramping load"},
            "spike": {"purpose": "Burst resilience test", "load_pattern": "Sudden spike"},
            "adaptive": {"purpose": "Adaptive capacity search", "load_pattern": "Stepwise escalation"},
            "sustained": {"purpose": "Long duration stability", "load_pattern": "Sustained load"},
            "recovery": {"purpose": "Recovery behavior test", "load_pattern": "Overload then drop"},
        }
        
        if not state.load_results:
            return {
                "test_cases": [],
                "note": "No test cases were executed"
            }
        
        test_cases = []
        seen_types = set()
        
        for r in state.load_results:
            test_type_base = r.test_type.split("_")[0].lower()
            
            if test_type_base in seen_types:
                continue
            
            type_results = [
                res for res in state.load_results 
                if res.test_type.split("_")[0].lower() == test_type_base
            ]
            max_vus = max(res.vus for res in type_results)
            duration = type_results[0].duration if type_results else "N/A"
            
            semantics = test_semantics.get(test_type_base, {"purpose": "Performance test", "load_pattern": "Variable"})
            
            test_cases.append({
                "test_case": test_type_base,
                "purpose": semantics["purpose"],
                "load_pattern": semantics["load_pattern"],
                "max_vus": max_vus,
                "duration": duration,
            })
            seen_types.add(test_type_base)
        
        return {
            "test_cases": test_cases,
            "note": None
        }
    
    def _test_case_coverage_summary(self, state: AgentState) -> Dict[str, Any]:
        """Test Case Coverage Summary - describes coverage achieved"""
        test_types_run = set()
        if state.load_results:
            for r in state.load_results:
                test_type_base = r.test_type.split("_")[0].lower()
                test_types_run.add(test_type_base)
        
        # Load Pattern Coverage
        load_patterns = {
            "baseline_steady": "baseline" in test_types_run,
            "stress_incremental_ramp": "stress" in test_types_run,
            "spike_sudden_burst": "spike" in test_types_run,
            "soak_long_duration": "sustained" in test_types_run,
            "recovery_after_overload": "recovery" in test_types_run,
        }
        
        # Failure Mode Coverage
        observed_failures = set()
        if state.failure_timeline:
            for event in state.failure_timeline:
                event_type = event.get("event_type", "")
                if event_type == "error_rate_breach":
                    observed_failures.add("error_rate_breach")
                elif event_type == "latency_degradation":
                    observed_failures.add("latency_degradation")
                elif event_type == "throughput_plateau":
                    observed_failures.add("throughput_plateau")
                elif event_type == "saturation":
                    observed_failures.add("infrastructure_saturation")
        
        if state.failure_category == "instability_under_burst":
            observed_failures.add("burst_instability")
        
        failure_modes = {
            "error_rate_breach": "error_rate_breach" in observed_failures,
            "latency_degradation": "latency_degradation" in observed_failures,
            "throughput_plateau": "throughput_plateau" in observed_failures,
            "infrastructure_saturation": "infrastructure_saturation" in observed_failures,
            "burst_instability": "burst_instability" in observed_failures,
        }
        
        # Observability Coverage
        observability = {
            "application_telemetry": state.telemetry_insights is not None,
            "infrastructure_metrics": state.infra_saturation is not None,
            "load_generator_metrics": len(state.load_results) > 0,
        }
        
        return {
            "load_pattern_coverage": load_patterns,
            "failure_mode_coverage": failure_modes,
            "observability_coverage": observability,
        }
    
    def _api_trigger_summary(self, state: AgentState) -> Dict[str, Any]:
        """API & Backend Trigger Summary - ALWAYS rendered, describes which APIs were tested"""
        # Check if we have API-level telemetry
        has_api_telemetry = bool(
            state.telemetry_insights and 
            state.telemetry_insights.endpoints and 
            len(state.telemetry_insights.endpoints) > 0
        )
        
        # Get endpoints from telemetry, generated tests, or config
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
        
        # Use target URL as fallback
        if not endpoints:
            endpoints = [state.target_url or "/ (default target)"]
        
        # Test phases
        test_phases = set()
        if state.load_results:
            for r in state.load_results:
                phase = r.test_type.split("_")[0].lower()
                test_phases.add(phase)
        
        # Observed effect with explicit values
        if not state.load_results:
            overall_effect = "not_tested"
        elif state.breaking_point and state.breaking_point.vus_at_break > 0:
            if state.breaking_point.failure_type == "error_rate_breach":
                overall_effect = "errors_increased"
            elif state.breaking_point.failure_type == "latency_degradation":
                overall_effect = "latency_increased"
            else:
                overall_effect = "degradation_observed"
        else:
            overall_effect = "no_degradation_detected"
        
        # Max VUs
        max_vus = max((r.vus for r in state.load_results), default=0) if state.load_results else 0
        
        # APIs exercised - always populated
        apis_exercised = []
        for endpoint in endpoints[:5]:
            apis_exercised.append({
                "endpoint": endpoint,
                "test_phases": sorted(list(test_phases)) if test_phases else ["not_executed"],
                "max_vus_applied": max_vus,
                "observed_effect": overall_effect,
            })
        
        # APIs contributing to instability
        instability_apis = []
        instability_note = None
        if state.breaking_point and state.breaking_point.signals:
            for signal in state.breaking_point.signals:
                if "/" in signal:
                    instability_apis.append({
                        "signal": signal,
                        "failure_type": state.breaking_point.failure_type,
                    })
        
        if not instability_apis:
            if not state.breaking_point or state.breaking_point.vus_at_break == 0:
                instability_note = "No instability detected during testing"
            else:
                instability_note = "No single API endpoint dominated failure behavior"
        
        return {
            "api_telemetry_available": has_api_telemetry,
            "note": "API-level telemetry available from application instrumentation" if has_api_telemetry else "API-level telemetry not available; summary inferred from load execution",
            "apis_exercised": apis_exercised,
            "apis_contributing_to_instability": instability_apis,
            "instability_note": instability_note,
            "total_endpoints_tested": len(endpoints),
        }
    
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
