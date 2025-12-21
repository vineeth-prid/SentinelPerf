"""Breaking point detection for SentinelPerf"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from sentinelperf.core.state import LoadTestResult, BreakingPoint
from sentinelperf.config.schema import LoadConfig


@dataclass
class ThresholdViolation:
    """A detected threshold violation"""
    metric: str
    threshold: float
    observed: float
    vus: int
    rps: float
    severity: float  # 0.0 to 1.0


class BreakingPointDetector:
    """
    Detects the first breaking point in load test results.
    
    Analyzes sequential test results to identify where the system
    first exceeds acceptable thresholds.
    
    Breaking point criteria (in order of priority):
    1. Error rate exceeds threshold
    2. P95 latency exceeds threshold
    3. Timeouts or connection failures
    """
    
    def __init__(self, config: LoadConfig):
        self.config = config
        self.error_threshold = config.error_rate_threshold
        self.latency_threshold = config.p95_latency_threshold_ms
    
    def detect(
        self,
        results: List[LoadTestResult],
    ) -> Optional[BreakingPoint]:
        """
        Detect breaking point from sequential test results.
        
        Args:
            results: List of LoadTestResult in chronological order
            
        Returns:
            BreakingPoint if detected, None if system handled all loads
        """
        if not results:
            return None
        
        violations: List[ThresholdViolation] = []
        
        for result in results:
            # Check error rate
            if result.error_rate > self.error_threshold:
                violations.append(ThresholdViolation(
                    metric="error_rate",
                    threshold=self.error_threshold,
                    observed=result.error_rate,
                    vus=result.vus,
                    rps=result.throughput_rps,
                    severity=min(1.0, result.error_rate / self.error_threshold),
                ))
            
            # Check P95 latency
            if result.latency_p95_ms > self.latency_threshold:
                violations.append(ThresholdViolation(
                    metric="p95_latency",
                    threshold=self.latency_threshold,
                    observed=result.latency_p95_ms,
                    vus=result.vus,
                    rps=result.throughput_rps,
                    severity=min(1.0, result.latency_p95_ms / self.latency_threshold),
                ))
        
        if not violations:
            return None
        
        # Find first violation (breaking point)
        first_violation = violations[0]
        
        # Calculate confidence based on consistency of violations
        confidence = self._calculate_confidence(violations)
        
        # Collect supporting signals
        signals = self._collect_signals(results, first_violation)
        
        return BreakingPoint(
            vus_at_break=first_violation.vus,
            rps_at_break=first_violation.rps,
            failure_type=first_violation.metric,
            threshold_exceeded=f"{first_violation.metric} > {first_violation.threshold}",
            observed_value=first_violation.observed,
            threshold_value=first_violation.threshold,
            confidence=confidence,
            signals=signals,
        )
    
    def _calculate_confidence(self, violations: List[ThresholdViolation]) -> float:
        """
        Calculate confidence score based on violation patterns.
        
        Higher confidence when:
        - Multiple violations at same VU level
        - Consistent pattern across tests
        - Clear threshold exceedance (not borderline)
        """
        if not violations:
            return 0.0
        
        # Base confidence from severity of first violation
        confidence = min(0.5 + (violations[0].severity * 0.3), 0.8)
        
        # Boost for multiple violations
        if len(violations) > 1:
            confidence += 0.1
        
        # Boost for consistent VU levels
        vus_set = set(v.vus for v in violations)
        if len(vus_set) == 1:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _collect_signals(self, results: List[LoadTestResult], violation: ThresholdViolation) -> List[str]:
        """
        Collect observable signals supporting the breaking point detection.
        
        LLM rule: Only include actually observed metrics, never infer.
        """
        signals = []
        
        signals.append(f"{violation.metric} exceeded at {violation.vus} VUs")
        signals.append(f"Observed: {violation.observed:.4f}, Threshold: {violation.threshold}")
        
        # Find the result at breaking point
        for result in results:
            if result.vus == violation.vus:
                if result.error_rate > 0:
                    signals.append(f"Error rate: {result.error_rate:.2%}")
                if result.latency_p95_ms > 0:
                    signals.append(f"P95 latency: {result.latency_p95_ms:.0f}ms")
                if result.throughput_rps > 0:
                    signals.append(f"Throughput: {result.throughput_rps:.1f} RPS")
                break
        
        return signals
