"""Breaking point detection for SentinelPerf

Implements rules-only breaking point detection with:
1. Sustained violation detection (not peak noise)
2. Failure timeline construction
3. Primary failure classification
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
from datetime import datetime

from sentinelperf.core.state import LoadTestResult, BreakingPoint
from sentinelperf.config.schema import LoadConfig


class FailureCategory(str, Enum):
    """Primary failure classification categories"""
    CAPACITY_EXHAUSTION = "capacity_exhaustion"
    LATENCY_AMPLIFICATION = "latency_amplification"
    ERROR_DRIVEN_COLLAPSE = "error_driven_collapse"
    INSTABILITY_UNDER_BURST = "instability_under_burst"
    ALREADY_DEGRADED_BASELINE = "already_degraded_baseline"
    NO_FAILURE = "no_failure"


class ViolationType(str, Enum):
    """Types of threshold violations"""
    ERROR_RATE_BREACH = "error_rate_breach"
    LATENCY_DEGRADATION = "latency_degradation"
    THROUGHPUT_PLATEAU = "throughput_plateau"
    SATURATION = "saturation"


@dataclass
class TimelineEvent:
    """A single event in the failure timeline"""
    timestamp: str  # Relative time marker (t0, t1, etc.)
    event_type: str
    description: str
    test_type: str
    vus: int
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "description": self.description,
            "test_type": self.test_type,
            "vus": self.vus,
            "metrics": self.metrics,
        }


@dataclass
class FailureTimeline:
    """Ordered timeline of failure events for LLM reasoning"""
    events: List[TimelineEvent] = field(default_factory=list)
    
    def add_event(self, event: TimelineEvent) -> None:
        self.events.append(event)
    
    def to_list(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.events]
    
    def to_text(self) -> str:
        """Generate human-readable timeline"""
        if not self.events:
            return "No failure events recorded"
        
        lines = []
        for event in self.events:
            lines.append(f"{event.timestamp}: {event.description}")
        return "\n".join(lines)


@dataclass
class Violation:
    """A detected threshold violation with context"""
    violation_type: ViolationType
    test_type: str
    vus: int
    rps: float
    metric_name: str
    threshold: float
    observed: float
    severity: float  # How much over threshold (ratio)
    sustained: bool  # Is this a sustained violation?
    
    @property
    def is_significant(self) -> bool:
        """Check if violation is significant (not noise)"""
        return self.severity >= 1.5 or self.sustained


@dataclass
class BreakingPointResult:
    """Complete breaking point analysis result"""
    detected: bool
    breaking_point: Optional[BreakingPoint]
    primary_category: FailureCategory
    category_confidence: float
    timeline: FailureTimeline
    violations: List[Violation]
    classification_rationale: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected": self.detected,
            "primary_category": self.primary_category.value,
            "category_confidence": self.category_confidence,
            "timeline": self.timeline.to_list(),
            "violations_count": len(self.violations),
            "classification_rationale": self.classification_rationale,
        }


class BreakingPointDetector:
    """
    Rules-only breaking point detection.
    
    Signals used (no more):
    1. Error rate threshold breach (sustained)
    2. Latency p95/p99 degradation (slope-based)
    3. Throughput plateau or drop despite rising VUs
    4. Saturation indicators (VU increase ≠ RPS increase)
    
    Output includes:
    - Test type (baseline / stress / spike)
    - VU / load level
    - Timestamp / phase
    - Violated condition(s)
    """
    
    # Thresholds for sustained violation detection
    SUSTAINED_WINDOW = 2  # Need 2+ consecutive violations
    LATENCY_SLOPE_THRESHOLD = 1.5  # 50% increase is significant
    SATURATION_THRESHOLD = 0.3  # <30% RPS increase for 2x VU increase = saturation
    
    def __init__(self, config: LoadConfig):
        self.config = config
        self.error_threshold = config.error_rate_threshold
        self.latency_threshold = config.p95_latency_threshold_ms
    
    def detect(self, results: List[LoadTestResult]) -> BreakingPointResult:
        """
        Detect breaking point from load test results.
        
        Returns complete analysis including timeline and classification.
        """
        if not results:
            return self._no_failure_result()
        
        # Build timeline and collect violations
        timeline = FailureTimeline()
        violations: List[Violation] = []
        
        # Track metrics for slope detection
        prev_result: Optional[LoadTestResult] = None
        event_counter = 0
        
        for result in results:
            # Add load increase event
            if prev_result is None or result.vus > prev_result.vus:
                timeline.add_event(TimelineEvent(
                    timestamp=f"t{event_counter}",
                    event_type="load_change",
                    description=f"Load increased to {result.vus} VUs ({result.test_type})",
                    test_type=result.test_type,
                    vus=result.vus,
                    metrics={"rps": result.throughput_rps},
                ))
                event_counter += 1
            
            # Check for violations
            new_violations = self._check_violations(result, prev_result, results)
            
            for violation in new_violations:
                violations.append(violation)
                
                # Add violation to timeline
                timeline.add_event(TimelineEvent(
                    timestamp=f"t{event_counter}",
                    event_type=violation.violation_type.value,
                    description=self._violation_description(violation),
                    test_type=result.test_type,
                    vus=result.vus,
                    metrics={
                        violation.metric_name: violation.observed,
                        "threshold": violation.threshold,
                    },
                ))
                event_counter += 1
            
            prev_result = result
        
        # Find first sustained violation (breaking point)
        breaking_point = self._find_breaking_point(violations, results)
        
        # Classify the failure
        category, confidence, rationale = self._classify_failure(
            violations, results, breaking_point
        )
        
        return BreakingPointResult(
            detected=breaking_point is not None,
            breaking_point=breaking_point,
            primary_category=category,
            category_confidence=confidence,
            timeline=timeline,
            violations=violations,
            classification_rationale=rationale,
        )
    
    def _check_violations(
        self,
        result: LoadTestResult,
        prev_result: Optional[LoadTestResult],
        all_results: List[LoadTestResult],
    ) -> List[Violation]:
        """Check for all violation types in a single result"""
        violations = []
        
        # 1. Error rate breach
        if result.error_rate > self.error_threshold:
            # Check if sustained
            sustained = self._is_sustained_error(result, all_results)
            
            violations.append(Violation(
                violation_type=ViolationType.ERROR_RATE_BREACH,
                test_type=result.test_type,
                vus=result.vus,
                rps=result.throughput_rps,
                metric_name="error_rate",
                threshold=self.error_threshold,
                observed=result.error_rate,
                severity=result.error_rate / self.error_threshold if self.error_threshold > 0 else float('inf'),
                sustained=sustained,
            ))
        
        # 2. Latency degradation (slope-based)
        if prev_result and prev_result.latency_p95_ms > 0:
            latency_ratio = result.latency_p95_ms / prev_result.latency_p95_ms
            
            if latency_ratio >= self.LATENCY_SLOPE_THRESHOLD:
                violations.append(Violation(
                    violation_type=ViolationType.LATENCY_DEGRADATION,
                    test_type=result.test_type,
                    vus=result.vus,
                    rps=result.throughput_rps,
                    metric_name="latency_p95_slope",
                    threshold=self.LATENCY_SLOPE_THRESHOLD,
                    observed=latency_ratio,
                    severity=latency_ratio / self.LATENCY_SLOPE_THRESHOLD,
                    sustained=False,  # Slope is point-in-time
                ))
        
        # Also check absolute latency threshold
        if result.latency_p95_ms > self.latency_threshold:
            sustained = self._is_sustained_latency(result, all_results)
            
            violations.append(Violation(
                violation_type=ViolationType.LATENCY_DEGRADATION,
                test_type=result.test_type,
                vus=result.vus,
                rps=result.throughput_rps,
                metric_name="latency_p95_absolute",
                threshold=self.latency_threshold,
                observed=result.latency_p95_ms,
                severity=result.latency_p95_ms / self.latency_threshold if self.latency_threshold > 0 else float('inf'),
                sustained=sustained,
            ))
        
        # 3. Throughput plateau/drop
        if prev_result and prev_result.throughput_rps > 0:
            rps_change = (result.throughput_rps - prev_result.throughput_rps) / prev_result.throughput_rps
            vus_change = (result.vus - prev_result.vus) / prev_result.vus if prev_result.vus > 0 else 0
            
            # Plateau: VUs increased but RPS didn't proportionally
            if vus_change > 0.2 and rps_change < 0.1:  # VUs up 20%+, RPS up <10%
                violations.append(Violation(
                    violation_type=ViolationType.THROUGHPUT_PLATEAU,
                    test_type=result.test_type,
                    vus=result.vus,
                    rps=result.throughput_rps,
                    metric_name="throughput_plateau",
                    threshold=vus_change * 0.5,  # Expect at least 50% of VU increase
                    observed=rps_change,
                    severity=abs(vus_change - rps_change) / vus_change if vus_change > 0 else 1.0,
                    sustained=False,
                ))
            
            # Drop: RPS decreased despite VU increase
            if vus_change > 0 and rps_change < -0.1:
                violations.append(Violation(
                    violation_type=ViolationType.THROUGHPUT_PLATEAU,
                    test_type=result.test_type,
                    vus=result.vus,
                    rps=result.throughput_rps,
                    metric_name="throughput_drop",
                    threshold=0,
                    observed=rps_change,
                    severity=abs(rps_change) + 1.0,  # Drops are severe
                    sustained=False,
                ))
        
        # 4. Saturation indicator
        if prev_result and prev_result.vus > 0 and prev_result.throughput_rps > 0:
            vus_ratio = result.vus / prev_result.vus
            rps_ratio = result.throughput_rps / prev_result.throughput_rps if prev_result.throughput_rps > 0 else 1
            
            # If VUs doubled but RPS increased < 30%, that's saturation
            if vus_ratio >= 1.5 and rps_ratio < 1.0 + self.SATURATION_THRESHOLD:
                violations.append(Violation(
                    violation_type=ViolationType.SATURATION,
                    test_type=result.test_type,
                    vus=result.vus,
                    rps=result.throughput_rps,
                    metric_name="saturation",
                    threshold=vus_ratio * 0.5,  # Expect at least 50% of VU increase in RPS
                    observed=rps_ratio,
                    severity=(vus_ratio - rps_ratio) / vus_ratio,
                    sustained=False,
                ))
        
        return violations
    
    def _is_sustained_error(self, result: LoadTestResult, all_results: List[LoadTestResult]) -> bool:
        """Check if error rate breach is sustained across multiple results"""
        # Find results at same or higher VU level
        similar_results = [
            r for r in all_results
            if r.vus >= result.vus * 0.8 and r.test_type == result.test_type
        ]
        
        breach_count = sum(1 for r in similar_results if r.error_rate > self.error_threshold)
        return breach_count >= self.SUSTAINED_WINDOW
    
    def _is_sustained_latency(self, result: LoadTestResult, all_results: List[LoadTestResult]) -> bool:
        """Check if latency breach is sustained"""
        similar_results = [
            r for r in all_results
            if r.vus >= result.vus * 0.8 and r.test_type == result.test_type
        ]
        
        breach_count = sum(1 for r in similar_results if r.latency_p95_ms > self.latency_threshold)
        return breach_count >= self.SUSTAINED_WINDOW
    
    def _find_breaking_point(
        self,
        violations: List[Violation],
        results: List[LoadTestResult],
    ) -> Optional[BreakingPoint]:
        """Find first SUSTAINED violation as breaking point"""
        
        # Prioritize sustained violations
        sustained = [v for v in violations if v.sustained and v.is_significant]
        
        if sustained:
            first = sustained[0]
        elif violations:
            # Fall back to first significant violation
            significant = [v for v in violations if v.is_significant]
            if significant:
                first = significant[0]
            else:
                return None
        else:
            return None
        
        # Build signals list
        signals = [
            f"{first.violation_type.value} at {first.vus} VUs",
            f"Observed: {first.observed:.4f}, Threshold: {first.threshold:.4f}",
            f"Severity: {first.severity:.2f}x threshold",
        ]
        
        if first.sustained:
            signals.append("Violation is SUSTAINED (not transient)")
        
        # Add context from result at this VU level
        for r in results:
            if r.vus == first.vus and r.test_type == first.test_type:
                signals.append(f"Error rate: {r.error_rate:.2%}")
                signals.append(f"P95 latency: {r.latency_p95_ms:.1f}ms")
                signals.append(f"Throughput: {r.throughput_rps:.1f} RPS")
                break
        
        # Calculate confidence
        confidence = self._calculate_confidence(violations, first)
        
        return BreakingPoint(
            vus_at_break=first.vus,
            rps_at_break=first.rps,
            failure_type=first.violation_type.value,
            threshold_exceeded=f"{first.metric_name} > {first.threshold}",
            observed_value=first.observed,
            threshold_value=first.threshold,
            confidence=confidence,
            signals=signals,
        )
    
    def _calculate_confidence(self, violations: List[Violation], primary: Violation) -> float:
        """Calculate confidence based on violation patterns"""
        confidence = 0.5  # Base
        
        # Sustained = higher confidence
        if primary.sustained:
            confidence += 0.2
        
        # Severity matters
        if primary.severity >= 2.0:
            confidence += 0.15
        elif primary.severity >= 1.5:
            confidence += 0.1
        
        # Multiple violation types = higher confidence
        violation_types = set(v.violation_type for v in violations)
        if len(violation_types) >= 2:
            confidence += 0.1
        
        # Multiple violations at same VU = higher confidence
        same_vu_violations = [v for v in violations if v.vus == primary.vus]
        if len(same_vu_violations) >= 2:
            confidence += 0.1
        
        return min(confidence, 0.95)
    
    def _classify_failure(
        self,
        violations: List[Violation],
        results: List[LoadTestResult],
        breaking_point: Optional[BreakingPoint],
    ) -> Tuple[FailureCategory, float, List[str]]:
        """
        Classify failure into ONE primary category.
        
        Categories:
        1. CAPACITY_EXHAUSTION - System hit resource limits
        2. LATENCY_AMPLIFICATION - Latency spiraled without errors
        3. ERROR_DRIVEN_COLLAPSE - Errors caused cascading failures
        4. INSTABILITY_UNDER_BURST - Failed specifically during spike
        5. ALREADY_DEGRADED_BASELINE - Baseline test already showed problems
        """
        rationale = []
        
        if not violations:
            return FailureCategory.NO_FAILURE, 1.0, ["No violations detected"]
        
        # Check if baseline test had issues
        baseline_results = [r for r in results if r.test_type == "baseline"]
        baseline_had_errors = any(r.error_rate > self.error_threshold for r in baseline_results)
        baseline_had_latency = any(r.latency_p95_ms > self.latency_threshold for r in baseline_results)
        
        if baseline_had_errors or baseline_had_latency:
            rationale.append("Baseline test already showed degradation")
            if baseline_had_errors:
                rationale.append(f"Baseline error rate exceeded {self.error_threshold:.1%}")
            if baseline_had_latency:
                rationale.append(f"Baseline latency exceeded {self.latency_threshold}ms")
            return FailureCategory.ALREADY_DEGRADED_BASELINE, 0.85, rationale
        
        # Check for spike-specific failure
        spike_violations = [v for v in violations if v.test_type == "spike"]
        non_spike_violations = [v for v in violations if v.test_type != "spike"]
        
        if spike_violations and not non_spike_violations:
            rationale.append("Failures occurred only during spike test")
            rationale.append("System stable under gradual load but fails under burst")
            return FailureCategory.INSTABILITY_UNDER_BURST, 0.8, rationale
        
        # Analyze violation types
        error_violations = [v for v in violations if v.violation_type == ViolationType.ERROR_RATE_BREACH]
        latency_violations = [v for v in violations if v.violation_type == ViolationType.LATENCY_DEGRADATION]
        throughput_violations = [v for v in violations if v.violation_type in (
            ViolationType.THROUGHPUT_PLATEAU, ViolationType.SATURATION
        )]
        
        # ERROR_DRIVEN_COLLAPSE: Errors appeared before/during latency issues
        if error_violations:
            error_first = error_violations[0]
            
            # Check if errors caused the collapse
            if latency_violations:
                latency_first = latency_violations[0]
                if error_first.vus <= latency_first.vus:
                    rationale.append("Errors appeared before or at same load as latency degradation")
                    rationale.append(f"Error breach at {error_first.vus} VUs")
                    rationale.append("Errors likely caused cascading failures")
                    return FailureCategory.ERROR_DRIVEN_COLLAPSE, 0.8, rationale
            else:
                rationale.append("Error rate breach without significant latency degradation")
                rationale.append(f"Error breach at {error_first.vus} VUs ({error_first.observed:.1%})")
                return FailureCategory.ERROR_DRIVEN_COLLAPSE, 0.75, rationale
        
        # LATENCY_AMPLIFICATION: Latency spiraled but errors stayed low
        if latency_violations and not error_violations:
            rationale.append("Latency degraded without error rate breach")
            if len(latency_violations) >= 2:
                slope = latency_violations[-1].observed / latency_violations[0].observed
                rationale.append(f"Latency increased {slope:.1f}x during test")
            rationale.append("System slowed down but kept processing requests")
            return FailureCategory.LATENCY_AMPLIFICATION, 0.8, rationale
        
        # CAPACITY_EXHAUSTION: Throughput plateau/saturation
        if throughput_violations:
            rationale.append("Throughput plateaued or dropped despite increasing VUs")
            saturation = [v for v in throughput_violations if v.violation_type == ViolationType.SATURATION]
            if saturation:
                rationale.append(f"Saturation detected at {saturation[0].vus} VUs")
                rationale.append("VU increases did not translate to RPS increases")
            return FailureCategory.CAPACITY_EXHAUSTION, 0.75, rationale
        
        # Default: CAPACITY_EXHAUSTION if we have breaking point
        if breaking_point:
            rationale.append(f"Breaking point at {breaking_point.vus_at_break} VUs")
            rationale.append(f"Primary violation: {breaking_point.failure_type}")
            return FailureCategory.CAPACITY_EXHAUSTION, 0.6, rationale
        
        return FailureCategory.NO_FAILURE, 0.5, ["Could not classify failure"]
    
    def _violation_description(self, v: Violation) -> str:
        """Generate human-readable violation description"""
        if v.violation_type == ViolationType.ERROR_RATE_BREACH:
            return f"Error rate crossed threshold ({v.observed:.1%} > {v.threshold:.1%})"
        elif v.violation_type == ViolationType.LATENCY_DEGRADATION:
            if "slope" in v.metric_name:
                return f"Latency slope changed ({v.observed:.1f}x increase)"
            else:
                return f"Latency exceeded threshold ({v.observed:.0f}ms > {v.threshold:.0f}ms)"
        elif v.violation_type == ViolationType.THROUGHPUT_PLATEAU:
            if "drop" in v.metric_name:
                return f"Throughput dropped ({v.observed:.1%} change)"
            else:
                return f"Throughput plateaued (expected {v.threshold:.1%}, got {v.observed:.1%})"
        elif v.violation_type == ViolationType.SATURATION:
            return f"Saturation detected (VUs increased but RPS stagnant)"
        return f"{v.violation_type.value}: {v.observed:.4f}"
    
    def _no_failure_result(self) -> BreakingPointResult:
        """Return result for no failure detected"""
        return BreakingPointResult(
            detected=False,
            breaking_point=None,
            primary_category=FailureCategory.NO_FAILURE,
            category_confidence=1.0,
            timeline=FailureTimeline(),
            violations=[],
            classification_rationale=["No test results to analyze"],
        )
