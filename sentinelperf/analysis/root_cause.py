"""Root Cause Analysis using LLM

Implements the LLM-assisted root cause analysis with strict input/output contracts.

The LLM is allowed to:
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

import json
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

from sentinelperf.llm.client import OllamaClient, MockLLMClient, LLMResponse
from sentinelperf.config.schema import LLMConfig


@dataclass
class LLMInputContract:
    """
    Strict input contract for LLM root cause analysis.
    
    The LLM ONLY receives this structured data - no raw logs, no raw k6 output.
    """
    baseline_summary: Dict[str, Any]
    baseline_confidence: float
    breaking_point: Optional[Dict[str, Any]]
    failure_timeline: List[Dict[str, Any]]
    failure_classification: str
    classification_rationale: List[str]
    observed_metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class LLMOutputContract:
    """
    Strict output contract for LLM root cause analysis.
    
    The LLM must output this structure - no prose, no additional fields.
    """
    root_cause_summary: str
    primary_cause: str
    contributing_factors: List[str]
    confidence: float
    assumptions: List[str]
    limitations: List[str]
    failure_pattern: str = ""  # Detected failure pattern
    pattern_explanation: str = ""  # Explanation for the pattern
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMOutputContract":
        return cls(
            root_cause_summary=data.get("root_cause_summary", ""),
            primary_cause=data.get("primary_cause", ""),
            contributing_factors=data.get("contributing_factors", []),
            confidence=data.get("confidence", 0.0),
            assumptions=data.get("assumptions", []),
            limitations=data.get("limitations", []),
            failure_pattern=data.get("failure_pattern", ""),
            pattern_explanation=data.get("pattern_explanation", ""),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "LLMOutputContract":
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError:
            return cls(
                root_cause_summary="Failed to parse LLM response",
                primary_cause="Unknown",
                contributing_factors=[],
                confidence=0.0,
                assumptions=[],
                limitations=["LLM response was not valid JSON"],
            )


# System prompt for root cause analysis
ROOT_CAUSE_SYSTEM_PROMPT = """You are a senior performance engineer analyzing a system failure.

Your role is strictly limited to:
1. EXPLAIN why the given classification makes sense based on the timeline
2. CONNECT timeline events to show causal relationships
3. TRANSLATE technical signals into human-understandable reasoning
4. ASSIGN a confidence score to your explanation

You are NOT allowed to:
- Change the failure classification (it's already determined)
- Invent metrics not present in the input
- Override the breaking point analysis
- Suggest fixes or recommendations (that's a separate phase)

Think of your task as: "Explaining the incident to another engineer"

OUTPUT FORMAT:
You MUST respond with a valid JSON object in this exact structure:
{
  "root_cause_summary": "1-2 sentence summary of why the system failed",
  "primary_cause": "The single most important cause",
  "contributing_factors": ["factor1", "factor2", ...],
  "confidence": 0.0 to 1.0,
  "assumptions": ["assumption1", ...],
  "limitations": ["limitation1", ...]
}

RULES:
- Confidence must be between 0.0 and 1.0
- If data is insufficient, reduce confidence and add to limitations
- Do not hallucinate - only reference data in the input
- Be concise - each field should be clear and actionable
"""


class RootCauseAnalyzer:
    """
    LLM-assisted root cause analyzer.
    
    Implements strict input/output contracts to prevent hallucination.
    Falls back to rules-based analysis if LLM is unavailable.
    """
    
    def __init__(self, config: LLMConfig, verbose: bool = False):
        self.config = config
        self.verbose = verbose
        self.llm_mode = config.provider
        
        # Initialize appropriate client
        if config.provider == "ollama":
            self.client = OllamaClient(config)
        elif config.provider == "mock":
            self.client = MockLLMClient()
        else:
            self.client = None  # Rules-only mode
    
    async def analyze(
        self,
        baseline_summary: Dict[str, Any],
        baseline_confidence: float,
        breaking_point: Optional[Dict[str, Any]],
        failure_timeline: List[Dict[str, Any]],
        failure_classification: str,
        classification_rationale: List[str],
        observed_metrics: Dict[str, Any],
    ) -> tuple[LLMOutputContract, str, str, float]:
        """
        Perform root cause analysis.
        
        Returns:
            Tuple of (output, mode, model, latency_ms)
        """
        # Build strict input contract
        input_contract = LLMInputContract(
            baseline_summary=baseline_summary,
            baseline_confidence=baseline_confidence,
            breaking_point=breaking_point,
            failure_timeline=failure_timeline,
            failure_classification=failure_classification,
            classification_rationale=classification_rationale,
            observed_metrics=observed_metrics,
        )
        
        # If LLM is available and enabled, use it
        if self.client and self.llm_mode == "ollama":
            llm_available = await self.client.check_available()
            
            if llm_available:
                return await self._analyze_with_llm(input_contract)
            else:
                if self.verbose:
                    print("  ⚠ Ollama not available - using rules-based analysis")
        
        # Fall back to rules-based analysis
        return self._analyze_with_rules(input_contract), "rules", "", 0.0
    
    async def _analyze_with_llm(
        self,
        input_contract: LLMInputContract,
    ) -> tuple[LLMOutputContract, str, str, float]:
        """Analyze using LLM with strict contracts"""
        
        # Build user prompt with input contract
        user_prompt = f"""Analyze this performance test failure:

INPUT DATA:
```json
{input_contract.to_json()}
```

Provide your analysis as JSON following the required output format."""

        # Call LLM
        response: LLMResponse = await self.client.generate(
            prompt=user_prompt,
            system_prompt=ROOT_CAUSE_SYSTEM_PROMPT,
            temperature=self.config.temperature,
            max_tokens=1500,
        )
        
        if self.verbose:
            print(f"  LLM response received ({response.latency_ms:.0f}ms, {response.tokens_used} tokens)")
        
        # Parse response with strict contract
        output = self._parse_llm_response(response.content, input_contract)
        
        return output, "ollama", response.model, response.latency_ms
    
    def _parse_llm_response(
        self,
        content: str,
        input_contract: LLMInputContract,
    ) -> LLMOutputContract:
        """Parse LLM response enforcing output contract"""
        
        # Try to extract JSON from response
        try:
            # Handle responses with markdown code blocks
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            
            data = json.loads(content)
            output = LLMOutputContract.from_dict(data)
            
            # Validate confidence is in range
            output.confidence = max(0.0, min(1.0, output.confidence))
            
            return output
            
        except (json.JSONDecodeError, KeyError) as e:
            # Fall back to rules-based if parsing fails
            if self.verbose:
                print(f"  ⚠ Failed to parse LLM response: {e}")
            
            return self._analyze_with_rules(input_contract)
    
    def _analyze_with_rules(
        self,
        input_contract: LLMInputContract,
    ) -> LLMOutputContract:
        """
        Rules-based fallback analysis.
        
        Provides deterministic output when LLM is unavailable.
        """
        classification = input_contract.failure_classification
        breaking_point = input_contract.breaking_point
        timeline = input_contract.failure_timeline
        rationale = input_contract.classification_rationale
        
        # Build summary based on classification
        summaries = {
            "capacity_exhaustion": "System reached resource limits causing throughput plateau",
            "latency_amplification": "Request latency increased exponentially under load due to resource contention",
            "error_driven_collapse": "Error rate exceeded threshold causing cascading failures",
            "instability_under_burst": "System failed to handle sudden traffic spike",
            "already_degraded_baseline": "System showed degradation even at baseline load levels",
            "no_failure": "No breaking point detected within test parameters",
        }
        
        primary_causes = {
            "capacity_exhaustion": "Resource saturation (CPU, memory, or connections)",
            "latency_amplification": "Queuing at a bottleneck resource",
            "error_driven_collapse": "Error propagation and cascade effect",
            "instability_under_burst": "Insufficient burst capacity or cold-start issues",
            "already_degraded_baseline": "Pre-existing system issues before load testing",
            "no_failure": "System capacity exceeds tested load levels",
        }
        
        root_cause_summary = summaries.get(classification, "Unknown failure pattern")
        primary_cause = primary_causes.get(classification, "Unknown cause")
        
        # Add context from breaking point
        if breaking_point:
            vus = breaking_point.get("vus_at_break", 0)
            rps = breaking_point.get("rps_at_break", 0)
            root_cause_summary = f"{root_cause_summary} at {vus} VUs ({rps:.1f} RPS)"
        
        # Extract contributing factors from timeline
        contributing_factors = []
        violation_types = set()
        for event in timeline:
            event_type = event.get("event_type", "")
            if event_type != "load_change":
                violation_types.add(event_type)
        
        for vtype in violation_types:
            if vtype == "error_rate_breach":
                contributing_factors.append("Error rate exceeded threshold")
            elif vtype == "latency_degradation":
                contributing_factors.append("Latency degradation detected")
            elif vtype == "throughput_plateau":
                contributing_factors.append("Throughput stopped scaling with load")
            elif vtype == "saturation":
                contributing_factors.append("Resource saturation detected")
        
        # Build assumptions and limitations
        assumptions = [
            "Load tests accurately simulate production traffic patterns",
            "Telemetry data reflects actual system behavior",
        ]
        
        limitations = ["Analysis based on rules-only (LLM unavailable)"]
        
        if input_contract.baseline_confidence < 0.5:
            limitations.append("Low baseline confidence - limited historical data")
        
        if len(timeline) < 3:
            limitations.append("Limited timeline events for detailed analysis")
        
        # Calculate confidence based on data quality
        confidence = 0.6  # Base for rules-only
        if rationale:
            confidence += 0.1
        if breaking_point:
            confidence += 0.1
        if len(timeline) >= 5:
            confidence += 0.1
        
        confidence = min(confidence, 0.85)  # Cap rules-based confidence
        
        # Detect failure pattern and get explanation
        pattern, pattern_explanation = self._detect_failure_pattern(
            timeline, breaking_point, input_contract.observed_metrics
        )
        
        return LLMOutputContract(
            root_cause_summary=root_cause_summary,
            primary_cause=primary_cause,
            contributing_factors=contributing_factors,
            confidence=confidence,
            assumptions=assumptions,
            limitations=limitations,
            failure_pattern=pattern,
            pattern_explanation=pattern_explanation,
        )
    
    def _detect_failure_pattern(
        self,
        timeline: List[Dict[str, Any]],
        breaking_point: Optional[Dict[str, Any]],
        observed_metrics: Dict[str, Any],
    ) -> tuple[str, str]:
        """
        Detect failure pattern from timeline and metrics.
        
        Patterns:
        - errors_before_latency: Errors appeared before latency degradation
        - latency_before_errors: Latency degraded before errors appeared
        - throughput_plateau: Throughput stopped scaling before other issues
        - spike_only_failure: System stable under gradual load, failed only on spike
        - no_recovery_after_drop: System did not recover after load decreased
        
        Returns:
            Tuple of (pattern_name, explanation)
        """
        if not timeline:
            return "insufficient_data", "Not enough timeline data to detect pattern"
        
        # Extract violation events (non-load-change)
        violations = [e for e in timeline if e.get("event_type") != "load_change"]
        
        if not violations:
            return "no_violations", "No threshold violations detected during testing"
        
        # Get first occurrences of each violation type
        first_error = None
        first_latency = None
        first_throughput = None
        first_saturation = None
        
        for i, v in enumerate(violations):
            vtype = v.get("event_type", "")
            if vtype == "error_rate_breach" and first_error is None:
                first_error = i
            elif vtype == "latency_degradation" and first_latency is None:
                first_latency = i
            elif vtype == "throughput_plateau" and first_throughput is None:
                first_throughput = i
            elif vtype == "saturation" and first_saturation is None:
                first_saturation = i
        
        # Check for spike-only failure
        test_types = set(e.get("test_type", "") for e in violations)
        if test_types == {"spike"} or all("spike" in t.lower() for t in test_types if t):
            return (
                "spike_only_failure",
                "System handled gradual load increases but failed under sudden traffic burst. "
                "Likely causes: connection pool exhaustion, cold cache, thread starvation, "
                "or lack of backpressure mechanisms. The system needs burst capacity buffers."
            )
        
        # Check for no recovery after load drop
        recovery_detected = self._check_recovery(timeline, observed_metrics)
        if not recovery_detected and len(violations) >= 2:
            # Check if there was a load drop after violations
            load_events = [e for e in timeline if e.get("event_type") == "load_change"]
            if len(load_events) >= 2:
                last_vus = [e.get("vus", 0) for e in load_events]
                if len(last_vus) >= 2 and last_vus[-1] < max(last_vus):
                    return (
                        "no_recovery_after_drop",
                        "System did not recover after load decreased. "
                        "Likely causes: resource leaks, stuck connections, thread deadlocks, "
                        "or cascading failure state. Requires investigation of cleanup mechanisms."
                    )
        
        # Check for throughput plateau first
        if first_throughput is not None:
            if first_error is None or first_throughput < first_error:
                if first_latency is None or first_throughput < first_latency:
                    return (
                        "throughput_plateau",
                        "Throughput stopped scaling before errors or latency issues appeared. "
                        "Likely causes: external bottleneck (database, downstream service), "
                        "connection pool limits, or I/O saturation. Check dependency health."
                    )
        
        # Check errors before latency
        if first_error is not None and first_latency is not None:
            if first_error < first_latency:
                return (
                    "errors_before_latency",
                    "Errors appeared before significant latency degradation. "
                    "Likely causes: hard resource limits (memory OOM, file descriptors), "
                    "explicit rate limiting, or downstream service failures. "
                    "The system is failing fast rather than degrading gracefully."
                )
        
        # Check latency before errors
        if first_latency is not None and first_error is not None:
            if first_latency < first_error:
                return (
                    "latency_before_errors",
                    "Latency degraded significantly before errors appeared. "
                    "Likely causes: request queuing at a bottleneck, database contention, "
                    "lock contention, or garbage collection pressure. "
                    "The system queued requests until timeouts caused errors."
                )
        
        # Latency only (no errors)
        if first_latency is not None and first_error is None:
            return (
                "latency_amplification_only",
                "Only latency degradation observed without error threshold breach. "
                "Likely causes: gradual resource contention, inefficient algorithms "
                "under load, or cache misses. System remained functional but slow."
            )
        
        # Errors only (no latency degradation)
        if first_error is not None and first_latency is None:
            return (
                "errors_without_latency",
                "Errors occurred without prior latency degradation. "
                "Likely causes: hard limits being hit (connection refused, OOM killer), "
                "circuit breakers tripping, or external service failures."
            )
        
        # Saturation pattern
        if first_saturation is not None:
            return (
                "resource_saturation",
                "Resource saturation detected during load increase. "
                "Likely causes: CPU saturation, memory pressure, or I/O bottleneck. "
                "Consider scaling resources or optimizing hot paths."
            )
        
        return "mixed_failure", "Multiple failure modes detected simultaneously"
    
    def _check_recovery(
        self,
        timeline: List[Dict[str, Any]],
        observed_metrics: Dict[str, Any],
    ) -> bool:
        """Check if system showed recovery after load drop"""
        # Look for pattern: high errors followed by low errors
        error_events = [
            e for e in timeline 
            if e.get("event_type") == "error_rate_breach"
        ]
        
        if len(error_events) < 2:
            return True  # Not enough data to determine
        
        # Check test types for recovery pattern
        test_types = observed_metrics.get("test_types", [])
        if "spike" in test_types or "stress" in test_types:
            # If we have multiple test types, assume some recovery opportunity
            return True
        
        return False


def run_root_cause_analysis(
    config: LLMConfig,
    baseline_summary: Dict[str, Any],
    baseline_confidence: float,
    breaking_point: Optional[Dict[str, Any]],
    failure_timeline: List[Dict[str, Any]],
    failure_classification: str,
    classification_rationale: List[str],
    observed_metrics: Dict[str, Any],
    verbose: bool = False,
) -> tuple[LLMOutputContract, str, str, float]:
    """
    Synchronous wrapper for root cause analysis.
    
    Returns:
        Tuple of (output, mode, model, latency_ms)
    """
    analyzer = RootCauseAnalyzer(config, verbose)
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(
        analyzer.analyze(
            baseline_summary=baseline_summary,
            baseline_confidence=baseline_confidence,
            breaking_point=breaking_point,
            failure_timeline=failure_timeline,
            failure_classification=failure_classification,
            classification_rationale=classification_rationale,
            observed_metrics=observed_metrics,
        )
    )
