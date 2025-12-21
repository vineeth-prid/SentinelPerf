"""Recommendation Engine for SentinelPerf

Implements deterministic recommendation mapping based on failure classification.
LLM is optional and can ONLY polish existing recommendations.

Phase 6 Scope:
- Generate rule-based recommendations
- Optional LLM polishing (rephrasing only)
- Strict output contract

NOT allowed:
- Auto-fix
- Infra changes
- Scaling actions
- Code modification
- Adaptive retesting
"""

import json
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from enum import Enum

from sentinelperf.llm.client import OllamaClient, LLMResponse
from sentinelperf.config.schema import LLMConfig


class RiskLevel(str, Enum):
    """Risk levels for recommendations"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class Recommendation:
    """Single recommendation with all required fields"""
    action: str
    rationale: str
    expected_impact: str
    risk: str  # LOW, MEDIUM, HIGH
    confidence: float  # 0.0 to 1.0
    priority: int = 1  # 1 = highest
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "rationale": self.rationale,
            "expected_impact": self.expected_impact,
            "risk": self.risk,
            "confidence": self.confidence,
            "priority": self.priority,
        }


@dataclass
class RecommendationResult:
    """Complete recommendation output"""
    recommendations: List[Recommendation] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    polished_by_llm: bool = False
    llm_model: str = ""
    llm_latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommendations": [r.to_dict() for r in self.recommendations],
            "limitations": self.limitations,
            "polished_by_llm": self.polished_by_llm,
            "llm_model": self.llm_model,
            "llm_latency_ms": self.llm_latency_ms,
        }


# =============================================================================
# DETERMINISTIC RECOMMENDATION TEMPLATES
# =============================================================================

RECOMMENDATION_TEMPLATES: Dict[str, List[Recommendation]] = {
    "capacity_exhaustion": [
        Recommendation(
            action="Increase concurrency limits (thread pools, connection pools)",
            rationale="System hit resource limits causing throughput plateau",
            expected_impact="Higher throughput ceiling before saturation",
            risk=RiskLevel.MEDIUM.value,
            confidence=0.80,
            priority=1,
        ),
        Recommendation(
            action="Scale instances horizontally (add replicas)",
            rationale="Single instance reached capacity before load test max",
            expected_impact="Linear throughput scaling with instance count",
            risk=RiskLevel.MEDIUM.value,
            confidence=0.75,
            priority=2,
        ),
        Recommendation(
            action="Profile and optimize CPU/memory hotspots",
            rationale="Resource contention suggests inefficient code paths",
            expected_impact="Better resource utilization per request",
            risk=RiskLevel.LOW.value,
            confidence=0.70,
            priority=3,
        ),
    ],
    
    "latency_amplification": [
        Recommendation(
            action="Investigate blocking calls (DB queries, external APIs)",
            rationale="Latency increased without errors, indicating queuing",
            expected_impact="Reduced request queuing and lower P95",
            risk=RiskLevel.LOW.value,
            confidence=0.80,
            priority=1,
        ),
        Recommendation(
            action="Check database connection pool sizing and query plans",
            rationale="DB contention is common cause of latency amplification",
            expected_impact="Faster database response times under load",
            risk=RiskLevel.LOW.value,
            confidence=0.75,
            priority=2,
        ),
        Recommendation(
            action="Add caching for frequently accessed data",
            rationale="Reduce load on slow backend services",
            expected_impact="Lower latency for cached requests",
            risk=RiskLevel.LOW.value,
            confidence=0.70,
            priority=3,
        ),
    ],
    
    "error_driven_collapse": [
        Recommendation(
            action="Inspect error types and downstream dependencies",
            rationale="Errors cascaded causing system collapse",
            expected_impact="Identify root cause of error propagation",
            risk=RiskLevel.LOW.value,
            confidence=0.85,
            priority=1,
        ),
        Recommendation(
            action="Implement circuit breaker pattern",
            rationale="Prevent cascade failures from propagating",
            expected_impact="Graceful degradation instead of collapse",
            risk=RiskLevel.MEDIUM.value,
            confidence=0.80,
            priority=2,
        ),
        Recommendation(
            action="Add rate limiting at entry points",
            rationale="Protect system from overload-induced errors",
            expected_impact="Controlled rejection instead of uncontrolled failure",
            risk=RiskLevel.MEDIUM.value,
            confidence=0.75,
            priority=3,
        ),
    ],
    
    "instability_under_burst": [
        Recommendation(
            action="Add rate limiting with token bucket or leaky bucket",
            rationale="System failed specifically during traffic spike",
            expected_impact="Smooth out traffic bursts to manageable rate",
            risk=RiskLevel.MEDIUM.value,
            confidence=0.80,
            priority=1,
        ),
        Recommendation(
            action="Warm connection pools and caches before expected spikes",
            rationale="Cold start issues may amplify burst impact",
            expected_impact="Faster response to sudden load increases",
            risk=RiskLevel.LOW.value,
            confidence=0.70,
            priority=2,
        ),
        Recommendation(
            action="Implement request queuing with backpressure",
            rationale="Queue excess requests instead of failing them",
            expected_impact="Higher success rate during bursts",
            risk=RiskLevel.MEDIUM.value,
            confidence=0.75,
            priority=3,
        ),
    ],
    
    "already_degraded_baseline": [
        Recommendation(
            action="Fix baseline errors before load testing",
            rationale="System already degraded at minimal load",
            expected_impact="Establish healthy baseline for accurate testing",
            risk=RiskLevel.LOW.value,
            confidence=0.90,
            priority=1,
        ),
        Recommendation(
            action="Check service health and dependencies",
            rationale="Baseline errors suggest infrastructure issues",
            expected_impact="Identify and resolve pre-existing problems",
            risk=RiskLevel.LOW.value,
            confidence=0.85,
            priority=2,
        ),
        Recommendation(
            action="Review recent deployments or configuration changes",
            rationale="Baseline degradation may be from recent changes",
            expected_impact="Rollback if recent change caused degradation",
            risk=RiskLevel.LOW.value,
            confidence=0.70,
            priority=3,
        ),
    ],
    
    "no_failure": [
        Recommendation(
            action="Increase load test intensity to find actual limits",
            rationale="No breaking point found within test parameters",
            expected_impact="Establish true capacity ceiling",
            risk=RiskLevel.LOW.value,
            confidence=0.80,
            priority=1,
        ),
        Recommendation(
            action="Document current capacity as baseline",
            rationale="System handled all test scenarios successfully",
            expected_impact="Reference point for future capacity planning",
            risk=RiskLevel.LOW.value,
            confidence=0.90,
            priority=2,
        ),
    ],
}


# LLM polishing prompt - ONLY rephrases, cannot invent
LLM_POLISH_SYSTEM_PROMPT = """You are a senior engineer polishing runbook entries.

Your role is STRICTLY LIMITED to:
1. REPHRASE recommendations for clarity
2. ADD rationale context where helpful
3. IMPROVE wording for readability

You are NOT allowed to:
- Invent new recommendations
- Change the action itself
- Override priority or risk levels
- Add recommendations not in the input

Think of your task as: "Editing a colleague's draft for clarity"

OUTPUT FORMAT:
Return a JSON array with the same structure, only improving wording:
[
  {
    "action": "improved action wording",
    "rationale": "improved rationale",
    "expected_impact": "improved impact description",
    "risk": "LOW|MEDIUM|HIGH",  // KEEP SAME
    "confidence": 0.78,  // KEEP SAME
    "priority": 1  // KEEP SAME
  }
]

RULES:
- Keep ALL original recommendations (no deletions)
- Keep risk, confidence, priority EXACTLY the same
- Only improve wording, not substance
"""


class RecommendationEngine:
    """
    Deterministic recommendation engine with optional LLM polishing.
    
    Default: Rules-only (no LLM calls)
    Optional: LLM polishing if `polish_with_llm: true` in config
    """
    
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        polish_with_llm: bool = False,
        verbose: bool = False,
    ):
        self.llm_config = llm_config
        self.polish_with_llm = polish_with_llm
        self.verbose = verbose
        
        # Only initialize LLM client if polishing is enabled
        self.llm_client = None
        if polish_with_llm and llm_config and llm_config.provider == "ollama":
            self.llm_client = OllamaClient(llm_config)
    
    def generate(
        self,
        failure_classification: str,
        breaking_point: Optional[Dict[str, Any]] = None,
        root_cause_confidence: float = 0.5,
    ) -> RecommendationResult:
        """
        Generate recommendations based on failure classification.
        
        1. Get deterministic recommendations from templates
        2. Adjust confidence based on root cause analysis
        3. Optionally polish with LLM (if enabled)
        """
        # Step 1: Get base recommendations from templates
        base_recs = self._get_base_recommendations(failure_classification)
        
        # Step 2: Adjust confidence based on root cause analysis
        adjusted_recs = self._adjust_confidence(base_recs, root_cause_confidence)
        
        # Step 3: Add context from breaking point
        contextualized_recs = self._add_context(adjusted_recs, breaking_point)
        
        # Step 4: Build limitations
        limitations = self._build_limitations(failure_classification, breaking_point)
        
        # Step 5: Optional LLM polishing (async)
        if self.polish_with_llm and self.llm_client:
            return self._polish_with_llm_sync(contextualized_recs, limitations)
        
        return RecommendationResult(
            recommendations=contextualized_recs,
            limitations=limitations,
            polished_by_llm=False,
        )
    
    def _get_base_recommendations(self, classification: str) -> List[Recommendation]:
        """Get base recommendations from template"""
        templates = RECOMMENDATION_TEMPLATES.get(classification, [])
        
        if not templates:
            # Unknown classification - provide generic recommendation
            return [
                Recommendation(
                    action="Review test results and failure timeline manually",
                    rationale=f"Unknown classification: {classification}",
                    expected_impact="Manual analysis may reveal specific issues",
                    risk=RiskLevel.LOW.value,
                    confidence=0.50,
                    priority=1,
                ),
            ]
        
        # Deep copy to avoid mutating templates
        return [
            Recommendation(
                action=r.action,
                rationale=r.rationale,
                expected_impact=r.expected_impact,
                risk=r.risk,
                confidence=r.confidence,
                priority=r.priority,
            )
            for r in templates
        ]
    
    def _adjust_confidence(
        self,
        recs: List[Recommendation],
        root_cause_confidence: float,
    ) -> List[Recommendation]:
        """Adjust recommendation confidence based on root cause analysis"""
        # Scale confidence by root cause confidence
        for rec in recs:
            # Blend recommendation confidence with root cause confidence
            rec.confidence = rec.confidence * 0.7 + root_cause_confidence * 0.3
            rec.confidence = min(rec.confidence, 0.95)
        
        return recs
    
    def _add_context(
        self,
        recs: List[Recommendation],
        breaking_point: Optional[Dict[str, Any]],
    ) -> List[Recommendation]:
        """Add context from breaking point to recommendations"""
        if not breaking_point:
            return recs
        
        vus = breaking_point.get("vus_at_break", 0)
        rps = breaking_point.get("rps_at_break", 0)
        failure_type = breaking_point.get("failure_type", "unknown")
        
        # Add context to first recommendation
        if recs and vus > 0:
            recs[0].rationale = (
                f"{recs[0].rationale} "
                f"(detected at {vus} VUs, {rps:.1f} RPS via {failure_type})"
            )
        
        return recs
    
    def _build_limitations(
        self,
        classification: str,
        breaking_point: Optional[Dict[str, Any]],
    ) -> List[str]:
        """Build list of limitations for recommendations"""
        limitations = []
        
        # General limitations
        limitations.append("Recommendations are guidance, not prescriptions")
        limitations.append("Actual fixes require system-specific investigation")
        
        # Classification-specific limitations
        if classification == "no_failure":
            limitations.append("No failure detected - recommendations are exploratory")
        
        if classification == "already_degraded_baseline":
            limitations.append("Baseline issues must be fixed before further analysis")
        
        # Breaking point confidence
        if breaking_point:
            confidence = breaking_point.get("confidence", 0)
            if confidence < 0.7:
                limitations.append(f"Breaking point detection confidence is {confidence:.0%}")
        
        return limitations
    
    def _polish_with_llm_sync(
        self,
        recs: List[Recommendation],
        limitations: List[str],
    ) -> RecommendationResult:
        """Synchronous wrapper for LLM polishing"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self._polish_with_llm(recs, limitations)
        )
    
    async def _polish_with_llm(
        self,
        recs: List[Recommendation],
        limitations: List[str],
    ) -> RecommendationResult:
        """Polish recommendations with LLM (wording only)"""
        
        # Check if LLM is available
        llm_available = await self.llm_client.check_available()
        
        if not llm_available:
            if self.verbose:
                print("  ⚠ Ollama not available - skipping LLM polish")
            limitations.append("LLM polishing unavailable")
            return RecommendationResult(
                recommendations=recs,
                limitations=limitations,
                polished_by_llm=False,
            )
        
        # Build prompt with recommendations to polish
        recs_json = json.dumps([r.to_dict() for r in recs], indent=2)
        user_prompt = f"""Polish these recommendations for clarity (keep same structure, only improve wording):

```json
{recs_json}
```

Return improved JSON array with same fields."""

        # Call LLM
        response: LLMResponse = await self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=LLM_POLISH_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=2000,
        )
        
        if self.verbose:
            print(f"  LLM polish response ({response.latency_ms:.0f}ms)")
        
        # Parse polished recommendations
        polished_recs = self._parse_polished_response(response.content, recs)
        
        return RecommendationResult(
            recommendations=polished_recs,
            limitations=limitations,
            polished_by_llm=True,
            llm_model=response.model,
            llm_latency_ms=response.latency_ms,
        )
    
    def _parse_polished_response(
        self,
        content: str,
        original_recs: List[Recommendation],
    ) -> List[Recommendation]:
        """Parse LLM polished response, preserving originals on failure"""
        try:
            # Extract JSON from response
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            
            polished_data = json.loads(content)
            
            if not isinstance(polished_data, list):
                return original_recs
            
            # Validate and merge polished data
            polished_recs = []
            for i, orig in enumerate(original_recs):
                if i < len(polished_data):
                    pol = polished_data[i]
                    # Keep original risk, confidence, priority - only update wording
                    polished_recs.append(Recommendation(
                        action=pol.get("action", orig.action),
                        rationale=pol.get("rationale", orig.rationale),
                        expected_impact=pol.get("expected_impact", orig.expected_impact),
                        risk=orig.risk,  # NEVER change from LLM
                        confidence=orig.confidence,  # NEVER change from LLM
                        priority=orig.priority,  # NEVER change from LLM
                    ))
                else:
                    polished_recs.append(orig)
            
            return polished_recs
            
        except (json.JSONDecodeError, KeyError) as e:
            if self.verbose:
                print(f"  ⚠ Failed to parse LLM polish response: {e}")
            return original_recs


def generate_recommendations(
    failure_classification: str,
    breaking_point: Optional[Dict[str, Any]] = None,
    root_cause_confidence: float = 0.5,
    llm_config: Optional[LLMConfig] = None,
    polish_with_llm: bool = False,
    verbose: bool = False,
) -> RecommendationResult:
    """
    Convenience function to generate recommendations.
    
    Default: Rules-only (no LLM)
    Set `polish_with_llm=True` to enable LLM polishing
    """
    engine = RecommendationEngine(
        llm_config=llm_config,
        polish_with_llm=polish_with_llm,
        verbose=verbose,
    )
    
    return engine.generate(
        failure_classification=failure_classification,
        breaking_point=breaking_point,
        root_cause_confidence=root_cause_confidence,
    )
