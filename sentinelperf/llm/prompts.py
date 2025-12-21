"""Prompt templates for LLM analysis in SentinelPerf"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from sentinelperf.core.state import BreakingPoint, LoadTestResult


@dataclass
class PromptContext:
    """Context data for prompt generation"""
    breaking_point: BreakingPoint
    load_results: List[LoadTestResult]
    telemetry_summary: Optional[Dict[str, Any]] = None


class PromptTemplates:
    """
    Prompt templates for LLM-based analysis.
    
    All prompts enforce the LLM rules:
    - Only use provided metrics (no invention)
    - Step-by-step reasoning required
    - Confidence must be based on evidence
    """
    
    SYSTEM_PROMPT = """You are SentinelPerf AI, an expert performance engineering analyst.

STRICT RULES YOU MUST FOLLOW:
1. You may ONLY reference metrics and data explicitly provided in the input
2. You may NOT invent, assume, or infer any metrics not provided
3. You MUST explain your reasoning step-by-step
4. You MUST assign confidence scores based ONLY on the strength of provided evidence
5. If data is insufficient, state "Insufficient data" rather than guessing
6. Your analysis must be reproducible - another expert with the same data should reach the same conclusions

You are an explanation and reasoning layer, NOT a decision oracle.
"""

    ROOT_CAUSE_TEMPLATE = """Analyze the following performance test results to determine the root cause of system failure.

## Breaking Point Detected
- VUs at break: {vus_at_break}
- RPS at break: {rps_at_break:.1f}
- Failure type: {failure_type}
- Threshold exceeded: {threshold_exceeded}
- Observed value: {observed_value}
- Threshold value: {threshold_value}
- Detection confidence: {confidence:.0%}

## Observed Signals
{signals}

## Load Test Progression
{load_progression}

## Your Task
1. Identify the PRIMARY root cause based ONLY on the data above
2. Explain your reasoning step-by-step
3. List supporting evidence (only from provided data)
4. Provide actionable recommendations with confidence scores
5. Assign an overall confidence score based on evidence strength

## Response Format (JSON)
{{
  "primary_cause": "<one sentence description>",
  "reasoning_steps": [
    "Step 1: ...",
    "Step 2: ...",
    ...
  ],
  "supporting_evidence": [
    "<evidence 1>",
    "<evidence 2>",
    ...
  ],
  "recommendations": [
    {{
      "action": "<specific action>",
      "rationale": "<why this helps>",
      "confidence": <0.0-1.0>,
      "priority": <1-5>,
      "estimated_impact": "<Low/Medium/High>"
    }}
  ],
  "overall_confidence": <0.0-1.0>,
  "confidence_rationale": "<why this confidence level>"
}}
"""

    @classmethod
    def build_root_cause_prompt(cls, context: PromptContext) -> str:
        """Build root cause analysis prompt from context"""
        
        bp = context.breaking_point
        
        # Format signals
        signals = "\n".join([f"- {s}" for s in bp.signals]) if bp.signals else "- No additional signals"
        
        # Format load progression
        load_lines = []
        for r in context.load_results:
            load_lines.append(
                f"| {r.test_type:8} | {r.vus:4} VUs | "
                f"{r.throughput_rps:7.1f} RPS | "
                f"{r.error_rate:6.2%} errors | "
                f"P95: {r.latency_p95_ms:6.0f}ms |"
            )
        
        load_progression = "\n".join(load_lines) if load_lines else "No load test data available"
        
        return cls.ROOT_CAUSE_TEMPLATE.format(
            vus_at_break=bp.vus_at_break,
            rps_at_break=bp.rps_at_break,
            failure_type=bp.failure_type,
            threshold_exceeded=bp.threshold_exceeded,
            observed_value=bp.observed_value,
            threshold_value=bp.threshold_value,
            confidence=bp.confidence,
            signals=signals,
            load_progression=load_progression,
        )

    RECOMMENDATION_TEMPLATE = """Based on the root cause analysis, provide specific fix recommendations.

## Root Cause
{primary_cause}

## Confidence: {confidence:.0%}

## Evidence
{evidence}

## Context
- Breaking point: {vus_at_break} VUs at {rps_at_break:.1f} RPS
- Failure type: {failure_type}

## Your Task
Provide 3-5 actionable recommendations ranked by:
1. Likelihood of addressing the root cause
2. Implementation complexity
3. Expected impact

For each recommendation:
- Be specific (not "optimize database" but "add index on users.email column")
- Explain WHY this addresses the root cause
- Estimate the expected performance improvement
- Note any risks or dependencies

## Response Format (JSON)
{{
  "recommendations": [
    {{
      "action": "<specific action>",
      "rationale": "<why this addresses root cause>",
      "expected_improvement": "<quantified if possible>",
      "implementation_complexity": "<Low/Medium/High>",
      "risks": ["<risk 1>", "<risk 2>"],
      "priority": <1-5>
    }}
  ]
}}
"""

    @classmethod
    def build_recommendation_prompt(
        cls,
        primary_cause: str,
        confidence: float,
        evidence: List[str],
        breaking_point: BreakingPoint,
    ) -> str:
        """Build recommendation prompt"""
        
        evidence_text = "\n".join([f"- {e}" for e in evidence]) if evidence else "- No evidence available"
        
        return cls.RECOMMENDATION_TEMPLATE.format(
            primary_cause=primary_cause,
            confidence=confidence,
            evidence=evidence_text,
            vus_at_break=breaking_point.vus_at_break,
            rps_at_break=breaking_point.rps_at_break,
            failure_type=breaking_point.failure_type,
        )
