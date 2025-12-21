"""Root cause analysis for SentinelPerf"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from sentinelperf.core.state import (
    BreakingPoint,
    LoadTestResult,
    RootCauseAnalysis,
    TelemetryInsight,
)
from sentinelperf.config.schema import LLMConfig


@dataclass
class AnalysisEvidence:
    """Evidence supporting a root cause hypothesis"""
    source: str  # metric, trace, log
    observation: str
    relevance: float  # 0.0 to 1.0


@dataclass
class Recommendation:
    """A fix recommendation with confidence"""
    action: str
    rationale: str
    confidence: float
    priority: int  # 1 = highest
    estimated_impact: str


class RootCauseAnalyzer:
    """
    Analyzes breaking point to determine root cause.
    
    Supports three modes:
    1. ollama: Use local LLM for analysis
    2. rules: Use rule-based heuristics
    3. mock: Return placeholder analysis (testing)
    
    LLM Rules (strictly enforced):
    - LLM may NOT invent metrics
    - LLM may NOT infer causes without observed signals
    - LLM must explain reasoning step-by-step
    - LLM must assign confidence scores based on signal strength
    """
    
    def __init__(
        self,
        llm_config: LLMConfig,
        mode: str = "ollama",
    ):
        self.config = llm_config
        self.mode = mode
        self._llm_client = None
    
    async def analyze(
        self,
        breaking_point: BreakingPoint,
        load_results: List[LoadTestResult],
        telemetry: Optional[TelemetryInsight] = None,
    ) -> RootCauseAnalysis:
        """
        Analyze breaking point to determine root cause.
        
        Args:
            breaking_point: Detected breaking point
            load_results: All load test results
            telemetry: Optional telemetry data for context
            
        Returns:
            RootCauseAnalysis with cause, confidence, and recommendations
        """
        if self.mode == "mock":
            return self._mock_analysis(breaking_point)
        elif self.mode == "rules":
            return self._rules_analysis(breaking_point, load_results)
        else:
            return await self._llm_analysis(breaking_point, load_results, telemetry)
    
    def _mock_analysis(self, breaking_point: BreakingPoint) -> RootCauseAnalysis:
        """Return mock analysis for testing"""
        return RootCauseAnalysis(
            primary_cause="Mock analysis - no actual root cause determined",
            confidence=0.0,
            supporting_evidence=["Mock mode enabled"],
            reasoning_steps=["Mock mode - no reasoning performed"],
            recommendations=[{
                "action": "Run with --llm-mode=ollama for real analysis",
                "confidence": 1.0,
                "priority": 1,
            }],
            llm_mode="mock",
        )
    
    def _rules_analysis(
        self,
        breaking_point: BreakingPoint,
        load_results: List[LoadTestResult],
    ) -> RootCauseAnalysis:
        """Rule-based heuristic analysis"""
        
        reasoning_steps = []
        evidence = []
        recommendations = []
        
        # Step 1: Identify failure type
        reasoning_steps.append(f"Step 1: Identified failure type as '{breaking_point.failure_type}'")
        
        # Step 2: Analyze based on failure type
        if breaking_point.failure_type == "error_rate":
            reasoning_steps.append("Step 2: High error rate indicates capacity or stability issue")
            
            # Check if errors correlate with VU count
            error_progression = [(r.vus, r.error_rate) for r in load_results if r.error_rate > 0]
            if error_progression:
                reasoning_steps.append(f"Step 3: Error progression: {error_progression}")
                
                if len(error_progression) >= 2:
                    # Errors increase with load
                    primary_cause = "Connection pool exhaustion or resource saturation"
                    evidence.append("Errors increase proportionally with VU count")
                    recommendations.append({
                        "action": "Increase connection pool size or add horizontal scaling",
                        "rationale": "Errors correlate with concurrent connections",
                        "confidence": 0.7,
                        "priority": 1,
                        "estimated_impact": "High - directly addresses capacity bottleneck",
                    })
                else:
                    primary_cause = "Application error under load"
                    evidence.append("Errors appear at specific load threshold")
                    recommendations.append({
                        "action": "Review application logs for error patterns",
                        "rationale": "Sudden errors suggest application-level issue",
                        "confidence": 0.6,
                        "priority": 1,
                        "estimated_impact": "Medium - requires log analysis",
                    })
            else:
                primary_cause = "Unknown error source"
                
        elif breaking_point.failure_type == "p95_latency":
            reasoning_steps.append("Step 2: High latency indicates performance degradation")
            
            # Check latency progression
            latency_progression = [(r.vus, r.latency_p95_ms) for r in load_results if r.latency_p95_ms > 0]
            if latency_progression:
                reasoning_steps.append(f"Step 3: Latency progression: {latency_progression}")
                
                # Check if latency growth is linear or exponential
                if len(latency_progression) >= 3:
                    # Simple linear vs exponential check
                    growth_rates = []
                    for i in range(1, len(latency_progression)):
                        prev = latency_progression[i-1][1]
                        curr = latency_progression[i][1]
                        if prev > 0:
                            growth_rates.append(curr / prev)
                    
                    avg_growth = sum(growth_rates) / len(growth_rates) if growth_rates else 1.0
                    
                    if avg_growth > 1.5:
                        primary_cause = "Database or downstream service bottleneck"
                        evidence.append("Exponential latency growth suggests queuing")
                        recommendations.append({
                            "action": "Check database query performance and connection limits",
                            "rationale": "Exponential latency growth typically indicates queuing at a bottleneck",
                            "confidence": 0.75,
                            "priority": 1,
                            "estimated_impact": "High - likely the primary bottleneck",
                        })
                    else:
                        primary_cause = "CPU or memory pressure on application server"
                        evidence.append("Linear latency growth suggests compute limitation")
                        recommendations.append({
                            "action": "Monitor CPU and memory during load test",
                            "rationale": "Linear degradation suggests compute resource limitation",
                            "confidence": 0.65,
                            "priority": 1,
                            "estimated_impact": "Medium - may require infrastructure scaling",
                        })
                else:
                    primary_cause = "Performance degradation under load"
            else:
                primary_cause = "Unknown latency source"
        else:
            primary_cause = f"Unknown failure type: {breaking_point.failure_type}"
            reasoning_steps.append(f"Step 2: Unrecognized failure type '{breaking_point.failure_type}'")
        
        # Add evidence from breaking point signals
        for signal in breaking_point.signals:
            evidence.append(signal)
        
        # Calculate confidence based on evidence strength
        confidence = min(0.5 + (len(evidence) * 0.1), 0.85)
        
        return RootCauseAnalysis(
            primary_cause=primary_cause,
            confidence=confidence,
            supporting_evidence=evidence,
            reasoning_steps=reasoning_steps,
            recommendations=recommendations,
            llm_mode="rules",
        )
    
    async def _llm_analysis(
        self,
        breaking_point: BreakingPoint,
        load_results: List[LoadTestResult],
        telemetry: Optional[TelemetryInsight] = None,
    ) -> RootCauseAnalysis:
        """
        LLM-based root cause analysis using Ollama.
        
        Strictly follows the LLM rules:
        - Only uses observed metrics (no invention)
        - Step-by-step reasoning
        - Confidence based on signal strength
        """
        # TODO: Implement Ollama integration
        # For now, fall back to rules-based analysis
        
        # This will be implemented with:
        # 1. Structured prompt with observed metrics only
        # 2. Chain-of-thought prompting for reasoning
        # 3. Confidence calibration based on evidence count
        
        return self._rules_analysis(breaking_point, load_results)
