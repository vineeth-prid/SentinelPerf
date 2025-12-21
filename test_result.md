# SentinelPerf AI Test Results

## Overview
SentinelPerf AI is a CLI-first, autonomous performance engineering agent that:
1. Analyzes telemetry data to infer traffic patterns
2. Generates and executes load/stress/spike tests using k6
3. Detects breaking points using rules-based analysis
4. Classifies failures with deterministic analysis
5. Performs LLM-assisted root cause analysis (Ollama/Qwen2.5)
6. Generates deterministic recommendations with optional LLM polishing

## Test Status: ✅ PASSED

### Phase 6 Complete: Recommendations Engine

#### Architecture
- **Deterministic Mapping**: Classification → Recommendation templates
- **LLM Polishing**: OFF by default (token-safe)
- **Strict Output**: action, rationale, expected_impact, risk, confidence

#### Recommendation Templates
| Classification | Primary Recommendation |
|---------------|----------------------|
| CAPACITY_EXHAUSTION | Increase concurrency limits |
| LATENCY_AMPLIFICATION | Investigate blocking calls |
| ERROR_DRIVEN_COLLAPSE | Inspect error types |
| INSTABILITY_UNDER_BURST | Add rate limiting |
| ALREADY_DEGRADED_BASELINE | Fix baseline errors |

#### Key Files
- `/app/sentinelperf/analysis/recommendations.py` - Recommendation engine
- `/app/sentinelperf/config/schema.py` - RecommendationsConfig added
- `/app/sentinelperf/core/agent.py` - `_node_recommendations` added

### Configuration
```yaml
recommendations:
  enabled: true
  polish_with_llm: false  # Token-safe: LLM OFF by default
```

### Test Scenarios Validated
1. **already_degraded_baseline** → Fix baseline errors
2. **error_driven_collapse** → Circuit breaker, rate limiting
3. **no_failure** → Increase test intensity

### Phase 6 NOT Allowed
- ❌ Auto-fix
- ❌ Infra changes
- ❌ Scaling actions
- ❌ Code modification
- ❌ Adaptive retesting

### Incorporate User Feedback
- LLM polishing is OFF by default
- Recommendations are guidance, not prescriptions
- Risk/confidence/priority are NEVER changed by LLM
