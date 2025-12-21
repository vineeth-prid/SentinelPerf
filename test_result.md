# SentinelPerf AI Test Results

## Overview
SentinelPerf AI is a CLI-first, autonomous performance engineering agent that:
1. Analyzes telemetry data to infer traffic patterns
2. Generates and executes load/stress/spike tests using k6
3. Detects breaking points using rules-based analysis
4. Classifies failures with deterministic analysis
5. Performs LLM-assisted root cause analysis (Ollama/Qwen2.5)

## Test Status: ✅ PASSED

### Phase 5 Complete: LLM-Assisted Root Cause Analysis

#### Architecture Changes
- **Renamed** `RootCauseAnalysis` (old) → `DeterministicFailureAnalysis` (rules-based)
- **New** `RootCauseAnalysis` for LLM output with strict contracts
- **Removed** recommendations (deferred to Phase 6)

#### LLM Integration
- **Input Contract**: Strict JSON - no raw logs, no k6 output
- **Output Contract**: Structured JSON with confidence, assumptions, limitations
- **Fallback**: Rules-only mode when Ollama unavailable

#### Key Files Modified
- `/app/sentinelperf/core/state.py` - New dataclasses
- `/app/sentinelperf/analysis/root_cause.py` - LLM analyzer with contracts
- `/app/sentinelperf/core/agent.py` - Refactored root cause node
- `/app/sentinelperf/reports/markdown.py` - Updated report format
- `/app/sentinelperf/reports/json_report.py` - Updated JSON structure

### Features Tested

#### 1. Breaking Point Detection ✅
- Detects first sustained violation of performance thresholds

#### 2. Failure Timeline Construction ✅
- Ordered timeline with load changes and violations

#### 3. Deterministic Classification ✅
- Categories: CAPACITY_EXHAUSTION, LATENCY_AMPLIFICATION, ERROR_DRIVEN_COLLAPSE, etc.

#### 4. Root Cause Analysis ✅
- LLM mode: Ollama/Qwen2.5 with strict input/output contracts
- Rules mode: Fallback when LLM unavailable
- Output: summary, primary_cause, contributing_factors, confidence, assumptions, limitations

#### 5. Report Generation ✅
- Markdown with timeline and root cause analysis
- JSON with full structure for CI/CD

### Test Commands
```bash
# Run analysis with Ollama (if available)
python -m sentinelperf.cli run --env=test --verbose

# Run with rules-only mode
python -m sentinelperf.cli run --env=test --llm-mode=rules --verbose
```

### Configuration Example
```yaml
llm:
  provider: ollama
  model: qwen2.5:14b
  timeout: 60
```

### Incorporate User Feedback
- LLM is strictly limited to EXPLAIN, not CHANGE
- No recommendations in Phase 5 (deferred to Phase 6)
- Output is structured JSON, not prose
