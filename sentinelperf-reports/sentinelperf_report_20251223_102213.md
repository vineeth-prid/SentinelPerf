# SentinelPerf Analysis Report

**Target:** http://localhost:8765  
**Environment:** test  
**Generated:** 2025-12-23 10:22:13 UTC  
**Status:** ⚠️ report_generation

---

## Executive Summary

The system reached its **breaking point at 27 virtual users** (26.0 requests/second), where error_rate_breach exceeded acceptable thresholds.

**Primary Root Cause:** Resource saturation (CPU, memory, or connections) (confidence: 85%)

## Breaking Point Analysis

**Classification:** Capacity Exhaustion

| Metric | Value |
|--------|-------|
| Virtual Users at Break | 27 |
| Requests/Second at Break | 26.0 |
| Failure Type | error_rate_breach |
| Threshold Exceeded | error_rate > 0.05 |
| Observed Value | 0.2413 |
| Threshold Value | 0.0500 |
| Detection Confidence | 85% |

### Observed Signals

- error_rate_breach at 27 VUs
- Observed: 0.2413, Threshold: 0.0500
- Severity: 4.83x threshold
- Error rate: 24.13%
- P95 latency: 706.1ms
- Throughput: 26.0 RPS

### Failure Timeline

| Time | Event | Test Type | VUs | Description |
|------|-------|-----------|-----|-------------|
| t0 | 📈 load_change | adaptive_2vus | 2 | Load increased to 2 VUs (adaptive_2vus) |
| t1 | 📈 load_change | adaptive_7vus | 7 | Load increased to 7 VUs (adaptive_7vus) |
| t2 | 📈 load_change | adaptive_12vus | 12 | Load increased to 12 VUs (adaptive_12vus) |
| t3 | 📈 load_change | adaptive_17vus | 17 | Load increased to 17 VUs (adaptive_17vus) |
| t4 | 🟠 latency_degradation | adaptive_17vus | 17 | Latency slope changed (1.7x increase) |
| t5 | 📈 load_change | adaptive_22vus | 22 | Load increased to 22 VUs (adaptive_22vus) |
| t6 | 📈 load_change | adaptive_27vus | 27 | Load increased to 27 VUs (adaptive_27vus) |
| t7 | 🔴 error_rate_breach | adaptive_27vus | 27 | Error rate crossed threshold (24.1% > 5.0%) |
| t8 | 🟠 latency_degradation | adaptive_27vus | 27 | Latency slope changed (19.0x increase) |
| t9 | 🟠 latency_degradation | adaptive_27vus | 27 | Latency exceeded threshold (706ms > 200ms) |
| t10 | 🟡 throughput_plateau | adaptive_27vus | 27 | Throughput plateaued (expected 11.4%, got -20.5%) |
| t11 | 🟡 throughput_plateau | adaptive_27vus | 27 | Throughput dropped (-20.5% change) |

## Root Cause Analysis

**Analysis Mode:** rules  

**Confidence:** 85%

### Summary

System reached resource limits causing throughput plateau at 27 VUs (26.0 RPS)

### Primary Cause

Resource saturation (CPU, memory, or connections)

### Failure Pattern

**Detected Pattern:** Latency Before Errors

Latency degraded significantly before errors appeared. Likely causes: request queuing at a bottleneck, database contention, lock contention, or garbage collection pressure. The system queued requests until timeouts caused errors.

### Contributing Factors

- Error rate exceeded threshold
- Throughput stopped scaling with load
- Latency degradation detected

### Assumptions

- Load tests accurately simulate production traffic patterns
- Telemetry data reflects actual system behavior

### Limitations

- Analysis based on rules-only (LLM unavailable)
- Low baseline confidence - limited historical data

## Recommendations

### 1. Increase concurrency limits (thread pools, connection pools)

**Priority:** 🔴 P1  
**Risk:** 🟡 MEDIUM  
**Confidence:** 82%

**Rationale:** System hit resource limits causing throughput plateau (detected at 27 VUs, 26.0 RPS via error_rate_breach)

**Expected Impact:** Higher throughput ceiling before saturation

### 2. Scale instances horizontally (add replicas)

**Priority:** 🟡 P2  
**Risk:** 🟡 MEDIUM  
**Confidence:** 78%

**Rationale:** Single instance reached capacity before load test max

**Expected Impact:** Linear throughput scaling with instance count

### 3. Profile and optimize CPU/memory hotspots

**Priority:** 🟡 P3  
**Risk:** 🟢 LOW  
**Confidence:** 74%

**Rationale:** Resource contention suggests inefficient code paths

**Expected Impact:** Better resource utilization per request

### Limitations

- Recommendations are guidance, not prescriptions
- Actual fixes require system-specific investigation

## Load Test Results

| Test Type | VUs | Duration | Requests | Error Rate | P95 Latency | Throughput |
|-----------|-----|----------|----------|------------|-------------|------------|
| adaptive_2vus | 2 | 11s | 31 | 0.00% | 36ms | 3.0 RPS |
| adaptive_7vus | 7 | 11s | 117 | 0.00% | 36ms | 11.3 RPS |
| adaptive_12vus | 12 | 11s | 183 | 0.00% | 36ms | 17.5 RPS |
| adaptive_17vus | 17 | 11s | 268 | 0.75% | 62ms | 25.6 RPS |
| adaptive_22vus | 22 | 11s | 342 | 0.88% | 37ms | 32.7 RPS |
| adaptive_27vus | 27 | 11s | 286 | 24.13% | 706ms | 26.0 RPS |
| sustained | 13 | 51s | 1588 | 0.25% | 36ms | 31.7 RPS |
| recovery | 32 | 114s | 1531 | 58.85% | 715ms | 13.5 RPS |

## Test Coverage Summary

| Metric | Value |
|--------|-------|
| Max VUs Reached | 32 |
| Max Sustained RPS | 32.7 |
| Total Requests Executed | 4,346 |
| Longest Load Duration | 114s |
| Spike Severity | Not tested |
| Recovery Observed | Yes |

## Telemetry Analysis

No telemetry data was analyzed.

## Methodology

This analysis was performed using SentinelPerf AI, an autonomous performance engineering agent.

### Analysis Pipeline

1. **Telemetry Analysis** - Inferred traffic patterns from available telemetry
2. **Test Generation** - Generated load, stress, and spike test configurations
3. **Load Execution** - Executed tests using k6 load testing framework
4. **Breaking Point Detection** - Identified first point of failure
5. **Root Cause Analysis** - Determined probable cause using observed signals only

### LLM Rules Enforced

- LLM may NOT invent metrics not present in observed data
- LLM may NOT infer causes without supporting signals
- All reasoning is step-by-step and reproducible
- Confidence scores are based on evidence strength only

---

## Appendix


**Errors Encountered:** 1

- No telemetry source configured or enabled

---

*Generated by SentinelPerf AI v0.1.0*