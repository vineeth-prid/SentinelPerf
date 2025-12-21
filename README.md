# SentinelPerf AI
# Autonomous Performance Engineering Agent

## Overview

SentinelPerf is a CLI-first autonomous performance engineering agent that:
- Infers traffic behavior from telemetry
- Generates load, stress, and spike tests automatically
- Executes adaptive load testing
- Identifies the first breaking point
- Explains root cause with confidence scores
- Recommends fixes based on observed signals only

## Installation

```bash
# Install from source
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## Prerequisites

- Python 3.9+
- k6 load testing tool: https://k6.io/docs/getting-started/installation/
- (Optional) Ollama for local LLM: https://ollama.ai/

## Quick Start

```bash
# 1. Create configuration file
cp sentinelperf.yaml.example sentinelperf.yaml

# 2. Edit configuration for your environment
vim sentinelperf.yaml

# 3. Run analysis
sentinelperf run --env=local
```

## Usage

### Basic Commands

```bash
# Run performance analysis
sentinelperf run --env=<environment>

# Validate configuration
sentinelperf validate --config=./sentinelperf.yaml

# Show version
sentinelperf --version
```

### Options

```bash
sentinelperf run --env=staging \\n  --config=./sentinelperf.yaml \\n  --output-dir=./reports \\n  --llm-mode=ollama \\n  --verbose
```

| Option | Description | Default |
|--------|-------------|---------|
| `--env`, `-e` | Target environment (required) | - |
| `--config`, `-c` | Config file path | `./sentinelperf.yaml` |
| `--output-dir`, `-o` | Report output directory | `./sentinelperf-reports` |
| `--llm-mode` | LLM mode: ollama, rules, mock | `ollama` |
| `--verbose`, `-v` | Enable verbose output | `false` |

## Configuration

See `sentinelperf.yaml.example` for full configuration options.

### Required Inputs

1. **Base URL** - Target application URL
2. **Authentication** - Bearer token or custom header
3. **Telemetry Source** - OpenTelemetry, access logs, or Prometheus

### Telemetry Priority

Telemetry sources are checked in order: OTEL → Logs → Prometheus

## Output

SentinelPerf produces three outputs:

1. **Console Summary** (max 5 lines) - Quick status check
2. **Markdown Report** - Authoritative detailed report
3. **JSON Summary** - CI/CD integration friendly

### CI/CD Integration

```bash
# Run in CI pipeline
sentinelperf run --env=staging --llm-mode=rules

# Check exit code
if [ $? -eq 0 ]; then
  echo "Performance check passed"
else
  echo "Performance issues detected"
fi
```

## LLM Rules

SentinelPerf enforces strict rules for LLM-based analysis:

- LLM may **NOT** invent metrics
- LLM may **NOT** infer causes without observed signals
- LLM **MUST** explain reasoning step-by-step
- LLM **MUST** assign confidence scores based on signal strength

The LLM is an explanation and reasoning layer, not a decision oracle.

## Architecture

```
┌──────────────────────────────────────────┐
│         CLI (sentinelperf run)            │
└────────────────────┬─────────────────────┘
                       │
┌────────────────────▼─────────────────────┐
│     LangGraph Agent Orchestration          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │Telemetry │→│ Test Gen │→│ Load Exec│ │
│  └──────────┘ └──────────┘ └──────────┘ │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │ Report   │←│Root Cause│←│ Breaking │ │
│  └──────────┘ └──────────┘ └──────────┘ │
└──────────────────────────────────────────┘
          │             │             │
    ┌─────▼─────┐ ┌────▼─────┐ ┌────▼─────┐
    │    k6     │ │   OTEL   │ │ Ollama   │
    └───────────┘ └──────────┘ └──────────┘
```

## License

MIT
