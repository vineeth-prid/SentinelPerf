# SentinelPerf AI

CLI tool that runs load tests against your API, finds the breaking point, and tells you why it failed.

Uses k6 for load generation. No dashboards, no cloud, just results.

## Install

### Docker (recommended)
```bash
docker pull sentinelperf/sentinelperf:latest
```

### Binary
```bash
# Download from releases
curl -L https://github.com/sentinelperf/sentinelperf/releases/latest/download/sentinelperf-linux-amd64 -o sentinelperf
chmod +x sentinelperf
```

## Usage

1. Create config file `sentinelperf.yaml`:
```yaml
version: "1.0"
environments:
  local:
    target:
      base_url: "http://localhost:8080"
      endpoints:
        - "/api/users"
        - "/api/orders"
    load:
      initial_vus: 5
      max_vus: 50
      error_rate_threshold: 0.05
      p95_latency_threshold_ms: 500
```

2. Run:
```bash
# Docker
docker run -v $(pwd):/work sentinelperf/sentinelperf run --env=local

# Binary
./sentinelperf run --env=local
```

## Output

```
⚠ SentinelPerf analysis complete: http://localhost:8080
  Breaking point: 30 VUs @ 45.2 RPS (error_rate_breach)
  Root cause: Error propagation and cascade effect [●●●●○]
  Report: sentinelperf-reports/report.md
```

Reports generated:
- `sentinelperf-reports/report.md` - Full analysis
- `sentinelperf-reports/summary.json` - CI/CD friendly

## Requirements

- Target API must be reachable
- Docker or k6 installed locally
