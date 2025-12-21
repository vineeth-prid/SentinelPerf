#!/bin/bash
# Build SentinelPerf binary using PyInstaller
# Requires: pip install pyinstaller

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building SentinelPerf binary..."

# Install PyInstaller if not present
pip install pyinstaller --quiet

# Build single binary
pyinstaller \
    --onefile \
    --name sentinelperf \
    --add-data "sentinelperf:sentinelperf" \
    --hidden-import=sentinelperf.cli \
    --hidden-import=sentinelperf.core.agent \
    --hidden-import=sentinelperf.config.loader \
    --hidden-import=sentinelperf.config.schema \
    --hidden-import=sentinelperf.telemetry.otel \
    --hidden-import=sentinelperf.telemetry.baseline \
    --hidden-import=sentinelperf.load.generator \
    --hidden-import=sentinelperf.load.k6_executor \
    --hidden-import=sentinelperf.analysis.breaking_point \
    --hidden-import=sentinelperf.analysis.root_cause \
    --hidden-import=sentinelperf.analysis.recommendations \
    --hidden-import=sentinelperf.reports.console \
    --hidden-import=sentinelperf.reports.markdown \
    --hidden-import=sentinelperf.reports.json_report \
    --hidden-import=sentinelperf.llm.client \
    --clean \
    sentinelperf/cli.py

echo ""
echo "Build complete: dist/sentinelperf"
echo ""
echo "Test with: ./dist/sentinelperf --help"
