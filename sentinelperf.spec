# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['sentinelperf/cli.py'],
    pathex=[],
    binaries=[],
    datas=[('sentinelperf', 'sentinelperf')],
    hiddenimports=['sentinelperf.cli', 'sentinelperf.core.agent', 'sentinelperf.config.loader', 'sentinelperf.config.schema', 'sentinelperf.telemetry.otel', 'sentinelperf.telemetry.baseline', 'sentinelperf.load.generator', 'sentinelperf.load.k6_executor', 'sentinelperf.analysis.breaking_point', 'sentinelperf.analysis.root_cause', 'sentinelperf.analysis.recommendations', 'sentinelperf.reports.console', 'sentinelperf.reports.markdown', 'sentinelperf.reports.json_report', 'sentinelperf.llm.client'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='sentinelperf',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
