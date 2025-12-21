"""Report generation module for SentinelPerf"""

from sentinelperf.reports.console import print_summary, format_console_output
from sentinelperf.reports.markdown import MarkdownReporter
from sentinelperf.reports.json_report import JSONReporter

__all__ = [
    "print_summary",
    "format_console_output",
    "MarkdownReporter",
    "JSONReporter",
]
