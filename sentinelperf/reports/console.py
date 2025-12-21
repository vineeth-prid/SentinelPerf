"""Console output for SentinelPerf"""

from typing import Optional
from sentinelperf.core.state import ExecutionResult, AgentPhase


# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def format_console_output(result: ExecutionResult) -> str:
    """
    Format execution result for console output.
    
    Returns max 5 lines as per requirements:
    1. Status line (pass/fail with target)
    2. Breaking point info (if detected)
    3. Root cause with confidence
    4. Warnings count (if any)
    5. Report location
    """
    lines = []
    state = result.state
    
    # Line 1: Status with clear pass/fail indication
    if result.success:
        if state.breaking_point and state.breaking_point.vus_at_break > 0:
            lines.append(f"{YELLOW}⚠{RESET} SentinelPerf analysis complete: {state.target_url}")
        else:
            lines.append(f"{GREEN}✓{RESET} SentinelPerf analysis complete: {state.target_url}")
    else:
        lines.append(f"{RED}✗{RESET} SentinelPerf analysis failed: {state.target_url}")
    
    # Line 2: Breaking point (if detected)
    if state.breaking_point and state.breaking_point.vus_at_break > 0:
        bp = state.breaking_point
        lines.append(
            f"  Breaking point: {bp.vus_at_break} VUs @ {bp.rps_at_break:.1f} RPS "
            f"({bp.failure_type})"
        )
    
    # Line 3: Root cause with confidence bar
    if state.root_cause and state.root_cause.primary_cause:
        rc = state.root_cause
        confidence_bar = "●" * int(rc.confidence * 5) + "○" * (5 - int(rc.confidence * 5))
        cause = rc.primary_cause
        if len(cause) > 55:
            cause = cause[:55] + "..."
        lines.append(f"  Root cause: {cause} [{confidence_bar}]")
    
    # Line 4: Warnings (if any)
    if state.errors:
        lines.append(f"  {YELLOW}Warnings: {len(state.errors)}{RESET}")
    
    # Line 5: Report location
    if result.markdown_report_path:
        lines.append(f"  Report: {result.markdown_report_path}")
    
    return "\n".join(lines[:5])


def print_summary(result: ExecutionResult) -> None:
    """
    Print execution summary to console.
    
    Max 5 lines as per requirements.
    """
    output = format_console_output(result)
    print(output)


def print_progress(phase: AgentPhase, message: str, verbose: bool = False) -> None:
    """Print progress message if verbose mode enabled"""
    if verbose:
        phase_icons = {
            AgentPhase.INIT: "⚙",
            AgentPhase.TELEMETRY_ANALYSIS: "📊",
            AgentPhase.TEST_GENERATION: "📝",
            AgentPhase.LOAD_EXECUTION: "🚀",
            AgentPhase.RESULTS_COLLECTION: "📈",
            AgentPhase.BREAKING_POINT_DETECTION: "🎯",
            AgentPhase.ROOT_CAUSE_ANALYSIS: "🔍",
            AgentPhase.RECOMMENDATIONS: "💡",
            AgentPhase.REPORT_GENERATION: "📄",
            AgentPhase.COMPLETE: "✅",
            AgentPhase.ERROR: "❌",
        }
        icon = phase_icons.get(phase, "•")
        print(f"{icon} [{phase.value}] {message}")
