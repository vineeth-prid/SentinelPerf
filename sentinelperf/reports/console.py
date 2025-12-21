"""Console output for SentinelPerf"""

from typing import Optional
from sentinelperf.core.state import ExecutionResult, AgentPhase


def format_console_output(result: ExecutionResult) -> str:
    """
    Format execution result for console output.
    
    Returns max 5 lines as per requirements.
    """
    lines = []
    state = result.state
    
    # Line 1: Status
    if result.success:
        lines.append(f"\033[92m✓\033[0m SentinelPerf analysis complete: {state.target_url}")
    else:
        lines.append(f"\033[91m✗\033[0m SentinelPerf analysis failed: {state.target_url}")
    
    # Line 2: Breaking point (if detected)
    if state.breaking_point and state.breaking_point.vus_at_break > 0:
        bp = state.breaking_point
        lines.append(
            f"  Breaking point: {bp.vus_at_break} VUs @ {bp.rps_at_break:.1f} RPS "
            f"({bp.failure_type})"
        )
    
    # Line 3: Root cause (if analyzed)
    if state.root_cause and state.root_cause.primary_cause:
        rc = state.root_cause
        confidence_bar = "●" * int(rc.confidence * 5) + "○" * (5 - int(rc.confidence * 5))
        lines.append(
            f"  Root cause: {rc.primary_cause[:60]}... [{confidence_bar}]"
            if len(rc.primary_cause) > 60
            else f"  Root cause: {rc.primary_cause} [{confidence_bar}]"
        )
    
    # Line 4: Errors (if any)
    if state.errors:
        lines.append(f"  \033[93mWarnings: {len(state.errors)}\033[0m")
    
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
            AgentPhase.REPORT_GENERATION: "📄",
            AgentPhase.COMPLETE: "✅",
            AgentPhase.ERROR: "❌",
        }
        icon = phase_icons.get(phase, "•")
        print(f"{icon} [{phase.value}] {message}")
