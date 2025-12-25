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


def format_console_output(result: ExecutionResult, execution_id: str = "") -> str:
    """
    Format execution result for console output.
    
    Returns max 5 lines as per requirements:
    1. Status line (pass/fail with target + execution ID)
    2. Breaking point info (if detected)
    3. Root cause with confidence
    4. Warnings count (if any)
    5. Report location
    """
    lines = []
    state = result.state
    
    # Use execution_id from param or state
    exec_id = execution_id or state.execution_id or ""
    exec_id_short = exec_id[:8] if exec_id else "unknown"
    
    # Line 1: Status with clear pass/fail indication + execution ID
    if result.success:
        if state.breaking_point and state.breaking_point.vus_at_break > 0:
            lines.append(f"{YELLOW}⚠{RESET} SentinelPerf [{exec_id_short}] complete: {state.target_url}")
        else:
            lines.append(f"{GREEN}✓{RESET} SentinelPerf [{exec_id_short}] complete: {state.target_url}")
    else:
        lines.append(f"{RED}✗{RESET} SentinelPerf [{exec_id_short}] failed: {state.target_url}")
    
    # Line 2: Breaking point (if detected) with ACTUAL max VUs
    actual_max_vus = state.achieved_max_vus
    if actual_max_vus == 0 and state.load_results:
        actual_max_vus = max((r.vus for r in state.load_results), default=0)
    
    if state.breaking_point and state.breaking_point.vus_at_break > 0:
        bp = state.breaking_point
        lines.append(
            f"  Breaking point: {bp.vus_at_break} VUs @ {bp.rps_at_break:.1f} RPS "
            f"({bp.failure_type})"
        )
    elif actual_max_vus > 0:
        lines.append(f"  Max VUs executed: {actual_max_vus} (no breaking point detected)")
    
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


def print_summary(result: ExecutionResult, execution_id: str = "") -> None:
    """
    Print execution summary to console.
    
    Max 5 lines as per requirements.
    """
    output = format_console_output(result, execution_id)
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
