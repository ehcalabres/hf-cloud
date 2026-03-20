
"""CLI utility functions."""

from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def create_spinner(description: str) -> Progress:
    """Create a spinner progress indicator.

    Args:
        description: Text to display with spinner

    Returns:
        Progress context manager
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    )


def print_error(message: str, details: Optional[str] = None) -> None:
    """Print an error message.

    Args:
        message: Main error message
        details: Optional additional details
    """
    console.print(f"[bold red]Error:[/bold red] {message}")
    if details:
        console.print(f"[dim]{details}[/dim]")


def print_success(message: str) -> None:
    """Print a success message.

    Args:
        message: Success message
    """
    console.print(f"[bold green]✓[/bold green] {message}")


def print_warning(message: str) -> None:
    """Print a warning message.

    Args:
        message: Warning message
    """
    console.print(f"[bold yellow]⚠[/bold yellow] {message}")


def print_info(message: str) -> None:
    """Print an info message.

    Args:
        message: Info message
    """
    console.print(f"[bold blue]ℹ[/bold blue] {message}")


def format_duration(seconds: float) -> str:
    """Format a duration in human-readable form.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def truncate_string(s: str, max_length: int, suffix: str = "...") -> str:
    """Truncate a string to a maximum length.

    Args:
        s: String to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated string
    """
    if len(s) <= max_length:
        return s
    return s[: max_length - len(suffix)] + suffix
