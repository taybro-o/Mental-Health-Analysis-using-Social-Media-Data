"""
Interactive Command-Line Interface for Contextual Emotion Distribution Vectorizer (CEDV).

Modes:
  1. Direct Text Analysis:
     python cli.py --mode direct --text "I feel so exhausted and disconnected from everyone around me."
  2. Live Subreddit Fetch:
     python cli.py --mode live --subreddit anxiety
  3. Interactive Prompt Mode:
     python cli.py --mode interactive
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.model import CEDVInferenceEngine
from src.preprocess import CLASS_LABELS, clean_text
from src.scraper import CEDVScraper

console = Console()

MODEL_DEFAULT_PATH = "models/saved_cedv/cedv_model.keras"


def render_distribution_bar(value: float, max_width: int = 25, is_zscore: bool = False) -> str:
    """Render a visual ASCII bar representation for probability or z-score."""
    if is_zscore:
        # Z-scores typically range between -2.0 and +3.0
        # Normalize to 0..1 for bar display
        norm_val = max(0.0, min(1.0, (value + 2.0) / 5.0))
        filled = int(norm_val * max_width)
        return "█" * filled + "░" * (max_width - filled)
    else:
        # Simplex probability in [0, 1]
        filled = int(max(0.0, min(1.0, value)) * max_width)
        return "█" * filled + "░" * (max_width - filled)


def get_color_for_label(label: str) -> str:
    """Distinctive color palette for the 8 affective states."""
    colors = {
        "depression": "cyan",
        "anxiety": "yellow",
        "stress_burnout": "magenta",
        "adhd": "blue",
        "autism": "bright_green",
        "loneliness": "bright_blue",
        "anger": "red",
        "positive_neutral": "green",
    }
    return colors.get(label, "white")


def display_analysis(
    analysis: dict,
    mode_name: str,
    norm_type: str = "softmax",
    metadata: dict = None,
) -> None:
    """Render comprehensive Rich UI visualization of the 8D affective vector."""
    console.print()
    
    # Header panel
    header_text = f"[bold green]CEDV Emotion Distribution Analysis[/bold green] ({norm_type.upper()})"
    if metadata:
        meta_str = " | ".join(f"[bold]{k}:[/bold] {v}" for k, v in metadata.items())
        header_text += f"\n[dim]{meta_str}[/dim]"
    
    console.print(Panel(f"[italic]\"{analysis['cleaned_text']}\"[/italic]", title=header_text, expand=False))

    # Raw vector display
    if norm_type == "zscore":
        raw_vec = [round(float(v), 4) for v in analysis["zscore_vector"]]
        vec_title = "8D Standardized Z-Score Vector [N(0, 1)]"
    else:
        raw_vec = [round(float(v), 4) for v in analysis["simplex_vector"]]
        vec_title = "8D Normalized Simplex Vector [Sum = 1.0]"

    console.print(f"[bold dim]{vec_title}:[/bold dim]\n[bold cyan]{raw_vec}[/bold cyan]\n")

    # Affective breakdown table
    table = Table(title="Affective State Distribution Breakdown", show_header=True, header_style="bold white")
    table.add_column("Idx", justify="center", style="dim", width=4)
    table.add_column("Affective State", style="bold", width=18)
    table.add_column("Simplex Prob", justify="right", width=14)
    table.add_column("Z-Score", justify="right", width=10)
    table.add_column("Distribution Intensity", width=28)
    table.add_column("Status / Nuance", justify="center", width=18)

    primary = analysis["primary_emotion"]
    is_z = norm_type == "zscore"

    for item in analysis["breakdown"]:
        label = item["label"]
        idx = item["class_id"]
        prob = item["probability"]
        zsc = item["zscore"]
        color = get_color_for_label(label)

        # Bar graph visualization
        bar_val = zsc if is_z else prob
        bar_str = render_distribution_bar(bar_val, max_width=24, is_zscore=is_z)
        colored_bar = f"[{color}]{bar_str}[/{color}]"

        # Nuance status
        if label == primary:
            status = "[bold green]★ PRIMARY[/bold green]"
        elif zsc >= 0.5:
            status = f"[{color}]▲ Active Nuance[/{color}]"
        elif zsc <= -0.5:
            status = "[dim]▼ Subdued[/dim]"
        else:
            status = "[dim]— Neutral[/dim]"

        prob_str = f"{prob * 100:6.2f}%"
        zsc_str = f"{zsc:+5.2f}σ"

        table.add_row(
            str(idx),
            f"[{color}]{label}[/{color}]",
            prob_str,
            zsc_str,
            colored_bar,
            status,
        )

    console.print(table)

    # Nuance summary footer
    sec_nuances = [f"[bold]{n['label']}[/bold] ({n['zscore']:+.2f}σ)" for n in analysis["secondary_nuances"]]
    if sec_nuances:
        nuance_str = ", ".join(sec_nuances)
        console.print(f"\n[bold]Detected Co-occurring Nuances:[/bold] {nuance_str}")
    else:
        console.print("\n[dim]No strong secondary nuances detected above baseline.[/dim]")
    console.print()


def handle_direct_mode(
    engine: CEDVInferenceEngine,
    text: str,
    norm: str,
    temp: float,
) -> None:
    """Analyze directly provided text string."""
    analysis = engine.analyze_text(text, temperature=temp)
    display_analysis(analysis, mode_name="Direct Analysis", norm_type=norm)


def handle_live_mode(
    engine: CEDVInferenceEngine,
    subreddit: str,
    norm: str,
    temp: float,
) -> None:
    """Fetch live post from Reddit / Arctic Shift and analyze."""
    console.print(f"[dim]Connecting to Arctic Shift / Reddit for live r/{subreddit} post...[/dim]")
    scraper = CEDVScraper()
    posts = scraper.fetch_arctic_shift(subreddit=subreddit, limit=5)
    
    if not posts:
        # Try RSS
        posts = scraper.fetch_reddit_rss(subreddit=subreddit, limit=5)

    if not posts:
        console.print(f"[bold red]Error:[/bold red] Could not retrieve live posts from r/{subreddit}.")
        return

    post = posts[0]
    meta = {
        "Subreddit": f"r/{subreddit}",
        "Post ID": post.get("id", "N/A"),
        "Timestamp": post.get("timestamp", "N/A"),
    }
    
    analysis = engine.analyze_text(post["text"], temperature=temp)
    display_analysis(analysis, mode_name="Live Subreddit Analysis", norm_type=norm, metadata=meta)


def handle_interactive_mode(
    engine: CEDVInferenceEngine,
    norm: str,
    temp: float,
) -> None:
    """Continuous terminal prompt loop for testing sentences."""
    console.print(Panel(
        "[bold green]CEDV Interactive Session[/bold green]\n"
        "Type any paragraph or conversational text to compute continuous emotion distribution vectors.\n"
        "Type [bold red]'exit'[/bold red] or [bold red]'quit'[/bold red] to terminate.",
        title="Interactive Mode",
    ))
    while True:
        try:
            user_input = console.input("\n[bold cyan]Input Text > [/bold cyan]").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", ":q"]:
                console.print("[dim]Exiting interactive session. Goodbye![/dim]")
                break
            analysis = engine.analyze_text(user_input, temperature=temp)
            display_analysis(analysis, mode_name="Interactive Query", norm_type=norm)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session terminated.[/dim]")
            break


def main():
    parser = argparse.ArgumentParser(
        description="CEDV: Contextual Emotion Distribution Vectorizer CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["direct", "live", "interactive"],
        default="interactive",
        help="Execution mode (direct, live, or interactive)",
    )
    parser.add_argument("--text", type=str, help="Text to analyze (required for --mode direct)")
    parser.add_argument("--subreddit", type=str, default="anxiety", help="Subreddit to pull live post from")
    parser.add_argument("--model", type=str, default=MODEL_DEFAULT_PATH, help="Path to saved .keras model")
    parser.add_argument("--norm", choices=["softmax", "zscore"], default="softmax", help="Distribution display format")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax distribution temperature tau")

    args = parser.parse_args()

    if not os.path.exists(args.model):
        console.print(f"[bold red]Model checkpoint not found at:[/bold red] {args.model}")
        console.print("[yellow]Please run model training first:[/yellow] python -m src.train")
        sys.exit(1)

    console.print(f"[dim]Loading CEDV model from {args.model}...[/dim]")
    engine = CEDVInferenceEngine.load_from_path(args.model)
    console.print("[bold green]Model loaded successfully.[/bold green]")

    if args.mode == "direct":
        if not args.text:
            console.print("[bold red]Error:[/bold red] --text is required when running in direct mode.")
            sys.exit(1)
        handle_direct_mode(engine, args.text, args.norm, args.temperature)

    elif args.mode == "live":
        handle_live_mode(engine, args.subreddit, args.norm, args.temperature)

    elif args.mode == "interactive":
        handle_interactive_mode(engine, args.norm, args.temperature)


if __name__ == "__main__":
    main()
