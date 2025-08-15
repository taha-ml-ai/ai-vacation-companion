from __future__ import annotations
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from ..core.data_loader import load_json
from ..core.recommender import Preference, normalize_tags, recommend

console = Console()

def ask(prompt: str, default: str | None = None) -> str:
    if default:
        return console.input(f"[bold]{prompt}[/bold] [dim]({default})[/dim]: ").strip() or default
    return console.input(f"[bold]{prompt}[/bold]: ").strip()

def run_cli():
    console.print(Panel.fit("🌴 [bold cyan]AI Vacation Companion[/bold cyan] — find a perfect trip in seconds"))
    budget = ask("Budget [low/medium/high]", "medium").lower()
    climate = ask("Preferred climate [warm/cold/mild]", "warm").lower()
    acts = ask("Activities you enjoy (comma-separated)", "beach, culture")
    duration = ask("Duration in days", "6")
    month = ask("Travel month (optional)", "")

    try:
        duration_days = int(duration)
    except Exception:
        duration_days = None

    pref = Preference(
        budget=budget,
        climate=climate,
        activities=normalize_tags(acts),
        duration_days=duration_days,
        month=month or None,
    )

    destinations = load_json("destinations.json")
    packages = load_json("packages.json")

    results = recommend(pref, destinations, packages, top_k=5)

    if not results:
        console.print("[yellow]No matches found. Try loosening your filters.[/yellow]")
        return

    table = Table(title="Top Suggestions", show_lines=True)
    table.add_column("Destination")
    table.add_column("Package")
    table.add_column("Budget", justify="center")
    table.add_column("Nights", justify="right")
    table.add_column("Price", justify="right")
    table.add_column("Why it fits", justify="left")

    for r in results:
        dest = r["destination"]
        pkg = r["package"]
        why = f"climate:{dest.get('climate')} • tags:{pkg.get('activities') or dest.get('activities')} • score:{r['score']}"
        table.add_row(
            f"{dest.get('name')} ({dest.get('country')})",
            pkg.get("name",""),
            pkg.get("budget",""),
            str(pkg.get("nights","")),
            f"{pkg.get('price','')}" if pkg.get('price') is not None else "-",
            why,
        )

    console.print(table)

if __name__ == "__main__":
    run_cli()
