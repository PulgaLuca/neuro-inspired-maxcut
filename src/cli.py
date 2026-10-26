from pathlib import Path
import json
import typer
from config.loader import load_config
from config.validation import validate_config
from experiments.runner import plan_runs, execute_plan
from storage.results import create_experiment_dir

app = typer.Typer(name="neuro-inspired-maxcut", help="Benchmark Framework for Max-Cut problem", no_args_is_help=True)

def _read_validated(path: Path) -> dict:
    config = load_config(path)
    errors = validate_config(config)
    if errors:
        for error in errors:
            typer.echo(f"ERROR: {error}", err=True)
        raise typer.Exit(code=2)
    return config


@app.command()
def validate(config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True)):
    """Read and validate YAML file without running it."""
    _read_validated(config)
    typer.echo(f"Configurazione valida: {config}")


@app.command()
def plan(config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True)):
    """Shows the plan without running any YAML file."""
    cfg = _read_validated(config)
    runs = plan_runs(cfg)
    typer.echo(f"Planned runs: {len(runs)}")
    for i, item in enumerate(runs, 1):
        graph_label = (
            f"graph={item['instance']}"
            if item.get("instance") is not None
            else f"graph={item['graph']['name']} n={item['size']}"
        )
        suffix = "" if item["supported"] else f" [NON SUPPORTATO: {item['compatibility']}]"
        typer.echo(f"{i:04d}. {graph_label} "
                   f"algorithm={item['algorithm']['name']} backend={item['backend']['name']} "
                   f"repetition={item['repetition']}{suffix}")


@app.command()
def run(config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True)):
    """Run the given and custom configurated YAML file in configs."""
    cfg = _read_validated(config)
    plan_items = plan_runs(cfg)
    if not plan_items:
        typer.echo("No run planned.", err=True)
        raise typer.Exit(code=2)
    output_dir = create_experiment_dir(cfg, config)
    typer.echo(f"Output: {output_dir}")
    typer.echo(f"Planned runs: {len(plan_items)}")
    counts = execute_plan(cfg, config, output_dir, plan_items)
    (output_dir/"status.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    typer.echo(f"Completed: {counts['completed']} | timeout: {counts['time_limit']} | Failed: {counts['failed']}")
    typer.echo(f"Results: {output_dir}")


if __name__ == "__main__":
    app()
