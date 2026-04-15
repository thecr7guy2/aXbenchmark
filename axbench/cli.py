from importlib.metadata import version
from pathlib import Path
import json
import time
from datetime import datetime, timezone

import click
from rich.console import Console
from rich.table import Table

from axbench.client import LLMClient
from axbench.env import load_dotenv
from axbench.evaluators.base import TaskResult
from axbench.evaluators.perf import PerfEvaluator
from axbench.loader import TaskLoader
from axbench.perf_tasks import load_performance_tasks
from axbench.results import BenchmarkRun, RunMetadata
from axbench.runner import Runner
from axbench.standard_loader import download_standard_tasks, load_standard_tasks
from axbench.ui.dashboard import LiveDashboard
from axbench.ui.fallback import detect_terminal_capabilities, resolve_terminal_theme
from axbench.ui.splash import show_splash, should_show_splash
from axbench.ui.state import RunUIState


console = Console()


@click.group()
@click.version_option(version("axbench"))
def cli():
    """AXBench — Comprehensive LLM benchmarking for AX-Office.ai."""
    load_dotenv()


@cli.command()
@click.option(
    "--tasks-dir",
    default="tasks",
    help="Path to tasks directory (default: ./tasks)",
)
def download(tasks_dir: str):
    """Download and cache the standard benchmark datasets."""
    with console.status("Downloading standard benchmark datasets...", spinner="dots"):
        summary = download_standard_tasks(tasks_dir=tasks_dir)

    if not summary:
        console.print("[yellow]No standard datasets were cached.[/yellow]")
        return

    table = Table(title="AXBench Standard Dataset Cache", show_header=True)
    table.add_column("Dataset", style="cyan")
    table.add_column("Tasks Cached", justify="right")
    for dataset_name, task_count in sorted(summary.items()):
        table.add_row(dataset_name, str(task_count))
    console.print(table)


@cli.command()
@click.option("--base-url", required=True, help="OpenAI-compatible endpoint URL")
@click.option("--model", required=True, help="Model name")
@click.option("--api-key", default="EMPTY", help="API key (default: EMPTY)")
@click.option("--save", default=None, help="Path to save JSON results")
@click.option(
    "--pillar",
    multiple=True,
    type=click.Choice(
        ["standard", "performance", "general_coding", "team_real_world", "all"],
        case_sensitive=False,
    ),
    default=["all"],
    help="Which pillars to run",
)
@click.option(
    "--language",
    default=None,
    type=click.Choice(["python", "cpp", "bash", "sql", "text"], case_sensitive=False),
    help="Filter by language",
)
@click.option(
    "--difficulty",
    default=None,
    type=click.Choice(["easy", "medium", "hard"], case_sensitive=False),
    help="Filter by difficulty",
)
@click.option("--task", default=None, help="Run a single task by ID")
@click.option(
    "--no-ui",
    is_flag=True,
    default=False,
    help="Disable interactive splash/startup UX and use plain terminal output",
)
@click.option(
    "--tasks-dir",
    default="tasks",
    help="Path to tasks directory (default: ./tasks)",
)
def run(
    base_url: str,
    model: str,
    api_key: str,
    save: str | None,
    pillar: tuple[str, ...],
    language: str | None,
    difficulty: str | None,
    task: str | None,
    no_ui: bool,
    tasks_dir: str,
):
    """Run the benchmark suite against a model."""
    run_started = time.monotonic()
    client = LLMClient(base_url, model, api_key)
    loader = TaskLoader(tasks_dir)
    all_tasks = _available_tasks(loader)

    if task:
        tasks = [next((loaded_task for loaded_task in all_tasks if loaded_task.get("id") == task), None)]
        if tasks[0] is None:
            raise KeyError(f"Task not found: {task!r}")
        selected_task_ids = [tasks[0]["id"]]
        skipped_task_ids = []
    else:
        pillar_set = {item.lower() for item in pillar}
        tasks = all_tasks
        if language:
            tasks = [loaded_task for loaded_task in tasks if loaded_task.get("language") == language]
        if difficulty:
            tasks = [loaded_task for loaded_task in tasks if loaded_task.get("difficulty") == difficulty]
        if "all" not in pillar_set:
            tasks = [loaded_task for loaded_task in tasks if _task_matches_pillars(loaded_task, pillar_set)]

        selected_task_ids = [loaded_task["id"] for loaded_task in tasks]
        selected_task_ids_set = set(selected_task_ids)
        skipped_task_ids = [
            loaded_task["id"] for loaded_task in all_tasks if loaded_task.get("id") not in selected_task_ids_set
        ]

    if not tasks:
        console.print("[yellow]No tasks matched the filters.[/yellow]")
        return

    ui_state = RunUIState.create(
        model=model,
        base_url=base_url,
        total_tasks=len(tasks),
        run_id=f"run-{int(time.time())}",
    )
    ui_state.set_phase("startup", "Preparing benchmark run")

    capabilities = detect_terminal_capabilities(console, no_ui=no_ui)
    theme = resolve_terminal_theme(capabilities)
    interactive_mode = capabilities.interactive
    startup_mode = "plain"
    dashboard = None
    if interactive_mode and should_show_splash(no_ui=no_ui):
        show_splash(console, theme=theme)
        ui_state.add_event("run_started", "Interactive splash completed")
        startup_mode = "interactive"
    else:
        ui_state.add_event("run_started", "Plain startup path selected")

    quality_tasks = [loaded_task for loaded_task in tasks if loaded_task.get("evaluator") != "perf"]
    perf_tasks = [loaded_task for loaded_task in tasks if loaded_task.get("evaluator") == "perf"]

    if interactive_mode:
        dashboard = LiveDashboard(
            console=console,
            state=ui_state,
            selected_task_ids=selected_task_ids,
            skipped_task_ids=skipped_task_ids,
            theme=theme,
        )
        ui_state.set_phase("benchmarking", "Dashboard online")
        dashboard.start()
        dashboard.refresh()
    else:
        console.print(
            f"[bold]AXBench[/bold] — running [cyan]{len(tasks)}[/cyan] tasks "
            f"against [green]{model}[/green]"
        )
        console.print(
            f"[dim]Run ID:[/dim] {ui_state.run_id}  "
            f"[dim]Mode:[/dim] {startup_mode}"
        )
        _print_selection(selected_task_ids, skipped_task_ids)

    runner = Runner(client)
    benchmark_run = runner.run_tasks(
        quality_tasks,
        show_progress=not interactive_mode,
        event_callback=_make_ui_event_callback(ui_state, dashboard),
    ) if quality_tasks else _empty_run(client)
    for perf_task in perf_tasks:
        if interactive_mode:
            ui_state.set_phase("performance", f"Running performance benchmark: {perf_task['id']}")
            ui_state.mark_task_started(perf_task["id"])
            if dashboard is not None:
                dashboard.refresh()
            perf_result = _run_perf_task(
                client,
                api_key,
                perf_task,
                progress_callback=lambda line: _handle_perf_progress(ui_state, dashboard, line),
            )
        else:
            console.print(f"[bold]Starting performance benchmark:[/bold] [cyan]{perf_task['id']}[/cyan]")
            with console.status(
                f"Running llama-benchy for {perf_task['id']}...",
                spinner="dots",
            ) as status:
                perf_result = _run_perf_task(
                    client,
                    api_key,
                    perf_task,
                    progress_callback=lambda line: _update_perf_status(status, perf_task["id"], line),
                )
        benchmark_run.tasks.append(perf_result)
        if interactive_mode:
            perf_metrics = perf_result.test_results[0] if perf_result.test_results else {}
            ui_state.record_performance_metrics(
                perf_task["id"],
                perf_metrics,
                error=perf_result.error,
            )
            if perf_result.error:
                ui_state.mark_task_failed(
                    perf_task["id"],
                    duration_ms=perf_result.latency_ms or None,
                    error=perf_result.error,
                )
            else:
                ui_state.mark_task_completed(perf_task["id"], duration_ms=perf_result.latency_ms)
            if dashboard is not None:
                dashboard.refresh()
        else:
            if perf_result.error:
                console.print(
                    f"[red]Performance benchmark failed:[/red] "
                    f"{perf_task['id']} — {perf_result.error}"
                )
            else:
                console.print(
                    f"[green]Completed performance benchmark:[/green] "
                    f"{perf_task['id']} "
                    f"(TTFT {perf_result.latency_ms:.0f} ms)"
                )

    benchmark_run.metadata.duration_seconds = round(time.monotonic() - run_started, 2)
    benchmark_run.selected_task_ids = selected_task_ids
    benchmark_run.skipped_task_ids = skipped_task_ids

    ui_state.set_phase("finalizing", "Benchmark finished, generating summary")
    if dashboard is not None:
        dashboard.stop()

    _print_scorecard(benchmark_run)
    _print_execution_report(benchmark_run)

    if save:
        output_path = Path(save)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        benchmark_run.save(output_path)
        console.print(f"\n[green]Results saved to {output_path}[/green]")


@cli.command()
@click.argument("result_a", type=click.Path(exists=True, path_type=Path))
@click.argument("result_b", type=click.Path(exists=True, path_type=Path))
def compare(result_a: Path, result_b: Path):
    """Compare two benchmark result files side by side."""
    run_a = BenchmarkRun.load(result_a)
    run_b = BenchmarkRun.load(result_b)

    summary_a = run_a._build_summary()
    summary_b = run_b._build_summary()

    console.rule("[bold]AXBench Model Comparison[/bold]")
    console.print(
        f"[bold]Model A:[/bold] {run_a.metadata.model} "
        f"({run_a.metadata.timestamp[:10]})"
    )
    console.print(
        f"[bold]Model B:[/bold] {run_b.metadata.model} "
        f"({run_b.metadata.timestamp[:10]})"
    )
    console.print()

    table = Table(show_header=True, title="By Pillar")
    table.add_column("Pillar")
    table.add_column(f"A: {run_a.metadata.model[:20]}", style="cyan")
    table.add_column(f"B: {run_b.metadata.model[:20]}", style="magenta")
    table.add_column("Delta", style="bold")

    all_pillars = set(summary_a["by_pillar"]) | set(summary_b["by_pillar"])
    for current_pillar in sorted(all_pillars):
        score_a = summary_a["by_pillar"].get(current_pillar, {}).get("score", 0.0)
        score_b = summary_b["by_pillar"].get(current_pillar, {}).get("score", 0.0)
        delta = score_a - score_b
        if delta > 0:
            delta_str = f"[green]+{delta * 100:.1f}%[/green]"
        elif delta < 0:
            delta_str = f"[red]{delta * 100:.1f}%[/red]"
        else:
            delta_str = "[dim]0.0%[/dim]"
        table.add_row(
            current_pillar,
            f"{score_a * 100:.1f}%",
            f"{score_b * 100:.1f}%",
            delta_str,
        )
    console.print(table)

    overall_a = summary_a["overall_quality_score"]
    overall_b = summary_b["overall_quality_score"]
    overall_delta = overall_a - overall_b
    sign = "+" if overall_delta >= 0 else ""
    console.print(
        f"\n[bold]Overall quality:[/bold] "
        f"A={overall_a * 100:.1f}%  "
        f"B={overall_b * 100:.1f}%  "
        f"delta={sign}{overall_delta * 100:.1f}%"
    )

    tasks_a = {task.task_id: task for task in run_a.tasks}
    tasks_b = {task.task_id: task for task in run_b.tasks}
    changed = []
    for task_id in set(tasks_a) & set(tasks_b):
        left_task = tasks_a[task_id]
        right_task = tasks_b[task_id]
        if left_task.passed != right_task.passed:
            changed.append((task_id, right_task.passed, left_task.passed))

    if changed:
        console.print("\n[bold]Task changes (B -> A):[/bold]")
        for task_id, previous_passed, current_passed in sorted(changed):
            arrow = "[green][+][/green]" if current_passed else "[red][-][/red]"
            status = "FAIL -> PASS" if current_passed else "PASS -> FAIL"
            console.print(f"  {arrow} {task_id}  {status}")

    wins_a = 0
    wins_b = 0
    for current_pillar in all_pillars:
        score_a = summary_a["by_pillar"].get(current_pillar, {}).get("score", 0.0)
        score_b = summary_b["by_pillar"].get(current_pillar, {}).get("score", 0.0)
        if score_a > score_b:
            wins_a += 1
        elif score_b > score_a:
            wins_b += 1

    if wins_a > wins_b:
        recommendation = run_a.metadata.model
        lead_count = wins_a
    elif wins_b > wins_a:
        recommendation = run_b.metadata.model
        lead_count = wins_b
    else:
        recommendation = f"Tie between {run_a.metadata.model} and {run_b.metadata.model}"
        lead_count = wins_a

    console.rule()
    console.print(
        f"[bold]Recommendation:[/bold] [green]{recommendation}[/green] "
        f"leads on {lead_count}/{len(all_pillars)} pillars"
    )


@cli.command("list-tasks")
@click.option("--pillar", default=None, help="Filter by pillar")
@click.option("--language", default=None, help="Filter by language")
@click.option("--difficulty", default=None, help="Filter by difficulty")
@click.option("--source", default=None, help="Filter by source (e.g. team/tom)")
@click.option("--tasks-dir", default="tasks", help="Tasks directory")
def list_tasks(
    pillar: str | None,
    language: str | None,
    difficulty: str | None,
    source: str | None,
    tasks_dir: str,
):
    """List available benchmark tasks."""
    loader = TaskLoader(tasks_dir)
    tasks = _list_available_tasks(loader)

    if pillar:
        tasks = [task for task in tasks if task.get("pillar") == pillar]
    if language:
        tasks = [task for task in tasks if task.get("language") == language]
    if difficulty:
        tasks = [task for task in tasks if task.get("difficulty") == difficulty]
    if source:
        tasks = [task for task in tasks if str(task.get("source", "")).startswith(source)]

    table = Table(title=f"AXBench Tasks ({len(tasks)} total)", show_header=True)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Evaluator")
    table.add_column("Language")
    table.add_column("Difficulty")
    table.add_column("Source")
    table.add_column("Tags")

    for task in tasks:
        table.add_row(
            task.get("id") or "",
            task.get("evaluator") or "",
            task.get("language") or "",
            task.get("difficulty") or "",
            task.get("source") or "",
            ", ".join(task.get("tags", [])),
        )
    console.print(table)


def _task_matches_pillars(task: dict, pillars: set[str]) -> bool:
    source = task.get("source", "")
    evaluator = task.get("evaluator")

    if "team_real_world" in pillars and isinstance(source, str) and source.startswith("team/"):
        return True
    if "general_coding" in pillars and evaluator in {"code_gen", "bug_fix"} and not str(source).startswith("team/"):
        return True
    if "standard" in pillars and evaluator == "standard":
        return True
    if "performance" in pillars and evaluator == "perf":
        return True
    return False


def _available_tasks(loader: TaskLoader) -> list[dict]:
    return sorted(
        loader.load()
        + load_standard_tasks(tasks_dir=loader.tasks_dir)
        + load_performance_tasks(),
        key=lambda task: (
            task.get("evaluator", ""),
            task.get("language", ""),
            task.get("id", ""),
        ),
    )


def _list_available_tasks(loader: TaskLoader) -> list[dict]:
    tasks = loader.list_tasks()
    for task in load_standard_tasks(tasks_dir=loader.tasks_dir) + load_performance_tasks():
        tasks.append(
            {
                "id": task.get("id"),
                "evaluator": task.get("evaluator"),
                "language": task.get("language"),
                "difficulty": task.get("difficulty"),
                "source": task.get("source"),
                "tags": task.get("tags", []),
                "pillar": loader._task_pillar(task),
            }
        )
    return sorted(
        tasks,
        key=lambda task: (
            task["evaluator"] or "",
            task["language"] or "",
            task["id"] or "",
        ),
    )


def _empty_run(client: LLMClient) -> BenchmarkRun:
    return BenchmarkRun(
        metadata=RunMetadata(
            model=client.model,
            base_url=client.base_url,
            timestamp=datetime.now(timezone.utc).isoformat(),
            axbench_version=version("axbench"),
            duration_seconds=0.0,
        ),
        tasks=[],
    )


def _make_ui_event_callback(ui_state: RunUIState, dashboard: LiveDashboard | None):
    def handle_event(event_name: str, payload: dict) -> None:
        task = payload["task"]
        task_id = task["id"]
        if event_name == "task_started":
            ui_state.mark_task_started(task_id)
        elif event_name == "task_completed":
            result = payload["result"]
            ui_state.mark_task_completed(task_id, duration_ms=result.latency_ms)
        elif event_name == "task_failed":
            result = payload["result"]
            ui_state.mark_task_failed(
                task_id,
                duration_ms=result.latency_ms or None,
                error=result.error,
            )
        if dashboard is not None:
            dashboard.refresh()

    return handle_event


def _run_perf_task(
    client: LLMClient,
    api_key: str,
    task: dict,
    progress_callback=None,
) -> TaskResult:
    evaluator = PerfEvaluator()
    try:
        result = evaluator.run(
            base_url=client.base_url,
            model=client.model,
            api_key=api_key,
            progress_callback=progress_callback,
        )
        metrics = {
            "pp_tokens_per_sec": result.pp_tokens_per_sec,
            "tg_tokens_per_sec": result.tg_tokens_per_sec,
            "peak_tg_tokens_per_sec": result.peak_tg_tokens_per_sec,
            "ttft_ms": result.ttft_ms,
            "selected_benchmark": result.selected_benchmark,
        }
        return TaskResult(
            task_id=task["id"],
            evaluator="perf",
            pillar="performance",
            source=task.get("source", "performance/llama-benchy"),
            language=task.get("language", "text"),
            difficulty=task.get("difficulty", "benchmark"),
            passed=True,
            score=0.0,
            raw_output=json.dumps(result.raw, indent=2),
            extracted_code="",
            test_results=[metrics],
            error=None,
            latency_ms=result.ttft_ms,
        )
    except Exception as exc:
        return TaskResult(
            task_id=task["id"],
            evaluator="perf",
            pillar="performance",
            source=task.get("source", "performance/llama-benchy"),
            language=task.get("language", "text"),
            difficulty=task.get("difficulty", "benchmark"),
            passed=False,
            score=0.0,
            raw_output="",
            extracted_code="",
            test_results=[],
            error=str(exc),
            latency_ms=0.0,
        )


def _update_perf_status(status, task_id: str, line: str) -> None:
    summary = _summarize_perf_output(line)
    if summary is None:
        return
    status.update(f"{task_id}: {summary}")


def _handle_perf_progress(ui_state: RunUIState, dashboard: LiveDashboard | None, line: str) -> None:
    summary = _summarize_perf_output(line)
    if summary is None:
        return
    ui_state.add_event("perf_output", summary, task_id="performance_llama_benchy", level="active")
    if dashboard is not None:
        dashboard.refresh()


def _summarize_perf_output(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    interesting_prefixes = (
        "Warming up",
        "Running coherence test",
        "Coherence test",
        "Measuring latency",
        "Average latency",
        "Running test:",
        "Run ",
        "[Interrupted/Failed]",
    )
    if stripped.startswith(interesting_prefixes):
        return stripped
    return None


def _print_scorecard(run: BenchmarkRun):
    summary = run._build_summary()
    table = Table(title=f"AXBench Scorecard — {run.metadata.model}", show_header=True)
    table.add_column("Category", style="bold")
    table.add_column("Tasks")
    table.add_column("Passed")
    table.add_column("Score", style="cyan")

    for pillar, stats in summary["by_pillar"].items():
        score_pct = f"{stats['score'] * 100:.1f}%"
        table.add_row(pillar, str(stats["total"]), str(stats["passed"]), score_pct)

    console.print(table)
    console.print(
        f"\n[bold]Overall quality score:[/bold] "
        f"[cyan]{summary['overall_quality_score'] * 100:.1f}%[/cyan]"
    )
    standard_sources = {
        source: stats
        for source, stats in summary["by_source"].items()
        if str(source).startswith("standard/")
    }
    if standard_sources:
        standard_table = Table(title="Standard Benchmarks", show_header=True)
        standard_table.add_column("Benchmark", style="bold")
        standard_table.add_column("Tasks")
        standard_table.add_column("Passed")
        standard_table.add_column("Score", style="cyan")
        for source, stats in sorted(standard_sources.items()):
            standard_table.add_row(
                source.removeprefix("standard/"),
                str(stats["total"]),
                str(stats["passed"]),
                f"{stats['score'] * 100:.1f}%",
            )
        console.print(standard_table)

    if summary["performance"]:
        perf = summary["performance"]
        perf_table = Table(title="Performance (llama-benchy)", show_header=True)
        perf_table.add_column("Metric", style="bold")
        perf_table.add_column("Value", style="cyan")
        perf_table.add_row("Prompt throughput", f"{perf['pp_tokens_per_sec']:.2f} tok/s")
        perf_table.add_row("Generation throughput", f"{perf['tg_tokens_per_sec']:.2f} tok/s")
        perf_table.add_row("Peak generation throughput", f"{perf['peak_tg_tokens_per_sec']:.2f} tok/s")
        perf_table.add_row("TTFT", f"{perf['ttft_ms']:.2f} ms")
        if perf["error"]:
            perf_table.add_row("Error", perf["error"])
        console.print(perf_table)

    console.print(
        f"Executed: {summary['executed_tasks']}  "
        f"Skipped: {summary['skipped_tasks']}  "
        f"Errors: {summary['errored_tasks']}"
    )
    console.print(f"Duration: {run.metadata.duration_seconds:.1f}s")


def _print_selection(selected_task_ids: list[str], skipped_task_ids: list[str]) -> None:
    console.print(
        f"Selection: [cyan]{len(selected_task_ids)}[/cyan] queued, "
        f"[yellow]{len(skipped_task_ids)}[/yellow] skipped"
    )

    queued_table = Table(title="Queued Tasks", show_header=True)
    queued_table.add_column("Task ID", style="cyan", no_wrap=True)
    for task_id in selected_task_ids:
        queued_table.add_row(task_id)
    console.print(queued_table)

    if skipped_task_ids:
        skipped_table = Table(title="Skipped Tasks", show_header=True)
        skipped_table.add_column("Task ID", style="yellow", no_wrap=True)
        for task_id in skipped_task_ids:
            skipped_table.add_row(task_id)
        console.print(skipped_table)


def _print_execution_report(run: BenchmarkRun) -> None:
    table = Table(title="Executed Task Results", show_header=True)
    table.add_column("Task ID", style="cyan", no_wrap=True)
    table.add_column("Status")
    table.add_column("Score")
    table.add_column("Latency")

    for task in run.tasks:
        if task.pillar == "performance":
            status = "[magenta]PERF[/magenta]" if not task.error else "[red]ERROR[/red]"
            score = "-"
            latency = f"{task.latency_ms:.0f} ms TTFT" if task.latency_ms else "-"
        elif task.error:
            status = "[red]ERROR[/red]"
            score = f"{task.score:.2f}"
            latency = f"{task.latency_ms:.0f} ms"
        elif task.passed:
            status = "[green]PASS[/green]"
            score = f"{task.score:.2f}"
            latency = f"{task.latency_ms:.0f} ms"
        else:
            status = "[yellow]FAIL[/yellow]"
            score = f"{task.score:.2f}"
            latency = f"{task.latency_ms:.0f} ms"
        table.add_row(
            task.task_id,
            status,
            score,
            latency,
        )

    console.print(table)
