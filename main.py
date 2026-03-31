"""CLI entry point for PRBench-Eval."""

import asyncio

import typer

app = typer.Typer(
    help="PRBench-Eval: Benchmark AI agents on scientific paper reproduction"
)


@app.command()
def green(
    host: str = typer.Option("localhost", help="Host to bind"),
    port: int = typer.Option(9001, help="Port"),
    container_id: str = typer.Option(None, help="Docker container ID"),
    agent_type: str = typer.Option("claude", help="Agent CLI for grading: 'claude', 'codex', or 'opencode'"),
):
    """Start the green agent (assessment manager with model-judge grading)."""
    from src.green_agent import start_green_agent
    start_green_agent(host=host, port=port, docker_container_id=container_id, agent_type=agent_type)


@app.command()
def white(
    host: str = typer.Option("localhost", help="Host to bind"),
    port: int = typer.Option(9002, help="Port"),
    workspace: str = typer.Option(None, help="Workspace directory"),
    container_id: str = typer.Option(None, help="Docker container ID"),
    agent_type: str = typer.Option("claude", help="Agent CLI to use: 'claude', 'codex', or 'opencode'"),
):
    """Start the white agent (Claude Code, OpenAI Codex, or OpenCode executor inside Docker)."""
    from src.white_agent import start_white_agent
    start_white_agent(host=host, port=port, workspace_base=workspace,
                      docker_container_id=container_id, agent_type=agent_type)


@app.command()
def launch(
    task_id: str = typer.Option("task_creutz_1980", help="Task ID to evaluate"),
    green_port: int = typer.Option(9001, help="Port for green agent (0=auto)"),
    white_port: int = typer.Option(9002, help="Port for white agent (0=auto)"),
    code_only: bool = typer.Option(False, "--code-only", help="Code-only evaluation (no simulation execution)"),
    agent_type: str = typer.Option("claude", help="Default agent CLI for both white/green: 'claude', 'codex', or 'opencode'"),
    white_agent_type: str = typer.Option(None, "--white-agent-type", help="Override agent CLI for white agent (task solving)"),
    green_agent_type: str = typer.Option(None, "--green-agent-type", help="Override agent CLI for green agent (grading)"),
    no_archive: bool = typer.Option(False, "--no-archive", help="Skip archiving workspace to results/"),
    results_subdir: str = typer.Option(None, "--results-subdir", help="Override results subdirectory name (default: 'full' or 'code_only')"),
):
    """Launch complete evaluation: Docker setup + green/white agents + grading.

    Ports default to 0 (auto-allocate free ports), enabling parallel runs.

    Use --white-agent-type / --green-agent-type to run different CLIs for task
    solving vs grading (e.g. --white-agent-type opencode --green-agent-type claude).
    """
    from src.launcher import launch_evaluation
    asyncio.run(launch_evaluation(
        task_id=task_id,
        green_port=green_port,
        white_port=white_port,
        code_only=code_only,
        agent_type=agent_type,
        white_agent_type=white_agent_type,
        green_agent_type=green_agent_type,
        archive=not no_archive,
        results_subdir=results_subdir,
    ))


@app.command()
def batch(
    task_ids: str = typer.Option(..., help="Comma-separated task IDs (or 'all' for every task)"),
    code_only: bool = typer.Option(False, "--code-only", help="Code-only evaluation"),
    agent_type: str = typer.Option("claude", help="Default agent CLI: 'claude', 'codex', or 'opencode'"),
    white_agent_type: str = typer.Option(None, "--white-agent-type", help="Override agent CLI for white agent (task solving)"),
    green_agent_type: str = typer.Option(None, "--green-agent-type", help="Override agent CLI for green agent (grading)"),
    max_concurrent: int = typer.Option(2, "--max-concurrent", "-j", help="Max parallel evaluations"),
    no_archive: bool = typer.Option(False, "--no-archive", help="Skip archiving workspace to results/"),
    results_subdir: str = typer.Option(None, "--results-subdir", help="Override results subdirectory name (default: 'full' or 'code_only')"),
):
    """Run multiple evaluations in parallel with bounded concurrency.

    Each evaluation gets its own ports and Docker container automatically.

    Examples:
        python main.py batch --task-ids task_creutz_1980,task_lin_2020 -j 2
        python main.py batch --task-ids all --code-only -j 4
        python main.py batch --task-ids all --white-agent-type opencode --green-agent-type claude
    """
    import os
    from src.launcher import batch_evaluate, DATA_DIR

    if task_ids.strip().lower() == "all":
        ids = sorted([
            d for d in os.listdir(DATA_DIR)
            if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith("task_")
        ])
    else:
        ids = [t.strip() for t in task_ids.split(",") if t.strip()]

    if not ids:
        typer.echo("No task IDs found.")
        raise typer.Exit(1)

    typer.echo(f"Batch evaluation: {len(ids)} tasks, max_concurrent={max_concurrent}")
    for tid in ids:
        typer.echo(f"  - {tid}")

    asyncio.run(batch_evaluate(
        task_ids=ids,
        code_only=code_only,
        agent_type=agent_type,
        white_agent_type=white_agent_type,
        green_agent_type=green_agent_type,
        max_concurrent=max_concurrent,
        archive=not no_archive,
        results_subdir=results_subdir,
    ))


@app.command()
def grade(
    task_id: str = typer.Option("task_creutz_1980", help="Task ID to grade"),
    workspace: str = typer.Option(None, help="Workspace directory"),
    container_id: str = typer.Option(None, help="Docker container ID for grading"),
    code_only: bool = typer.Option(False, "--code-only", help="Code-only grading (no CSV comparison)"),
    agent_type: str = typer.Option("claude", help="Agent CLI for grading: 'claude', 'codex', or 'opencode'"),
):
    """Run model-judge grading on workspace results."""
    import json
    import os
    import time
    from src.green_agent.agent import (
        build_grading_prompt, load_task_config,
        PRBenchGreenAgentExecutor,
    )

    project_root = os.path.dirname(os.path.abspath(__file__))
    task_dir = os.path.join(project_root, "data", "tasks", task_id)
    workspace_path = workspace if workspace else os.path.join(task_dir, "workspace")

    if not os.path.isdir(task_dir):
        typer.echo(f"Task directory not found: {task_dir}")
        raise typer.Exit(1)
    if not os.path.isdir(workspace_path):
        typer.echo(f"Workspace not found: {workspace_path}")
        raise typer.Exit(1)

    log_dir = os.path.join(workspace_path, "eval_logs")
    os.makedirs(log_dir, exist_ok=True)

    task_config = load_task_config(task_dir)

    # Read white agent trace if available
    trace_path = os.path.join(log_dir, "white_agent_trace.log")
    white_trace = ""
    if os.path.exists(trace_path):
        with open(trace_path) as f:
            white_trace = f.read()

    prompt = build_grading_prompt(task_config, task_dir, workspace_path, white_trace,
                                  code_only=code_only)
    typer.echo(f"Grading prompt length: {len(prompt)} chars")

    # Run model-judge grading
    executor = PRBenchGreenAgentExecutor(docker_container_id=container_id, agent_type=agent_type)
    grading_result = executor._run_grading(prompt, container_id, log_dir, agent_type=agent_type)

    # Export traces from container if container-id provided
    if container_id:
        import subprocess
        _trace_map = {
            "codex": ("/home/agent/.codex", "_codex_traces"),
            "opencode": ("/home/agent/.local/share/opencode", "_opencode_traces"),
            "claude": ("/home/agent/.claude", "_claude_traces"),
        }
        container_path, folder_name = _trace_map.get(agent_type, _trace_map["claude"])
        trace_dest = os.path.join(log_dir, folder_name)
        os.makedirs(trace_dest, exist_ok=True)
        subprocess.run(
            ["docker", "cp", f"{container_id}:{container_path}", trace_dest],
            capture_output=True, timeout=60,
        )

    # Save report
    report = {
        "task_id": task_id,
        "paper": task_config["paper"],
        "grading": grading_result,
        "grading_method": "model_judge",
        "grading_agent_type": agent_type,
        "code_only": code_only,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    report_path = os.path.join(log_dir, "eval_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    typer.echo(json.dumps(report, indent=2, default=str))
    typer.echo(f"\nReport saved to {report_path}")


@app.command()
def cleanup(
    green_port: int = typer.Option(9001, help="Port for green agent"),
    white_port: int = typer.Option(9002, help="Port for white agent"),
):
    """Clean up any running agents and Docker containers from previous runs."""
    import os
    import signal
    import subprocess
    import docker

    typer.echo("Cleaning up PRBench resources...")

    # Kill processes on agent ports (robust: try lsof first, then fuser)
    for port in [green_port, white_port]:
        killed = False
        # Method 1: lsof
        try:
            result = subprocess.run(
                ["lsof", "-ti", f"tcp:{port}"],
                capture_output=True, text=True, timeout=5,
            )
            pids = result.stdout.strip().split()
            for pid in pids:
                if pid:
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                        typer.echo(f"  Killed PID {pid} on port {port}")
                        killed = True
                    except (ProcessLookupError, ValueError):
                        pass
        except FileNotFoundError:
            pass
        except Exception:
            pass

        # Method 2: fuser (fallback)
        if not killed:
            try:
                subprocess.run(["fuser", "-k", f"{port}/tcp"],
                               capture_output=True, timeout=5)
                typer.echo(f"  Killed process on port {port} (fuser)")
            except Exception:
                pass

    # Stop and remove python:3.11-slim containers
    try:
        client = docker.from_env()
        containers = client.containers.list(
            all=True, filters={"ancestor": "python:3.11-slim"}
        )
        for c in containers:
            typer.echo(f"  Removing container: {c.short_id}")
            c.remove(force=True)
    except Exception as e:
        typer.echo(f"  Docker cleanup: {e}")

    typer.echo("Cleanup complete.")


@app.command()
def migrate():
    """Migrate legacy workspaces into results/ directory structure."""
    from migrate_legacy import migrate as do_migrate
    do_migrate()


if __name__ == "__main__":
    app()
