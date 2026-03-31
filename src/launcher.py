"""Launcher module - orchestrates the full PRBench evaluation pipeline.

Flow:
1. Load task config from YAML
2. Create Docker environment (Python + Node.js + agent CLI + credentials)
3. Start green agent (A2A on port), passing Docker container_id
4. Start white agent (A2A on port), passing Docker container_id
5. Send task config to green agent
6. Green agent orchestrates: instruction -> white agent -> poll -> grade
7. Collect final report
8. Post-eval cleanup (ordered):
   a. Kill this eval's green/white agent processes
   b. Remove this eval's Docker container
   c. Archive workspace to results/
"""

import asyncio
import json
import logging
import multiprocessing
import os
import shutil
import signal
import subprocess
import time

import yaml

from src.green_agent.agent import start_green_agent
from src.white_agent.agent import start_white_agent
from src.my_util import my_a2a
from src.my_util.docker_manager import DockerEnvironment
from src.my_util.agent_env import resolve_env, build_docker_exec_env_flags
from src.my_util.ports import find_free_port_pair

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "tasks")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def _kill_process(proc: multiprocessing.Process | None, label: str) -> None:
    """Kill a specific child process by its PID. Does not touch any other process."""
    if proc is None:
        return
    pid = proc.pid
    if pid is None:
        return
    # Try graceful SIGTERM first
    try:
        if proc.is_alive():
            os.kill(pid, signal.SIGTERM)
            proc.join(timeout=3)
    except (ProcessLookupError, AssertionError, OSError):
        pass
    # Force kill if still alive
    try:
        if proc.is_alive():
            os.kill(pid, signal.SIGKILL)
            proc.join(timeout=2)
    except (ProcessLookupError, AssertionError, OSError):
        pass


def _remove_container(container_id: str | None) -> None:
    """Remove exactly one Docker container by its ID.

    chmod -R 777 /workspace first so host user can delete files after
    archiving, even when Docker userns-remap remaps container UIDs.
    """
    if not container_id:
        return
    try:
        subprocess.run(
            ["docker", "exec", container_id, "chmod", "-R", "777", "/workspace"],
            capture_output=True, timeout=30,
        )
    except Exception:
        pass
    try:
        subprocess.run(
            ["docker", "rm", "-f", container_id],
            capture_output=True, timeout=30,
        )
    except Exception:
        pass


def _archive_workspace(task_id: str, workspace_dir: str, code_only: bool,
                       results_subdir: str | None = None) -> str | None:
    """Move workspace to results/ directory and copy eval_report.json.

    Returns the destination directory path, or None on failure.
    If results_subdir is given (e.g. "full_codex"), use it instead of the
    default "full" / "code_only".
    """
    mode = results_subdir or ("code_only" if code_only else "full")
    mode_dir = os.path.join(RESULTS_DIR, mode)
    dest_dir = os.path.join(mode_dir, task_id)
    os.makedirs(dest_dir, exist_ok=True)

    ws_name = "workspace_code" if code_only else "workspace_full"
    ws_dest = os.path.join(dest_dir, ws_name)

    if os.path.exists(ws_dest):
        date_suffix = time.strftime("%Y%m%d")
        ws_dest = os.path.join(dest_dir, f"{ws_name}_{date_suffix}")
        if os.path.exists(ws_dest):
            ws_dest = os.path.join(dest_dir, f"{ws_name}_{time.strftime('%Y%m%d_%H%M%S')}")

    if os.path.isdir(workspace_dir):
        try:
            shutil.move(workspace_dir, ws_dest)
        except Exception as e:
            print(f"  [ARCHIVE] Failed to move workspace: {e}", flush=True)
            return None

        report_src = os.path.join(ws_dest, "eval_logs", "eval_report.json")
        if os.path.isfile(report_src):
            shutil.copy2(report_src, os.path.join(dest_dir, "eval_report.json"))

    return dest_dir


def _archive_trace(task_id: str, code_only: bool,
                   results_subdir: str | None = None) -> None:
    """Copy the .out trace file (if any) to the results directory."""
    mode = results_subdir or ("code_only" if code_only else "full")
    trace_dir = os.path.join(PROJECT_ROOT, f"trace_{mode}")
    out_file = os.path.join(trace_dir, f"{task_id}.out")
    if os.path.isfile(out_file):
        dest_dir = os.path.join(RESULTS_DIR, mode, task_id)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(out_file, os.path.join(dest_dir, f"{task_id}.out"))


_IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".gif", ".bmp", ".svg", ".webp"}


def _copy_paper_images(task_dir: str, workspace_dir: str) -> None:
    """Copy image files from the task directory into workspace/_paper_images/.

    This allows the white agent to view paper figures without having access
    to the full task directory (which contains ground truth data and
    reference reproduction code).

    Scans the task root and any immediate subdirectories whose names start
    with 'image' (e.g. images/, images1/) for image files.
    """
    dest = os.path.join(workspace_dir, "_paper_images")
    os.makedirs(dest, exist_ok=True)

    def _copy_images_from(src_dir: str, dest_dir: str) -> int:
        count = 0
        if not os.path.isdir(src_dir):
            return 0
        for fname in os.listdir(src_dir):
            if os.path.splitext(fname)[1].lower() in _IMAGE_EXTENSIONS:
                dst_path = os.path.join(dest_dir, fname)
                if os.path.exists(dst_path):
                    os.remove(dst_path)
                shutil.copy(os.path.join(src_dir, fname), dst_path)
                count += 1
        return count

    total = 0

    # Copy images from task root (paper2/3 style: _page_*.jpeg in root)
    total += _copy_images_from(task_dir, dest)

    # Copy images from image subdirectories (paper1 style: images1/)
    for entry in os.listdir(task_dir):
        entry_path = os.path.join(task_dir, entry)
        if os.path.isdir(entry_path) and entry.lower().startswith("image"):
            sub_dest = os.path.join(dest, entry)
            os.makedirs(sub_dest, exist_ok=True)
            total += _copy_images_from(entry_path, sub_dest)

    if total > 0:
        logger.info(f"Copied {total} paper image(s) to workspace/_paper_images/")


def _copy_paper_markdown(task_config: dict, task_dir: str, workspace_dir: str) -> None:
    """Copy the paper markdown file into workspace under its original filename.

    The paper content is already embedded inside _instruction.md, but copying it
    as a standalone file lets the agent reference it directly by name
    (e.g. `cat Bude2021.md`) without having to parse _instruction.md.
    """
    paper_file = task_config.get("paper", {}).get("paper_file", "")
    if not paper_file:
        return
    src = os.path.join(task_dir, paper_file)
    if os.path.isfile(src):
        dst = os.path.join(workspace_dir, os.path.basename(paper_file))
        if os.path.exists(dst):
            os.remove(dst)
        shutil.copy2(src, dst)
        logger.info(f"Copied paper markdown to workspace/{os.path.basename(paper_file)}")
    else:
        logger.warning(f"Paper markdown file not found: {paper_file}")


def _copy_input_files(task_config: dict, task_dir: str, workspace_dir: str) -> None:
    """Copy input files specified in task.yaml to workspace root.

    These are files the white agent needs to run the simulation (e.g., data tables)
    but are NOT part of the ground truth output.
    """
    input_files = task_config.get("input_files", [])
    if not input_files:
        return

    count = 0
    for rel_path in input_files:
        src = os.path.join(task_dir, rel_path)
        if os.path.isfile(src):
            dst = os.path.join(workspace_dir, os.path.basename(rel_path))
            if os.path.exists(dst):
                os.remove(dst)
            shutil.copy2(src, dst)
            count += 1
        else:
            logger.warning(f"Input file not found: {rel_path}")

    if count > 0:
        logger.info(f"Copied {count} input file(s) to workspace/")


def _export_traces_for_type(
    docker_env: DockerEnvironment, agent_type: str, eval_logs_dir: str,
) -> None:
    """Export agent CLI traces from Docker container for a given agent type."""
    if agent_type == "codex":
        trace_dest = os.path.join(eval_logs_dir, "_codex_traces")
        docker_env.export_codex_traces(trace_dest)
    elif agent_type == "opencode":
        trace_dest = os.path.join(eval_logs_dir, "_opencode_traces")
        docker_env.export_opencode_traces(trace_dest)
    else:  # claude
        trace_dest = os.path.join(eval_logs_dir, "_claude_traces")
        docker_env.export_claude_traces(trace_dest)


def setup_docker_environment(
    task_config: dict,
    task_dir: str,
    workspace_dir: str,
    white_agent_type: str = "claude",
    green_agent_type: str = "claude",
) -> DockerEnvironment:
    """Create and configure the Docker environment with the chosen agent CLI(s) installed.

    When white_agent_type and green_agent_type differ, both CLIs are installed
    and all relevant credentials are injected.
    """
    docker_cfg = task_config.get("docker", {})
    needed_types = {white_agent_type, green_agent_type}

    # Collect credentials for all needed agent types via centralized resolver
    env_vars: dict[str, str] = {}
    for t in needed_types:
        env_vars.update(resolve_env(t))

    docker_env = DockerEnvironment(
        image=docker_cfg.get("image", "python:3.11-slim"),
        task_dir=task_dir,
        workspace_dir=workspace_dir,
        memory_limit=docker_cfg.get("memory_limit", "4g"),
        timeout=docker_cfg.get("timeout", 10800),
        env_vars=env_vars,
        pip_packages=docker_cfg.get("pip_install", ["numpy", "scipy", "matplotlib"]),
    )

    logger.info("Creating Docker environment...")
    docker_env.start()

    if not docker_env.check_health():
        raise RuntimeError("Docker health check failed")
    logger.info("Docker: Python environment ready.")

    # Install all needed CLIs
    for agent_type in sorted(needed_types):
        if agent_type == "codex":
            logger.info("Docker: Installing OpenAI Codex CLI (Node.js + npm)...")
            success = docker_env.install_codex()
            if not success:
                raise RuntimeError("Failed to install OpenAI Codex CLI in Docker")
            if not docker_env.check_codex_health():
                raise RuntimeError("OpenAI Codex CLI verification failed in Docker")
            logger.info("Docker: OpenAI Codex CLI ready.")
        elif agent_type == "opencode":
            logger.info("Docker: Installing OpenCode CLI (Node.js + npm + git)...")
            success = docker_env.install_opencode()
            if not success:
                raise RuntimeError("Failed to install OpenCode CLI in Docker")
            if not docker_env.check_opencode_health():
                raise RuntimeError("OpenCode CLI verification failed in Docker")
            logger.info("Docker: OpenCode CLI ready.")
        else:  # claude
            logger.info("Docker: Installing Claude Code CLI (Node.js + npm)...")
            success = docker_env.install_claude_code()
            if not success:
                raise RuntimeError("Failed to install Claude Code CLI in Docker")
            if not docker_env.check_claude_health():
                raise RuntimeError("Claude Code CLI verification failed in Docker")
            logger.info("Docker: Claude Code CLI ready.")

    return docker_env


async def launch_evaluation(
    task_id: str = "task_creutz_1980",
    green_host: str = "localhost",
    green_port: int = 0,
    white_host: str = "localhost",
    white_port: int = 0,
    code_only: bool = False,
    agent_type: str = "claude",
    white_agent_type: str | None = None,
    green_agent_type: str | None = None,
    archive: bool = True,
    results_subdir: str | None = None,
):
    """Launch the complete evaluation pipeline for a single paper task.

    Ports default to 0 which means auto-allocate free ports.
    Set archive=True (default) to move workspace to results/ after evaluation.
    results_subdir overrides the default "full"/"code_only" subdirectory name
    under results/ (e.g. "full_codex").

    agent_type is the default for both white and green agents.
    white_agent_type / green_agent_type override agent_type for their respective roles.
    """
    effective_white_type = white_agent_type or agent_type
    effective_green_type = green_agent_type or agent_type
    if green_port == 0 or white_port == 0:
        auto_green, auto_white = find_free_port_pair()
        if green_port == 0:
            green_port = auto_green
        if white_port == 0:
            white_port = auto_white

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [LAUNCHER] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("launcher.log"),
        ],
    )

    task_dir = os.path.join(DATA_DIR, task_id)
    if not os.path.isdir(task_dir):
        logger.error(f"Task directory not found: {task_dir}")
        return

    task_yaml = os.path.join(task_dir, "task.yaml")
    if not os.path.exists(task_yaml):
        logger.error(f"task.yaml not found in {task_dir}")
        return

    with open(task_yaml, "r") as f:
        task_config = yaml.safe_load(f)

    logger.info("=" * 60)
    logger.info(f"PRBench Evaluation: {task_id}")
    logger.info(f"Paper: {task_config['paper']['title']}")
    logger.info(f"Author: {task_config['paper']['author']}")
    logger.info(f"Ports: green={green_port}, white={white_port}")
    logger.info("=" * 60)

    workspace_dir = os.path.join(task_dir, "workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    os.makedirs(os.path.join(workspace_dir, "eval_logs"), exist_ok=True)
    os.makedirs(os.path.join(workspace_dir, "reproduction"), exist_ok=True)
    os.makedirs(os.path.join(workspace_dir, "data"), exist_ok=True)

    # Copy paper images into workspace so the white agent can view figures
    # WITHOUT having access to the full task directory (which contains
    # ground truth data and reference code).
    _copy_paper_images(task_dir, workspace_dir)

    # Copy the paper markdown file into workspace under its original filename
    # so the agent can open it directly (e.g. `cat Bude2021.md`) rather than
    # having to parse _instruction.md.
    _copy_paper_markdown(task_config, task_dir, workspace_dir)

    # Copy any input files specified in task.yaml (e.g., data tables needed
    # for simulation but not part of ground truth output)
    _copy_input_files(task_config, task_dir, workspace_dir)

    docker_env = None
    p_green = None
    p_white = None
    container_id = None

    try:
        # Step 1: Create Docker environment
        _type_labels = {"claude": "Claude Code", "codex": "OpenAI Codex", "opencode": "OpenCode"}
        white_label = _type_labels.get(effective_white_type, effective_white_type)
        green_label = _type_labels.get(effective_green_type, effective_green_type)
        if effective_white_type == effective_green_type:
            logger.info(f"[1/5] Setting up Docker environment with {white_label}...")
        else:
            logger.info(f"[1/5] Setting up Docker (white={white_label}, green={green_label})...")
        docker_env = setup_docker_environment(task_config, task_dir, workspace_dir,
                                              white_agent_type=effective_white_type,
                                              green_agent_type=effective_green_type)
        container_id = docker_env.container_id
        assert container_id is not None, "Docker container ID is None after start"
        logger.info(f"[1/5] Docker ready. Container: {container_id[:12]}")

        # Step 2: Start green agent
        logger.info(f"[2/5] Launching green agent on port {green_port}...")
        green_url = f"http://{green_host}:{green_port}"
        p_green = multiprocessing.Process(
            target=start_green_agent,
            args=("prbench_green_agent", green_host, green_port, container_id, effective_green_type),
            daemon=True,
        )
        p_green.start()
        ready = await my_a2a.wait_agent_ready(green_url, timeout=30)
        if not ready:
            logger.error("Green agent failed to start!")
            return
        logger.info("[2/5] Green agent ready.")

        # Step 3: Start white agent
        logger.info(f"[3/5] Launching white agent on port {white_port}...")
        white_url = f"http://{white_host}:{white_port}"
        p_white = multiprocessing.Process(
            target=start_white_agent,
            args=("prbench_white_agent", white_host, white_port,
                  workspace_dir, container_id, effective_white_type),
            daemon=True,
        )
        p_white.start()
        ready = await my_a2a.wait_agent_ready(white_url, timeout=30)
        if not ready:
            logger.error("White agent failed to start!")
            return
        logger.info("[3/5] White agent ready.")

        # Step 4: Send task to green agent
        logger.info("[4/5] Sending task to green agent...")
        incoming_config = {
            "task_id": task_id,
            "data_dir": task_dir,
            "docker_container_id": container_id,
            "code_only": code_only,
        }
        task_text = (
            f"Your task is to evaluate the paper reproduction agent located at:\n"
            f"<white_agent_url>\n"
            f"http://{white_host}:{white_port}/\n"
            f"</white_agent_url>\n"
            f"You should use the following task configuration:\n"
            f"<task_config>\n"
            f"{json.dumps(incoming_config, indent=2)}\n"
            f"</task_config>\n"
        )

        logger.info("Sending to green agent (may take hours)...")
        start_time = time.time()

        try:
            response = await my_a2a.send_message(green_url, task_text)
            elapsed = time.time() - start_time
            logger.info(f"Green agent responded after {elapsed:.0f}s")
            logger.info(str(response))
        except Exception as e:
            elapsed = time.time() - start_time
            logger.exception(f"Error after {elapsed:.0f}s: {e}")

        # Step 5: Report
        report_path = os.path.join(workspace_dir, "eval_logs", "eval_report.json")
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                report = json.load(f)
            logger.info("=" * 60)
            logger.info("FINAL REPORT")
            logger.info("=" * 60)
            logger.info(json.dumps(report, indent=2, default=str))
        else:
            logger.warning(f"No report at {report_path}")

    finally:
        # ============================================================
        # Post-evaluation cleanup (ordered)
        # ============================================================

        # 0. Export traces from docker BEFORE killing anything
        if docker_env and docker_env.container_id and workspace_dir:
            eval_logs_dir = os.path.join(workspace_dir, "eval_logs")
            os.makedirs(eval_logs_dir, exist_ok=True)
            # Export traces for white agent type
            _export_traces_for_type(docker_env, effective_white_type, eval_logs_dir)
            # If green agent uses a different CLI, also export its traces
            if effective_green_type != effective_white_type:
                _export_traces_for_type(docker_env, effective_green_type, eval_logs_dir)
            try:
                logs = docker_env.get_logs()
                with open(os.path.join(eval_logs_dir, "docker_container.log"), "w") as f:
                    f.write(logs)
            except Exception as e:
                print(f"  [WARN] Failed to save container logs: {e}", flush=True)

        # 1. Kill THIS eval's green agent process (by PID)
        print(f"[POST-EVAL] Step 1: Killing green agent (PID={p_green.pid if p_green else None})...", flush=True)
        _kill_process(p_green, "green")
        print(f"[POST-EVAL] Step 1: Killing white agent (PID={p_white.pid if p_white else None})...", flush=True)
        _kill_process(p_white, "white")
        print(f"[POST-EVAL] Step 1: Agent processes killed.", flush=True)

        # 2. Remove THIS eval's Docker container (by container ID)
        print(f"[POST-EVAL] Step 2: Removing Docker container {container_id[:12] if container_id else 'N/A'}...", flush=True)
        _remove_container(container_id)
        print(f"[POST-EVAL] Step 2: Docker container removed.", flush=True)

        # 3. Archive workspace to results/
        if archive and os.path.isdir(workspace_dir):
            print(f"[POST-EVAL] Step 3: Archiving workspace...", flush=True)
            try:
                dest = _archive_workspace(task_id, workspace_dir, code_only, results_subdir)
                _archive_trace(task_id, code_only, results_subdir)
                print(f"[POST-EVAL] Step 3: Archived to {dest}", flush=True)
            except Exception as e:
                print(f"[POST-EVAL] Step 3: Archive error: {e}", flush=True)
        else:
            print(f"[POST-EVAL] Step 3: Skipped (archive={archive}, workspace exists={os.path.isdir(workspace_dir) if workspace_dir else False})", flush=True)

    print(f"[POST-EVAL] Evaluation of {task_id} complete.", flush=True)


async def batch_evaluate(
    task_ids: list[str],
    code_only: bool = False,
    agent_type: str = "claude",
    white_agent_type: str | None = None,
    green_agent_type: str | None = None,
    max_concurrent: int = 2,
    archive: bool = True,
    results_subdir: str | None = None,
):
    """Run multiple evaluations in parallel with bounded concurrency.

    Each evaluation gets its own dynamically allocated ports and Docker container.
    After each evaluation completes, its agents and container are cleaned up.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _run_one(task_id: str):
        async with semaphore:
            print(f"[BATCH] Starting evaluation: {task_id}", flush=True)
            try:
                await launch_evaluation(
                    task_id=task_id,
                    green_port=0,  # auto-allocate
                    white_port=0,  # auto-allocate
                    code_only=code_only,
                    agent_type=agent_type,
                    white_agent_type=white_agent_type,
                    green_agent_type=green_agent_type,
                    archive=archive,
                    results_subdir=results_subdir,
                )
                print(f"[BATCH] Completed: {task_id}", flush=True)
            except Exception as e:
                print(f"[BATCH] Failed: {task_id}: {e}", flush=True)

    tasks = [asyncio.create_task(_run_one(tid)) for tid in task_ids]
    await asyncio.gather(*tasks, return_exceptions=True)
    print(f"[BATCH] All {len(task_ids)} evaluations finished.", flush=True)
