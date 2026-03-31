"""Green agent implementation - Claude Code model-judge evaluator.

Architecture:
- Fully generic: loads ALL task-specific content from task.yaml
- Docker environment created by the launcher; container_id passed in
- Uses Claude Code inside Docker for model-judge grading
- Supports polling: sends instruction, then polls white agent every 30s
- Works with any paper task, not just Creutz 1980
"""

import asyncio
import json
import logging
import os
import shutil
import time
import tomllib

import uvicorn
import yaml
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, SendMessageSuccessResponse, Message, Task
from a2a.utils import new_agent_text_message, get_text_parts

from src.my_util import parse_tags, my_a2a
from src.my_util.docker_manager import DockerEnvironment

logger = logging.getLogger(__name__)

POLL_INTERVAL = 30  # seconds between status checks


def load_agent_card_toml(agent_name: str) -> dict:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, f"{agent_name}.toml"), "rb") as f:
        return tomllib.load(f)


def load_task_config(task_dir: str) -> dict:
    """Load task.yaml from a task directory."""
    yaml_path = os.path.join(task_dir, "task.yaml")
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def read_file_safe(path: str, max_chars: int = 50000) -> str:
    """Read a file, returning empty string if it doesn't exist."""
    if not os.path.exists(path):
        return ""
    with open(path, "r", errors="replace") as f:
        content = f.read(max_chars)
    if len(content) >= max_chars:
        content += "\n... [truncated]"
    return content


_SKIP_DIRS = {".git", "__pycache__", "node_modules"}


def collect_workspace_files(workspace_dir: str) -> list[str]:
    """List all files in workspace relative to workspace root.

    Skips internal files (prefixed with _), .git trees, and bulky trace
    sub-directories that would overwhelm the listing.
    """
    files = []
    if not os.path.isdir(workspace_dir):
        return files
    for root, dirs, fnames in os.walk(workspace_dir):
        # Prune directories we never want to descend into
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        # Inside trace export dirs, only keep top-level logs/db, skip snapshot objects
        rel_root = os.path.relpath(root, workspace_dir)
        if "/snapshot/" in rel_root or rel_root.endswith("/snapshot"):
            dirs.clear()
            continue
        for fname in fnames:
            if not fname.startswith("_"):  # skip internal files
                rel = os.path.relpath(os.path.join(root, fname), workspace_dir)
                files.append(rel)
    return sorted(files)


def _copy_ground_truth_to_workspace(
    task_config: dict, task_dir: str, workspace_dir: str,
    code_only: bool = False,
) -> None:
    """Copy ground truth files into workspace/_ground_truth/ for the grading agent.

    Called AFTER the white agent finishes and BEFORE the grading agent starts.
    The Docker container does not mount /task, so this is the only way for the
    grading agent (running inside Docker) to access reference materials.

    Copies:
    - metadata.md                      → _ground_truth/metadata.md
    - data/*.csv  (full mode only)     → _ground_truth/data/
    - <ground_truth_code_dir>/*  (all) → _ground_truth/<dir_name>/

    In code_only mode, data/ is NOT copied (agent did not produce data, no
    CSV comparison needed). Only metadata + reference code are provided so
    the grading agent can evaluate code correctness.
    """
    gt_root = os.path.join(workspace_dir, "_ground_truth")
    os.makedirs(gt_root, exist_ok=True)

    # 1. metadata.md — always copied
    meta_file = task_config.get("metadata_file", "metadata.md")
    meta_src = os.path.join(task_dir, meta_file)
    if os.path.isfile(meta_src):
        shutil.copy2(meta_src, os.path.join(gt_root, os.path.basename(meta_file)))

    # 2. data/ — full mode only
    # If ground_truth_copy_all is set, copy ALL files; otherwise only CSV
    if not code_only:
        gt_data_name = task_config.get("ground_truth_data_dir", "data")
        gt_data_src = os.path.join(task_dir, gt_data_name)
        gt_data_dst = os.path.join(gt_root, "data")
        copy_all = task_config.get("ground_truth_copy_all", False)
        if os.path.isdir(gt_data_src):
            os.makedirs(gt_data_dst, exist_ok=True)
            for fname in os.listdir(gt_data_src):
                src_path = os.path.join(gt_data_src, fname)
                if os.path.isfile(src_path):
                    # Copy if copy_all=True OR file ends with .csv
                    if copy_all or fname.endswith(".csv"):
                        shutil.copy2(src_path, os.path.join(gt_data_dst, fname))

    # 3. Reference code — always copied (all file types)
    code_dir_name = task_config.get("ground_truth_code_dir", "reproduction")
    code_src = os.path.join(task_dir, code_dir_name)
    if os.path.isdir(code_src):
        code_dst = os.path.join(gt_root, code_dir_name)
        if os.path.exists(code_dst):
            shutil.rmtree(code_dst)
        shutil.copytree(
            code_src, code_dst,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

    mode_label = "code-only: metadata + code" if code_only else "full: metadata + data(csv) + code"
    logger.info(f"Copied ground truth to workspace/_ground_truth/ ({mode_label})")


def build_white_agent_instruction(task_config: dict, task_dir: str,
                                   code_only: bool = False) -> str:
    """Build a GENERIC instruction message for the white agent.

    Paper-specific content comes entirely from files referenced in task.yaml.
    The prompt structure is the same for ANY paper.

    If code_only=True, appends the code-only instruction suffix and uses
    the code-only expected outputs (no data files).
    """
    instruction_path = os.path.join(task_dir, task_config["instruction_file"])
    instruction_content = read_file_safe(instruction_path, max_chars=100000)

    # In code-only mode, append the suffix that tells agent not to run simulations
    if code_only and "code_only" in task_config:
        suffix = task_config["code_only"].get("instruction_suffix", "")
        if suffix:
            instruction_content += "\n\n" + suffix

    paper_path = os.path.join(task_dir, task_config["paper"]["paper_file"])
    paper_content = read_file_safe(paper_path, max_chars=200000)

    # Use code-only expected outputs if in code-only mode
    if code_only and "code_only" in task_config:
        expected = task_config["code_only"].get("expected_outputs", {})
    else:
        expected = task_config.get("expected_outputs", {})
    all_expected = []
    for _category, files in expected.items():
        for f in (files or []):
            all_expected.append(f)

    docker_cfg = task_config.get("docker", {})
    pip_packages = docker_cfg.get("pip_install", [])

    if code_only:
        task_steps = """1. Analyze the paper's methodology and write your analysis
2. Implement all required code from scratch
3. Do NOT run simulations or generate data files — only write code"""
        notes = """- All paths are relative to /workspace
- Create directories as needed (mkdir -p)
- You must implement the simulation from scratch (see banned libraries in instruction)
- Do NOT run simulation scripts — focus on writing correct code only"""
    else:
        task_steps = """1. Analyze the paper's methodology and write your analysis
2. Implement all required code from scratch
3. Run your code to generate the expected output data files"""
        notes = """- All paths are relative to /workspace
- Create directories as needed (mkdir -p)
- You must implement the simulation from scratch (see banned libraries in instruction)
- Run all scripts to generate output data before finishing
- Export numerical data with 8 decimal places

## CRITICAL: Anti-Cheating Requirements

For tasks that require simulation (Monte Carlo, DMRG, FDTD, SCF, trajectory propagation, etc.):

1. **DO NOT hardcode, fabricate, or approximate output data** using analytic formulas,
   lookup tables, or values transcribed from the paper text.
2. **DO NOT write separate "model" functions** that generate fake data with invented
   parameters (e.g., `rate=0.42`, `A0=5.0`).
3. **ALL numerical results MUST come from actually running your implemented simulation code.**
4. Your figure/data generation scripts MUST call the actual simulation functions you wrote,
   not bypass them with shortcuts.
5. If the task requires iterative computation (convergence loops, Monte Carlo sampling,
   time-stepping), you MUST implement and run those iterations.

Violations will result in a failing grade for data_accuracy."""

    paper_filename = os.path.basename(task_config["paper"]["paper_file"])

    return f"""You are an AI agent tasked with reproducing results from a scientific paper.

You are working inside a Docker container. Your working directory is /workspace.
Available Python packages: {', '.join(pip_packages)}

The full paper is available as /workspace/{paper_filename} — you can read it directly with `cat /workspace/{paper_filename}`.

## Your Task

Read the instruction and the full paper below. Then:
{task_steps}

## Instruction

{instruction_content}

## Full Paper Content

{paper_content}

## Expected Output Files

The following files must exist in /workspace when you are done:
{chr(10).join(f'- {f}' for f in all_expected)}

## Important Notes

{notes}

Begin working now. Start with analysis, then code{', then data generation.' if not code_only else '.'}
"""


def build_grading_prompt(
    task_config: dict,
    task_dir: str,
    workspace_dir: str,
    white_agent_trace: str,
    code_only: bool = False,
) -> str:
    """Build a grading prompt for Claude Code model-judge.

    Ground truth files (metadata, reference CSVs, reference code) have been
    copied to /workspace/_ground_truth/ by _copy_ground_truth_to_workspace().
    The grading agent is instructed to read them directly from disk rather
    than receiving everything inline — this keeps the prompt compact and lets
    the judge inspect files at full fidelity.

    If code_only=True, uses code-only rubric and focuses on code review
    instead of CSV data comparison.
    """
    # Select rubric based on mode
    if code_only and "code_only" in task_config:
        grading_cfg = task_config["code_only"].get("grading", {})
        expected = task_config["code_only"].get("expected_outputs", {})
    else:
        grading_cfg = task_config.get("grading", {})
        expected = task_config.get("expected_outputs", {})

    dimensions = grading_cfg.get("dimensions", [])
    rubric_text = ""
    for dim in dimensions:
        rubric_text += f"\n### {dim['name']} (weight: {dim['weight']})\n{dim['description']}\n"

    workspace_files = collect_workspace_files(workspace_dir)

    # Build list of expected output paths for the judge to check
    expected_files = []
    for _category, files in expected.items():
        expected_files.extend(files)

    # List what ground truth files are available
    gt_root = os.path.join(workspace_dir, "_ground_truth")
    gt_files = []
    if os.path.isdir(gt_root):
        for root, _, fnames in os.walk(gt_root):
            for fname in fnames:
                rel = os.path.relpath(os.path.join(root, fname), workspace_dir)
                gt_files.append(rel)

    json_template = (
        "```json\n{{\n  \"scores\": {{\n"
        + chr(10).join(
            f'    "{dim["name"]}": {{"score": 0.0, "justification": "..."}},'
            for dim in dimensions
        )
        + '\n  }},\n  "overall_score": 0.0,\n  "summary": "Brief overall assessment"\n}}\n```'
    )

    code_dir_name = task_config.get("ground_truth_code_dir", "reproduction")

    if code_only:
        return f"""You are an expert evaluator for scientific paper reproduction benchmarks.

## Task
Evaluate the AI agent's code written to reproduce results from the paper:
"{task_config['paper']['title']}" by {task_config['paper']['author']}

## Evaluation Mode: CODE-ONLY
The agent was asked to write code only (no simulation execution).
You must evaluate the code by **carefully reading every file** and checking for correctness.

## Grading Rubric
{rubric_text}

## Ground Truth Reference Files
The following reference files are available for you to read:
{chr(10).join(f'- /workspace/{f}' for f in sorted(gt_files))}

**You MUST read these files** to understand the correct methodology, formulas,
and reference implementation before grading the agent's work.

Key files:
- `/workspace/_ground_truth/metadata.md` — ground truth formulas & methodology
- `/workspace/_ground_truth/{code_dir_name}/` — reference implementation code

## Agent's Expected Output Files
{chr(10).join(f'- /workspace/{f}' for f in expected_files)}

## Agent's Workspace Files
All files found: {json.dumps(workspace_files, indent=2)}

## Your Evaluation Steps

1. **Read** `/workspace/_ground_truth/metadata.md` for ground truth formulas AND Banned Libraries
2. **Read** every file in `/workspace/_ground_truth/{code_dir_name}/` as reference code
3. **Read** every agent code file listed above under Agent's Expected Output Files
4. **CHECK FOR BANNED LIBRARIES**: Scan ALL agent code files (imports, pip installs, subprocess calls) for any banned libraries listed in metadata.md.
   - If ANY banned library is used: **code_correctness score MUST be ≤ 0.2**
   - Common violations: importing banned packages, pip installing banned packages in code, calling banned CLI tools
5. **Compare** the agent's implementation against the reference — look for:
   - Incorrect formula implementations
   - Off-by-one errors in loop equations or indexing
   - Wrong parameter values or ranges
   - Missing algorithmic steps
   - Logical errors that would produce wrong results
6. **Score** each dimension from 0.0 to 1.0 with detailed justification

Respond with ONLY a JSON object in this exact format:
{json_template}

Calculate overall_score as the weighted sum using the rubric weights.
"""
    else:
        return f"""You are an expert evaluator for scientific paper reproduction benchmarks.

## Task
Evaluate the AI agent's attempt to reproduce results from the paper:
"{task_config['paper']['title']}" by {task_config['paper']['author']}

## Grading Rubric
{rubric_text}

## Ground Truth Reference Files
The following reference files are available for you to read:
{chr(10).join(f'- /workspace/{f}' for f in sorted(gt_files))}

**You MUST read these files** to understand the correct methodology and
compare against the agent's output.

Key files:
- `/workspace/_ground_truth/metadata.md` — ground truth formulas & methodology
- `/workspace/_ground_truth/data/*.csv` — ground truth numerical data
- `/workspace/_ground_truth/{code_dir_name}/` — reference implementation code

## Agent's Expected Output Files
{chr(10).join(f'- /workspace/{f}' for f in expected_files)}

## Agent's Workspace Files
All files found: {json.dumps(workspace_files, indent=2)}

## Agent's Execution Trace (last 3000 chars)
{white_agent_trace[-3000:] if white_agent_trace else "(no trace available)"}

## Your Evaluation Steps

1. **Read** `/workspace/_ground_truth/metadata.md` for ground truth formulas AND Banned Libraries
2. **Read** every ground truth CSV in `/workspace/_ground_truth/data/`
3. **Read** every file in `/workspace/_ground_truth/{code_dir_name}/` as reference code
4. **Read** the agent's output files (code under agent's expected output, data CSVs, analysis)
5. **CHECK FOR BANNED LIBRARIES**: Scan ALL agent code files (imports, pip installs, subprocess calls) for any banned libraries listed in metadata.md.
   - If ANY banned library is used: **code_correctness score MUST be ≤ 0.2**
   - Common violations: importing banned packages, pip installing banned packages in code, calling banned CLI tools
6. **For data_accuracy**: compare the agent's CSV files in `/workspace/data/` against
   ground truth CSVs in `/workspace/_ground_truth/data/`.
   Allow ~30% relative tolerance for Monte Carlo stochastic results, ~10% for
   deterministic computations. Check column names, row counts, value ranges.
7. **For code_correctness**: compare the agent's code against reference code and
   ground truth formulas. Flag any bugs that would produce incorrect results.
8. **Score** each dimension from 0.0 to 1.0 with justification

Respond with ONLY a JSON object in this exact format:
{json_template}

Calculate overall_score as the weighted sum using the rubric weights.
"""


def _parse_grading_json(text: str) -> dict:
    """Extract JSON from Claude's grading response."""
    import re
    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    json_match = re.search(r'\{[\s\S]*"overall_score"[\s\S]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    logger.warning(f"Green agent: Could not parse grading JSON (len={len(text)})")
    return {
        "error": "parse_failure",
        "overall_score": 0.0,
        "scores": {},
        "summary": f"Failed to parse grading output. Raw: {text[:500]}",
    }


class PRBenchGreenAgentExecutor(AgentExecutor):
    """Green agent that orchestrates paper reproduction evaluation.

    Fully generic: loads all task-specific content from task.yaml.
    Uses Claude Code (or OpenAI Codex) inside Docker for model-judge grading.

    agent_type: "claude" (default) or "codex" — must match the white agent's CLI.
    """

    def __init__(self, docker_container_id: str | None = None, agent_type: str = "claude"):
        self.docker_container_id = docker_container_id
        self.agent_type = agent_type

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        timestamp_start = time.time()
        logger.info("Green agent: Received task, parsing configuration...")

        user_input = context.get_user_input()
        tags = parse_tags(user_input)
        white_agent_url = tags["white_agent_url"]
        task_config_incoming = json.loads(tags["task_config"])

        task_id = task_config_incoming["task_id"]
        task_dir = task_config_incoming["data_dir"]
        code_only = task_config_incoming.get("code_only", False)

        # Override docker container ID if passed in config
        container_id = task_config_incoming.get("docker_container_id") or self.docker_container_id

        # Load full task config from YAML
        task_config = load_task_config(task_dir)
        logger.info(f"Green agent: Task '{task_config['paper']['title']}'")

        workspace_dir = os.path.join(task_dir, "workspace")
        os.makedirs(workspace_dir, exist_ok=True)
        log_dir = os.path.join(workspace_dir, "eval_logs")
        os.makedirs(log_dir, exist_ok=True)

        try:
            # ============================================================
            # Step 1: Send instruction to white agent
            # ============================================================
            logger.info("[1/3] Sending instruction to white agent...")

            instruction_message = build_white_agent_instruction(task_config, task_dir,
                                                                code_only=code_only)

            with open(os.path.join(log_dir, "instruction_sent.txt"), "w") as f:
                f.write(instruction_message)

            logger.info(f"Green agent: Sending instruction ({len(instruction_message)} chars)")
            context_id = None

            response = await my_a2a.send_message(
                white_agent_url, instruction_message, context_id=context_id
            )

            res_root = response.root
            assert isinstance(res_root, SendMessageSuccessResponse)
            res_result = res_root.result

            if isinstance(res_result, Message):
                context_id = res_result.context_id
                text_parts = get_text_parts(res_result.parts)
                first_response = text_parts[0] if text_parts else ""
            elif isinstance(res_result, Task):
                context_id = res_result.context_id
                first_response = str(res_result.status)
            else:
                first_response = str(res_result)

            logger.info(f"Green agent: White agent response: {first_response[:300]}")

            # ============================================================
            # Step 2: Poll white agent for completion
            # ============================================================
            logger.info("[2/3] White agent working... polling every 30s")

            white_agent_trace = first_response
            poll_count = 0
            max_polls = 360  # 3 hours

            while poll_count < max_polls:
                if self._is_completed(white_agent_trace):
                    logger.info("Green agent: White agent completed.")
                    break

                poll_count += 1
                await asyncio.sleep(POLL_INTERVAL)

                try:
                    poll_response = await my_a2a.send_message(
                        white_agent_url,
                        "STATUS_CHECK",
                        context_id=context_id,
                    )
                    poll_root = poll_response.root
                    if isinstance(poll_root, SendMessageSuccessResponse):
                        poll_result = poll_root.result
                        if isinstance(poll_result, Message):
                            text_parts = get_text_parts(poll_result.parts)
                            poll_text = text_parts[0] if text_parts else ""
                            white_agent_trace += f"\n\n--- Poll #{poll_count} ---\n{poll_text}"

                            logger.info(
                                f"Green agent: Poll #{poll_count} "
                                f"({poll_count * POLL_INTERVAL}s): "
                                f"{poll_text[:150]}"
                            )

                            if self._is_completed(poll_text):
                                break
                except Exception as e:
                    logger.warning(f"Green agent: Poll #{poll_count} failed: {e}")

            workspace_files = collect_workspace_files(workspace_dir)
            logger.info(f"Green agent: {len(workspace_files)} files in workspace")

            await event_queue.enqueue_event(
                new_agent_text_message(
                    f"[2/3] White agent finished. {len(workspace_files)} files."
                )
            )

            with open(os.path.join(log_dir, "white_agent_trace.log"), "w") as f:
                f.write(white_agent_trace)

            # ============================================================
            # Step 2.5: Copy ground truth into workspace for grading agent
            # ============================================================
            # The Docker container does NOT have /task mounted (to prevent
            # the white agent from reading answers).  Now that the white
            # agent has finished, we copy ground truth files into
            # /workspace/_ground_truth/ so the grading agent running
            # inside Docker can read them directly.
            _copy_ground_truth_to_workspace(
                task_config, task_dir, workspace_dir, code_only=code_only
            )

            # ============================================================
            # Step 3: Grade results using Claude Code model-judge
            # ============================================================
            await event_queue.enqueue_event(
                new_agent_text_message("[3/3] Grading with model-judge (Claude Code in Docker)...")
            )

            grading_prompt = build_grading_prompt(
                task_config, task_dir, workspace_dir, white_agent_trace,
                code_only=code_only
            )

            with open(os.path.join(log_dir, "grading_prompt.txt"), "w") as f:
                f.write(grading_prompt)

            # Run grading inside Docker (in thread to avoid blocking event loop)
            grading_result = await asyncio.to_thread(
                self._run_grading, grading_prompt, container_id, log_dir, self.agent_type
            )

            # Save final report
            time_used = time.time() - timestamp_start
            final_report = {
                "task_id": task_id,
                "paper": task_config["paper"],
                "grading": grading_result,
                "time_used_seconds": time_used,
                "workspace_files": workspace_files,
                "poll_count": poll_count,
            }

            report_path = os.path.join(log_dir, "eval_report.json")
            with open(report_path, "w") as f:
                json.dump(final_report, f, indent=2, default=str)

            overall = grading_result.get("overall_score", 0.0)
            summary = grading_result.get("summary", "No summary")

            await event_queue.enqueue_event(
                new_agent_text_message(
                    f"[3/3] Evaluation complete.\n"
                    f"Overall Score: {overall}\n"
                    f"Summary: {summary}\n"
                    f"Time: {time_used:.0f}s\n"
                    f"Report: {report_path}\n\n"
                    f"{json.dumps(final_report, indent=2, default=str)}"
                )
            )

        except Exception as e:
            logger.exception(f"Green agent: Error: {e}")
            await event_queue.enqueue_event(
                new_agent_text_message(f"ERROR: {e}")
            )

    def _run_grading(self, grading_prompt: str, container_id: str | None,
                     log_dir: str, agent_type: str = "claude") -> dict:
        """Run model-judge grading using the agent CLI available in the container.

        For claude:    uses Claude Code (`claude`) with Anthropic credentials.
        For codex:     uses OpenAI Codex (`codex --full-auto`) with OpenAI credentials.
        For opencode:  uses OpenCode (`opencode run`) with auto-detected credentials.
        Falls back to running locally if no container_id.
        """
        import subprocess

        # Write grading prompt to workspace so the agent can read it in Docker
        prompt_path = os.path.join(log_dir, "_grading_prompt.txt")
        with open(prompt_path, "w") as f:
            f.write(grading_prompt)

        grading_instruction = (
            "Read /workspace/eval_logs/_grading_prompt.txt and follow the instructions. "
            "Return ONLY the JSON result as specified."
        )

        if container_id:
            from src.my_util.agent_env import build_docker_exec_env_flags, resolve_env
            env_flags = build_docker_exec_env_flags(agent_type)

            if agent_type == "codex":
                logger.info("Green agent: Running grading via OpenAI Codex in Docker...")
                grading_out = "/workspace/eval_logs/_grading_output.txt"
                cmd = [
                    "docker", "exec",
                    "-u", "agent",
                    "-w", "/workspace",
                    *env_flags,
                    container_id,
                    "codex", "exec",
                    "--dangerously-bypass-approvals-and-sandbox",
                    "--skip-git-repo-check",
                    "--json",
                    "--output-last-message", grading_out,
                    "--enable", "unified_exec",
                    "--",
                    grading_instruction,
                ]
            elif agent_type == "opencode":
                logger.info("Green agent: Running grading via OpenCode in Docker...")
                model = resolve_env("opencode").get("OPENCODE_MODEL", "")
                model_flags = ["--model", model] if model else []

                cmd = [
                    "docker", "exec",
                    "-u", "agent",
                    "-w", "/workspace",
                    *env_flags,
                    container_id,
                    "opencode", "run",
                    *model_flags,
                    grading_instruction,
                ]
            else:
                logger.info("Green agent: Running grading via Claude Code in Docker...")
                cmd = [
                    "docker", "exec",
                    "-u", "agent",
                    "-w", "/workspace",
                    *env_flags,
                    container_id,
                    "claude",
                    "-p", grading_instruction,
                    "--output-format", "text",
                    "--max-turns", "100",
                ]
        else:
            # Fallback: run locally
            if agent_type == "codex":
                logger.info("Green agent: Running grading via local OpenAI Codex (exec mode)...")
                grading_out = os.path.join(log_dir, "_grading_output.txt")
                cmd = [
                    "codex", "exec",
                    "--dangerously-bypass-approvals-and-sandbox",
                    "--skip-git-repo-check",
                    "--json",
                    "--output-last-message", grading_out,
                    "--enable", "unified_exec",
                    "--",
                    grading_prompt,
                ]
            elif agent_type == "opencode":
                logger.info("Green agent: Running grading via local OpenCode...")
                from src.my_util.agent_env import resolve_env as _resolve
                model = _resolve("opencode").get("OPENCODE_MODEL", "")
                model_flags = ["--model", model] if model else []
                cmd = [
                    "opencode", "run",
                    *model_flags,
                    grading_prompt,
                ]
            else:
                logger.info("Green agent: Running grading via local Claude Code...")
                cmd = [
                    "env", "-u", "CLAUDECODE", "-u", "CLAUDE_CODE_ENTRYPOINT",
                    "-u", "CLAUDE_CODE_SSE_PORT",
                    "claude",
                    "-p", grading_prompt,
                    "--output-format", "text",
                    "--max-turns", "100",
                ]

        try:
            result = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,   # codex exec requires non-TTY stdin
                capture_output=True,
                text=True,
                timeout=600,
            )

            with open(os.path.join(log_dir, "grading_trace.log"), "w") as f:
                f.write(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n")

            logger.info(f"Green agent: Grading exit={result.returncode}, output={len(result.stdout)} chars")

            # For codex: parse the --output-last-message file (plain text, easier to parse).
            # For claude / opencode: parse stdout directly.
            if agent_type == "codex":
                out_host = os.path.join(log_dir, "_grading_output.txt")
                if os.path.exists(out_host):
                    with open(out_host) as f:
                        grading_text = f.read()
                    logger.info(f"Green agent: Codex grading output file: {len(grading_text)} chars")
                else:
                    logger.warning("Green agent: Codex grading output file not found, falling back to stdout")
                    grading_text = result.stdout
                return _parse_grading_json(grading_text)
            else:
                return _parse_grading_json(result.stdout)

        except subprocess.TimeoutExpired:
            logger.error("Green agent: Grading timed out")
            return {"error": "timeout", "overall_score": 0.0, "scores": {}, "summary": "Grading timed out"}
        except FileNotFoundError:
            logger.error("Green agent: CLI not found for grading")
            return {"error": "not_found", "overall_score": 0.0, "scores": {}, "summary": "CLI not found"}

    def _is_completed(self, text: str) -> bool:
        # Only check the FIRST non-empty line of each poll response.
        # Codex --json output contains "status":"completed" in its JSON stream,
        # which would falsely match if we searched the whole text.
        for line in text.split('\n'):
            stripped = line.strip()
            if stripped:
                return stripped.startswith("COMPLETED")
        return False

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


def start_green_agent(
    agent_name: str = "prbench_green_agent",
    host: str = "localhost",
    port: int = 9001,
    docker_container_id: str | None = None,
    agent_type: str = "claude",
):
    """Start the green agent A2A server.

    agent_type: "claude" (default) or "codex" — used to pick the right CLI for grading.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [GREEN] %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("green_agent.log"),
        ],
    )
    logger.info(f"Starting green agent on {host}:{port} (grading via {agent_type})")

    agent_card_dict = load_agent_card_toml(agent_name)
    url = f"http://{host}:{port}"
    agent_card_dict["url"] = url

    request_handler = DefaultRequestHandler(
        agent_executor=PRBenchGreenAgentExecutor(
            docker_container_id=docker_container_id,
            agent_type=agent_type,
        ),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=AgentCard(**agent_card_dict),
        http_handler=request_handler,
    )

    uvicorn.run(app.build(), host=host, port=port)
