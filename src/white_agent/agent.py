"""White agent implementation - runs Claude Code inside Docker for paper reproduction.

Architecture:
- Receives instruction via A2A from the green agent
- Runs Claude Code CLI inside the Docker container (via `docker exec`)
- Tracks subprocess PID for monitoring
- Supports multi-turn A2A: first message starts task, subsequent messages check status
- Captures full trace log
- Fully isolated: Claude Code runs inside Docker with its own env vars
"""

import json
import logging
import os
import subprocess
import threading
import time
import uuid

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill, AgentCard, AgentCapabilities
from a2a.utils import new_agent_text_message

logger = logging.getLogger(__name__)


def _export_traces(container_id: str, dest_dir: str, agent_type: str = "claude") -> None:
    """Copy agent traces from container to host via docker cp.

    For opencode, selectively copies only useful files (db, logs, storage),
    skipping the bulky snapshot/ directory with thousands of git objects.
    """
    os.makedirs(dest_dir, exist_ok=True)
    if agent_type == "codex":
        container_path = "/home/agent/.codex"
        agent_label = "Codex"
    elif agent_type == "opencode":
        agent_label = "OpenCode"
        # Selective copy — skip snapshot/ which has thousands of git blobs
        src_base = "/home/agent/.local/share/opencode"
        for sp in ["opencode.db", "opencode.db-wal", "opencode.db-shm",
                    "log", "storage", "bin"]:
            try:
                subprocess.run(
                    ["docker", "cp", f"{container_id}:{src_base}/{sp}", dest_dir],
                    capture_output=True, timeout=60,
                )
            except Exception:
                pass
        logger.info(f"Exported {agent_label} traces to {dest_dir}")
        return
    else:
        container_path = "/home/agent/.claude"
        agent_label = "Claude"
    try:
        subprocess.run(
            ["docker", "cp", f"{container_id}:{container_path}", dest_dir],
            capture_output=True, timeout=60,
        )
        logger.info(f"Exported {agent_label} traces to {dest_dir}")
    except Exception as e:
        logger.warning(f"Failed to export {agent_label} traces: {e}")


def prepare_white_agent_card(url: str) -> AgentCard:
    skill = AgentSkill(
        id="paper_reproduction",
        name="Scientific Paper Reproduction",
        description="Reads a scientific paper and reproduces its results using Claude Code inside Docker",
        tags=["paper reproduction", "scientific computing", "white agent"],
        examples=[],
    )
    card = AgentCard(
        name="prbench_white_agent",
        description="White agent for PRBench - reproduces scientific papers using Claude Code in Docker",
        url=url,
        version="0.3.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )
    return card


class RunningTask:
    """Tracks a running agent CLI subprocess inside Docker."""

    def __init__(self, session_id: str, process, workspace_dir: str, trace_path: str,
                 container_id: str | None = None, agent_type: str = "claude"):
        self.session_id = session_id
        self.process = process
        self.workspace_dir = workspace_dir
        self.trace_path = trace_path
        self.container_id = container_id
        self.agent_type = agent_type
        self.start_time = time.time()
        self.stdout_data = ""
        self.stderr_data = ""
        self.completed = False
        self.exit_code = None
        self._lock = threading.Lock()

        # Background threads for reading output
        self._stdout_thread = threading.Thread(
            target=self._read_stream, args=(process.stdout, "stdout"), daemon=True
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stream, args=(process.stderr, "stderr"), daemon=True
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

        # Monitor thread
        self._monitor_thread = threading.Thread(target=self._monitor, daemon=True)
        self._monitor_thread.start()

    def _read_stream(self, stream, name: str):
        try:
            for line in iter(stream.readline, ""):
                with self._lock:
                    if name == "stdout":
                        self.stdout_data += line
                    else:
                        self.stderr_data += line
        except (ValueError, OSError):
            pass

    def _monitor(self):
        self.process.wait()
        with self._lock:
            self.completed = True
            self.exit_code = self.process.returncode
        self._save_trace()
        # Export agent traces from container
        if self.container_id:
            if self.agent_type == "codex":
                folder = "_codex_traces"
            elif self.agent_type == "opencode":
                folder = "_opencode_traces"
            else:
                folder = "_claude_traces"
            trace_dest = os.path.join(self.workspace_dir, "eval_logs", folder)
            _export_traces(self.container_id, trace_dest, agent_type=self.agent_type)
        _agent_labels = {"codex": "OpenAI Codex", "opencode": "OpenCode", "claude": "Claude Code"}
        agent_label = _agent_labels.get(self.agent_type, "Claude Code")
        logger.info(f"White agent [{self.session_id}]: {agent_label} finished (exit={self.exit_code})")

    def _save_trace(self):
        with self._lock:
            trace = (
                f"=== Session: {self.session_id} ===\n"
                f"=== Start: {time.ctime(self.start_time)} ===\n"
                f"=== Duration: {time.time() - self.start_time:.0f}s ===\n"
                f"=== Exit Code: {self.exit_code} ===\n\n"
                f"=== STDOUT ===\n{self.stdout_data}\n\n"
                f"=== STDERR ===\n{self.stderr_data}\n"
            )
        try:
            with open(self.trace_path, "w") as f:
                f.write(trace)
        except OSError as e:
            logger.error(f"Failed to save trace: {e}")

    _SKIP_DIRS = {".git", "__pycache__", "node_modules"}

    def get_status(self) -> dict:
        with self._lock:
            elapsed = time.time() - self.start_time
            workspace_files = []
            if os.path.isdir(self.workspace_dir):
                for root, dirs, fnames in os.walk(self.workspace_dir):
                    dirs[:] = [d for d in dirs if d not in self._SKIP_DIRS]
                    rel_root = os.path.relpath(root, self.workspace_dir)
                    if "/snapshot/" in rel_root or rel_root.endswith("/snapshot"):
                        dirs.clear()
                        continue
                    for fname in fnames:
                        if not fname.startswith("_"):
                            rel = os.path.relpath(os.path.join(root, fname), self.workspace_dir)
                            workspace_files.append(rel)

            return {
                "session_id": self.session_id,
                "completed": self.completed,
                "exit_code": self.exit_code,
                "pid": self.process.pid,
                "elapsed_seconds": elapsed,
                "stdout_length": len(self.stdout_data),
                "stderr_length": len(self.stderr_data),
                "workspace_files": sorted(workspace_files),
                "last_stdout": self.stdout_data[-1500:] if self.stdout_data else "",
            }

    def get_full_trace(self) -> str:
        with self._lock:
            return (
                f"=== STDOUT ({len(self.stdout_data)} chars) ===\n"
                f"{self.stdout_data}\n\n"
                f"=== STDERR ({len(self.stderr_data)} chars) ===\n"
                f"{self.stderr_data}"
            )


class ClaudeCodeWhiteAgentExecutor(AgentExecutor):
    """White agent that runs Claude Code or OpenAI Codex inside Docker.

    Multi-turn A2A conversation:
    - First message: receives instruction, starts agent CLI in Docker, returns "STARTED"
    - Subsequent messages (STATUS_CHECK): returns current status (RUNNING/COMPLETED)

    agent_type: "claude" (default) or "codex"
    """

    def __init__(self, workspace_base: str | None = None,
                 docker_container_id: str | None = None,
                 agent_type: str = "claude"):
        self.workspace_base = workspace_base or "/tmp/prbench_workspaces"
        self.docker_container_id = docker_container_id
        self.agent_type = agent_type
        os.makedirs(self.workspace_base, exist_ok=True)
        self._tasks: dict[str, RunningTask] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()
        ctx_id = context.context_id

        # Status poll for existing task
        if ctx_id and ctx_id in self._tasks:
            task = self._tasks[ctx_id]
            status = task.get_status()

            if status["completed"]:
                trace = task.get_full_trace()
                response = (
                    f"COMPLETED\n"
                    f"Claude Code finished in {status['elapsed_seconds']:.0f}s "
                    f"(exit code: {status['exit_code']})\n"
                    f"Workspace files: {json.dumps(status['workspace_files'])}\n\n"
                    f"TASK_DONE\n\n"
                    f"Trace (last 3000 chars):\n{trace[-3000:]}"
                )
            else:
                response = (
                    f"RUNNING\n"
                    f"PID: {status['pid']}\n"
                    f"Elapsed: {status['elapsed_seconds']:.0f}s\n"
                    f"Output so far: {status['stdout_length']} chars\n"
                    f"Workspace files: {json.dumps(status['workspace_files'])}\n"
                    f"Recent output:\n{status['last_stdout'][-500:]}"
                )

            await event_queue.enqueue_event(
                new_agent_text_message(response, context_id=ctx_id)
            )
            return

        # === First message: start new task ===
        session_id = uuid.uuid4().hex[:8]
        logger.info(f"White agent [{session_id}]: Starting new paper reproduction task")

        # The workspace is the bind-mounted directory. Since we run inside Docker,
        # files created in /workspace inside the container appear in workspace_base on host.
        workspace_dir = self.workspace_base
        trace_path = os.path.join(workspace_dir, f"_trace_{session_id}.log")

        # Save instruction for Claude Code to read inside the container
        instruction_path = os.path.join(workspace_dir, "_instruction.md")
        with open(instruction_path, "w") as f:
            f.write(user_input)

        # Build the prompt for Claude Code
        # Claude Code runs inside /workspace in the container
        prompt = (
            f"You have a paper reproduction task. "
            f"Read the full instruction from /workspace/_instruction.md and follow it completely. "
            f"Your working directory is /workspace. "
            f"Create all output files there (reproduction/, data/ subdirectories). "
            f"When you are completely done with ALL files, print TASK_DONE as your final message."
        )

        if not self.docker_container_id:
            error_msg = "ERROR: No Docker container ID configured. Cannot run agent CLI."
            logger.error(f"White agent [{session_id}]: {error_msg}")
            await event_queue.enqueue_event(
                new_agent_text_message(error_msg, context_id=ctx_id or session_id)
            )
            return

        # Launch agent CLI inside Docker via docker exec (as non-root 'agent' user)
        import subprocess
        from src.my_util.agent_env import build_docker_exec_env_flags, resolve_env

        env_flags = build_docker_exec_env_flags(self.agent_type)

        if self.agent_type == "codex":
            cmd = [
                "docker", "exec",
                "-u", "agent",
                "-w", "/workspace",
                *env_flags,
                self.docker_container_id,
                "codex", "exec",
                "--dangerously-bypass-approvals-and-sandbox",
                "--skip-git-repo-check",
                "--json",
                "--enable", "unified_exec",
                "--",
                prompt,
            ]
            logger.info(f"White agent [{session_id}]: docker exec {self.docker_container_id[:12]} codex exec ...")
        elif self.agent_type == "opencode":
            model = resolve_env("opencode").get("OPENCODE_MODEL", "")
            model_flags = ["--model", model] if model else []

            cmd = [
                "docker", "exec",
                "-u", "agent",
                "-w", "/workspace",
                *env_flags,
                self.docker_container_id,
                "opencode", "run",
                *model_flags,
                prompt,
            ]
            logger.info(f"White agent [{session_id}]: docker exec {self.docker_container_id[:12]} opencode run ...")
        else:
            cmd = [
                "docker", "exec",
                "-u", "agent",
                "-w", "/workspace",
                *env_flags,
                self.docker_container_id,
                "claude",
                "-p", prompt,
                "--output-format", "text",
                "--max-turns", "500",
            ]
            logger.info(f"White agent [{session_id}]: docker exec {self.docker_container_id[:12]} claude ...")

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,   # codex exec requires stdin=/dev/null in headless mode
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        _agent_labels = {"codex": "OpenAI Codex", "opencode": "OpenCode", "claude": "Claude Code"}
        agent_label = _agent_labels.get(self.agent_type, "Claude Code")
        logger.info(f"White agent [{session_id}]: {agent_label} started in Docker, host PID={process.pid}")

        # Track the task
        running_task = RunningTask(session_id, process, workspace_dir, trace_path,
                                   container_id=self.docker_container_id,
                                   agent_type=self.agent_type)
        effective_ctx_id = ctx_id or session_id
        self._tasks[effective_ctx_id] = running_task

        response = (
            f"STARTED\n"
            f"Session: {session_id}\n"
            f"PID: {process.pid}\n"
            f"Container: {self.docker_container_id[:12]}\n"
            f"Context ID for polling: {effective_ctx_id}\n"
            f"{agent_label} is now working on the paper reproduction task inside Docker."
        )

        await event_queue.enqueue_event(
            new_agent_text_message(response, context_id=effective_ctx_id)
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        ctx_id = context.context_id
        if ctx_id and ctx_id in self._tasks:
            task = self._tasks[ctx_id]
            try:
                task.process.terminate()
                logger.info(f"White agent: Terminated PID {task.process.pid}")
            except OSError:
                pass


def start_white_agent(
    agent_name: str = "prbench_white_agent",
    host: str = "localhost",
    port: int = 9002,
    workspace_base: str | None = None,
    docker_container_id: str | None = None,
    agent_type: str = "claude",
):
    """Start the white agent A2A server.

    agent_type: "claude" (default) uses Claude Code CLI; "codex" uses OpenAI Codex CLI.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [WHITE] %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("white_agent.log"),
        ],
    )
    _agent_labels = {"codex": "OpenAI Codex", "opencode": "OpenCode", "claude": "Claude Code"}
    agent_label = _agent_labels.get(agent_type, "Claude Code")
    logger.info(f"Starting white agent ({agent_label}) on {host}:{port}")
    if docker_container_id:
        logger.info(f"Docker container: {docker_container_id[:12]}")

    url = f"http://{host}:{port}"
    card = prepare_white_agent_card(url)

    request_handler = DefaultRequestHandler(
        agent_executor=ClaudeCodeWhiteAgentExecutor(
            workspace_base=workspace_base,
            docker_container_id=docker_container_id,
            agent_type=agent_type,
        ),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )

    uvicorn.run(app.build(), host=host, port=port)
