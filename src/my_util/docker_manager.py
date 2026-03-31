"""Docker environment manager for running task code in isolation.

Supports installing Claude Code CLI inside the container for fully isolated
agent execution. Environment variables (Anthropic credentials) are injected
at container creation time.
"""

import json
import logging
import os
import subprocess
import docker
from docker.errors import APIError

logger = logging.getLogger(__name__)


class DockerEnvironment:
    """Manages a Docker container for executing task reproduction code.

    Provides:
    - Bind mount for workspace (read-write); NO task directory mount
      (ground truth isolation: the white agent must not see reference data/code)
    - Python scientific stack (numpy, scipy, matplotlib)
    - Optional Claude Code CLI installation (Node.js + npm)
    - Anthropic credentials injection
    """

    def __init__(
        self,
        image: str = "python:3.11-slim",
        task_dir: str | None = None,
        workspace_dir: str | None = None,
        memory_limit: str = "4g",
        timeout: int = 10800,
        env_vars: dict[str, str] | None = None,
        pip_packages: list[str] | None = None,
    ):
        self.image = image
        self.task_dir = task_dir
        self.workspace_dir = workspace_dir
        self.memory_limit = memory_limit
        self.timeout = timeout
        self.env_vars = env_vars or {}
        self.pip_packages = pip_packages or ["numpy", "scipy", "matplotlib"]
        self.container = None
        self.container_id: str | None = None
        self.client = docker.from_env()

    def start(self) -> None:
        """Pull image and start the container with bind mounts."""
        logger.info(f"Pulling Docker image: {self.image}")
        try:
            self.client.images.pull(self.image)
        except APIError as e:
            logger.warning(f"Could not pull image (may already exist locally): {e}")

        volumes = {}
        # NOTE: task_dir is intentionally NOT mounted into the container.
        # The white agent must not have access to ground truth data or
        # reference code.  Paper content and images are delivered via the
        # prompt / workspace instead.  Ground truth files are copied into
        # the workspace only AFTER the white agent finishes, right before
        # the grading agent runs.
        if self.workspace_dir:
            os.makedirs(self.workspace_dir, exist_ok=True)
            volumes[os.path.abspath(self.workspace_dir)] = {
                "bind": "/workspace",
                "mode": "rw",
            }

        # Build environment list for container
        environment = dict(self.env_vars)

        logger.info("Creating Docker container...")
        self.container = self.client.containers.run(
            self.image,
            command="sleep infinity",
            volumes=volumes,
            mem_limit=self.memory_limit,
            detach=True,
            working_dir="/workspace",
            environment=environment,
        )
        self.container_id = self.container.id
        logger.info(f"Container started: {self.container.short_id} (id={self.container_id[:12]})")

        # Create a non-root user matching the host user's UID/GID
        # This ensures files created inside the container are owned by the host user
        host_uid = os.getuid()
        host_gid = os.getgid()
        self.host_uid = host_uid
        self.host_gid = host_gid
        create_user_result = self.exec_command(
            f"groupadd -o -g {host_gid} agent 2>/dev/null || true; "
            f"useradd -o -m -s /bin/bash -u {host_uid} -g {host_gid} agent 2>/dev/null || true; "
            f"chown -R {host_uid}:{host_gid} /workspace; "
            f"chown -R {host_uid}:{host_gid} /home/agent; "
            f"id agent"
        )
        logger.info(f"Container user 'agent': {create_user_result['stdout'].strip()}")

        # Install Python dependencies
        if self.pip_packages:
            pkgs = " ".join(self.pip_packages)
            logger.info(f"Installing Python packages: {pkgs}")
            self.exec_command(f"pip install --quiet {pkgs}", timeout=300)

        # Create directory structure in workspace with wide permissions
        # (Docker userns remapping may cause UID mismatches between container and host)
        self.exec_command(
            "mkdir -p /workspace/reproduction /workspace/data /workspace/eval_logs && "
            "chmod -R 777 /workspace"
        )

    def install_claude_code(self) -> bool:
        """Install Node.js and Claude Code CLI inside the container.

        Returns True if installation succeeded.
        """
        if self.container is None:
            raise RuntimeError("Container not started")

        logger.info("Installing Node.js and Claude Code CLI in container...")

        # Install curl and Node.js
        steps = [
            ("Installing curl...", "apt-get update -qq && apt-get install -y -qq curl ca-certificates > /dev/null 2>&1"),
            ("Installing Node.js 22...", "curl -fsSL https://deb.nodesource.com/setup_22.x | bash - > /dev/null 2>&1 && apt-get install -y -qq nodejs > /dev/null 2>&1"),
            ("Installing Claude Code CLI...", "npm install -g @anthropic-ai/claude-code > /dev/null 2>&1"),
        ]

        for desc, cmd in steps:
            logger.info(f"  {desc}")
            result = self.exec_command(cmd, timeout=300)
            if result["exit_code"] != 0:
                logger.error(f"  FAILED: {result['stderr'][:500]}")
                return False

        # Verify installation
        result = self.exec_command("claude --version")
        if result["exit_code"] == 0:
            version = result["stdout"].strip()
            logger.info(f"  Claude Code installed: {version}")
        else:
            logger.error(f"  Claude Code verification failed: {result['stderr'][:200]}")
            return False

        # Setup Claude Code settings for non-interactive use as 'agent' user
        # Create settings that allow all tools without prompting
        settings_content = {
            "permissions": {
                "allow": ["Bash(*)", "Read(*)", "Write(*)", "Edit(*)", "Glob(*)", "Grep(*)"],
                "deny": []
            }
        }
        # Write settings file in two steps to avoid shell escaping issues
        self.exec_command("mkdir -p /home/agent/.claude")
        settings_json = json.dumps(settings_content)
        # Use a heredoc approach for reliable content writing
        write_cmd = f'''cat > /home/agent/.claude/settings.json << 'EOFSET'
{settings_json}
EOFSET'''
        result = self.exec_command(write_cmd)
        if result["exit_code"] != 0:
            logger.warning(f"  Settings write warning: {result['stderr'][:200]}")

        # Fix ownership: chown the ENTIRE .claude directory (not just the file)
        # so the agent user can create additional files (JSONL logs, cache, etc.)
        self.exec_command(
            f"chown -R {self.host_uid}:{self.host_gid} /home/agent/.claude && "
            f"chown -R {self.host_uid}:{self.host_gid} /home/agent"
        )

        # Verify settings were written
        verify = self.exec_command("cat /home/agent/.claude/settings.json")
        if "allow" in verify["stdout"]:
            logger.info("  Claude Code settings configured for non-interactive use")
        else:
            logger.warning(f"  Settings verification failed: {verify['stdout'][:100]}")

        # Accept TOS non-interactively by running claude with --accept-tos as agent user
        # This prevents Claude Code from hanging on first run
        tos_result = self.exec_command(
            f'su - agent -c "HOME=/home/agent claude --accept-tos --output-format text -p \'echo hello\' --max-turns 1" '
            f'2>/dev/null || true'
        )
        logger.info(f"  TOS acceptance: exit={tos_result['exit_code']}")

        return True

    def install_codex(self) -> bool:
        """Install Node.js and OpenAI Codex CLI inside the container.

        Returns True if installation succeeded.
        """
        if self.container is None:
            raise RuntimeError("Container not started")

        logger.info("Installing Node.js and OpenAI Codex CLI in container...")

        steps = [
            ("Installing curl...", "apt-get update -qq && apt-get install -y -qq curl ca-certificates > /dev/null 2>&1"),
            ("Installing Node.js 22...", "curl -fsSL https://deb.nodesource.com/setup_22.x | bash - > /dev/null 2>&1 && apt-get install -y -qq nodejs > /dev/null 2>&1"),
            ("Installing OpenAI Codex CLI...", "npm install -g @openai/codex > /dev/null 2>&1"),
        ]

        for desc, cmd in steps:
            logger.info(f"  {desc}")
            result = self.exec_command(cmd, timeout=300)
            if result["exit_code"] != 0:
                logger.error(f"  FAILED: {result['stderr'][:500]}")
                return False

        # Verify installation
        result = self.exec_command("codex --version")
        if result["exit_code"] == 0:
            version = result["stdout"].strip()
            logger.info(f"  OpenAI Codex CLI installed: {version}")
        else:
            logger.error(f"  OpenAI Codex CLI verification failed: {result['stderr'][:200]}")
            return False

        # Create ~/.codex/auth.json for API key authentication (CI/headless mode)
        # This is the preferred auth mechanism for non-interactive runs.
        openai_api_key = self.env_vars.get("OPENAI_API_KEY", "")
        if openai_api_key:
            auth_content = json.dumps({"OPENAI_API_KEY": openai_api_key})
            self.exec_command("mkdir -p /home/agent/.codex")
            write_cmd = f'''cat > /home/agent/.codex/auth.json << 'EOFAUTH'\n{auth_content}\nEOFAUTH'''
            r = self.exec_command(write_cmd)
            if r["exit_code"] == 0:
                logger.info("  Codex auth.json written to /home/agent/.codex/auth.json")
            else:
                logger.warning(f"  Failed to write auth.json: {r['stderr'][:200]}")

        # Fix ownership so the agent user can write config/cache files
        self.exec_command(
            f"chown -R {self.host_uid}:{self.host_gid} /home/agent"
        )

        return True

    def install_opencode(self) -> bool:
        """Install Node.js, git, and OpenCode CLI inside the container.

        Returns True if installation succeeded.
        """
        if self.container is None:
            raise RuntimeError("Container not started")

        logger.info("Installing Node.js, git, and OpenCode CLI in container...")

        steps = [
            ("Installing curl and git...",
             "apt-get update -qq && apt-get install -y -qq curl ca-certificates git > /dev/null 2>&1"),
            ("Installing Node.js 22...",
             "curl -fsSL https://deb.nodesource.com/setup_22.x | bash - > /dev/null 2>&1 "
             "&& apt-get install -y -qq nodejs > /dev/null 2>&1"),
            ("Installing OpenCode CLI...",
             "npm install -g opencode-ai@latest > /dev/null 2>&1"),
        ]

        for desc, cmd in steps:
            logger.info(f"  {desc}")
            result = self.exec_command(cmd, timeout=300)
            if result["exit_code"] != 0:
                logger.error(f"  FAILED: {result['stderr'][:500]}")
                return False

        # Verify installation
        result = self.exec_command("opencode --version 2>/dev/null || opencode version 2>/dev/null || echo 'opencode installed'")
        logger.info(f"  OpenCode CLI installed: {result['stdout'].strip()}")

        # Init a git repo in /workspace (opencode may require one)
        self.exec_command(
            'cd /workspace && git init -q && git config user.email "agent@prbench" && '
            'git config user.name "PRBench Agent" && git add -A 2>/dev/null; '
            'git commit -m "init" --allow-empty -q 2>/dev/null || true'
        )
        logger.info("  Initialized git repo in /workspace")

        # Write opencode.json config for provider base URLs if custom ones are set.
        # env_vars already have correct values (including /v1 where needed)
        # from the centralized resolve_env() in launcher.py.
        config: dict = {}
        model = self.env_vars.get("OPENCODE_MODEL", "")
        if model:
            config["model"] = model

        providers: dict = {}
        anthropic_base = self.env_vars.get("ANTHROPIC_BASE_URL", "")
        if anthropic_base:
            providers["anthropic"] = {"options": {"baseURL": anthropic_base}}
        openai_base = self.env_vars.get("OPENAI_BASE_URL", "")
        if openai_base:
            # When a custom base URL is set, the target is an OpenAI-compatible
            # proxy (e.g. yunwu.ai), not the real OpenAI API.  Register a
            # custom provider named "openai_compat" using the
            # @ai-sdk/openai-compatible adapter so that ANY model name is
            # accepted (the built-in "openai" provider rejects non-OpenAI
            # model IDs like kimi-k2.5).
            model_name = model.split("/", 1)[1] if "/" in model else model
            providers["openai_compat"] = {
                "npm": "@ai-sdk/openai-compatible",
                "name": "OpenAI Compatible",
                "options": {
                    "baseURL": openai_base,
                    "apiKey": "{env:OPENAI_API_KEY}",
                },
                "models": {model_name: {}},
            }
            # Rewrite model to use the custom provider prefix
            if model.startswith("openai/") or model.startswith("openai_compat/"):
                config["model"] = f"openai_compat/{model_name}"

        if providers:
            config["provider"] = providers

        # Write global config
        self.exec_command("mkdir -p /home/agent/.config/opencode")
        if config:
            config_json = json.dumps(config, indent=2)
            write_cmd = f'''cat > /home/agent/.config/opencode/opencode.json << 'EOFCFG'
{config_json}
EOFCFG'''
            r = self.exec_command(write_cmd)
            if r["exit_code"] == 0:
                logger.info("  OpenCode config written to /home/agent/.config/opencode/opencode.json")
            else:
                logger.warning(f"  Failed to write opencode config: {r['stderr'][:200]}")

        # Pre-install @ai-sdk/openai-compatible if a custom provider is registered
        if "openai_compat" in providers:
            logger.info("  Installing @ai-sdk/openai-compatible...")
            self.exec_command(
                "cd /home/agent/.config/opencode && "
                "npm init -y > /dev/null 2>&1 && "
                "npm install @ai-sdk/openai-compatible > /dev/null 2>&1"
            )

        # Create XDG data dir for opencode sessions
        self.exec_command("mkdir -p /home/agent/.local/share/opencode")

        # Fix ownership
        self.exec_command(
            f"chown -R {self.host_uid}:{self.host_gid} /home/agent"
        )

        return True

    def exec_command(self, cmd: str, timeout: int | None = None) -> dict:
        """Execute a command in the container via Docker SDK.

        Returns dict with exit_code, stdout, stderr.
        """
        if self.container is None:
            raise RuntimeError("Container not started")

        try:
            exit_code, output = self.container.exec_run(
                ["bash", "-c", cmd],
                workdir="/workspace",
                demux=True,
            )
            stdout = output[0].decode("utf-8", errors="replace") if output[0] else ""
            stderr = output[1].decode("utf-8", errors="replace") if output[1] else ""
            return {"exit_code": exit_code, "stdout": stdout, "stderr": stderr}
        except Exception as e:
            return {"exit_code": -1, "stdout": "", "stderr": str(e)}

    def _build_docker_exec_env_flags(self) -> list[str]:
        """Build -e flags for docker exec with all Anthropic env vars + HOME."""
        flags = ["-e", "HOME=/home/agent"]
        for key, val in self.env_vars.items():
            if val:
                flags.extend(["-e", f"{key}={val}"])
        return flags

    def exec_claude_code(self, prompt: str, max_turns: int = 100,
                         timeout: int | None = None) -> subprocess.Popen:
        """Start Claude Code inside the container as a non-blocking subprocess.

        Uses `docker exec` via subprocess.Popen for PID tracking and streaming.
        Runs as non-root 'agent' user to avoid permission restrictions.
        Returns the Popen object (caller manages the process lifecycle).
        """
        if self.container_id is None:
            raise RuntimeError("Container not started")

        env_flags = self._build_docker_exec_env_flags()

        cmd = [
            "docker", "exec",
            "-u", "agent",
            "-w", "/workspace",
            *env_flags,
            self.container_id,
            "claude",
            "-p", prompt,
            "--output-format", "text",
            "--max-turns", str(max_turns),
        ]

        logger.info(f"Starting claude code in container {self.container_id[:12]}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        logger.info(f"Claude Code started in container, host PID={process.pid}")
        return process

    def exec_claude_code_sync(self, prompt: str, max_turns: int = 5,
                              timeout: int = 600) -> dict:
        """Run Claude Code inside container synchronously (for grading).

        Returns dict with stdout, stderr, exit_code.
        """
        if self.container_id is None:
            raise RuntimeError("Container not started")

        env_flags = self._build_docker_exec_env_flags()

        cmd = [
            "docker", "exec",
            "-u", "agent",
            "-w", "/workspace",
            *env_flags,
            self.container_id,
            "claude",
            "-p", prompt,
            "--output-format", "text",
            "--max-turns", str(max_turns),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {"exit_code": -1, "stdout": "", "stderr": "Timed out"}

    def copy_file_to(self, src_path: str, dest_path: str) -> None:
        """Copy a file from the host into the container."""
        if self.container is None:
            raise RuntimeError("Container not started")

        import tarfile
        import io

        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            tar.add(src_path, arcname=os.path.basename(dest_path))
        tar_stream.seek(0)

        dest_dir = os.path.dirname(dest_path) or "/"
        self.container.put_archive(dest_dir, tar_stream)

    def copy_file_from(self, container_path: str, host_path: str) -> None:
        """Copy a file from the container to the host."""
        if self.container is None:
            raise RuntimeError("Container not started")

        import tarfile
        import io

        bits, _ = self.container.get_archive(container_path)
        tar_stream = io.BytesIO()
        for chunk in bits:
            tar_stream.write(chunk)
        tar_stream.seek(0)

        os.makedirs(os.path.dirname(host_path), exist_ok=True)
        with tarfile.open(fileobj=tar_stream, mode="r") as tar:
            member = tar.getmembers()[0]
            f = tar.extractfile(member)
            if f:
                with open(host_path, "wb") as out:
                    out.write(f.read())

    def copy_dir_from(self, container_path: str, host_dir: str) -> None:
        """Copy a directory from the container to host."""
        if self.container is None:
            raise RuntimeError("Container not started")

        import tarfile
        import io

        os.makedirs(host_dir, exist_ok=True)
        bits, _ = self.container.get_archive(container_path)
        tar_stream = io.BytesIO()
        for chunk in bits:
            tar_stream.write(chunk)
        tar_stream.seek(0)
        with tarfile.open(fileobj=tar_stream, mode="r") as tar:
            tar.extractall(path=host_dir)

    def export_claude_traces(self, host_dest_dir: str) -> bool:
        """Export Claude Code traces from /home/agent/.claude/ to host."""
        try:
            self.copy_dir_from("/home/agent/.claude", host_dest_dir)
            logger.info(f"Exported Claude traces to {host_dest_dir}")
            return True
        except Exception as e:
            logger.warning(f"Failed to export Claude traces: {e}")
            return False

    def export_codex_traces(self, host_dest_dir: str) -> bool:
        """Export OpenAI Codex traces from /home/agent/.codex/ to host."""
        try:
            self.copy_dir_from("/home/agent/.codex", host_dest_dir)
            logger.info(f"Exported Codex traces to {host_dest_dir}")
            return True
        except Exception as e:
            logger.warning(f"Failed to export Codex traces: {e}")
            return False

    def export_opencode_traces(self, host_dest_dir: str) -> bool:
        """Export OpenCode traces from /home/agent/.local/share/opencode/ to host.

        Only copies useful artefacts (database, logs, session diffs, binary).
        Skips the bulky snapshot/ directory which contains git object blobs
        that can number in the thousands and provide no diagnostic value.
        """
        if self.container is None:
            return False
        os.makedirs(host_dest_dir, exist_ok=True)

        # Selective copy: grab each useful sub-path individually
        src_base = "/home/agent/.local/share/opencode"
        sub_paths = ["opencode.db", "opencode.db-wal", "opencode.db-shm",
                      "log", "storage", "bin"]
        ok = False
        for sp in sub_paths:
            try:
                self.copy_dir_from(f"{src_base}/{sp}", host_dest_dir)
                ok = True
            except Exception:
                pass  # sub-path may not exist
        if ok:
            logger.info(f"Exported OpenCode traces to {host_dest_dir}")
        else:
            logger.warning("No OpenCode trace files found to export")
        return ok

    def get_logs(self) -> str:
        """Get container logs."""
        if self.container is None:
            return ""
        return self.container.logs().decode("utf-8", errors="replace")

    def check_health(self) -> bool:
        """Check if the container is healthy."""
        if self.container is None:
            return False
        result = self.exec_command("python --version")
        return result["exit_code"] == 0

    def check_claude_health(self) -> bool:
        """Check if Claude Code CLI is available in the container."""
        if self.container is None:
            return False
        result = self.exec_command("claude --version")
        return result["exit_code"] == 0

    def check_codex_health(self) -> bool:
        """Check if OpenAI Codex CLI is available in the container."""
        if self.container is None:
            return False
        result = self.exec_command("codex --version")
        return result["exit_code"] == 0

    def check_opencode_health(self) -> bool:
        """Check if OpenCode CLI is available in the container."""
        if self.container is None:
            return False
        result = self.exec_command("opencode --version 2>/dev/null || opencode version 2>/dev/null")
        return result["exit_code"] == 0

    def stop(self) -> None:
        """Stop and remove the container. Fix workspace permissions first."""
        if self.container:
            # Make workspace world-writable so host user can clean up
            if self.workspace_dir:
                try:
                    self.exec_command("chmod -R 777 /workspace")
                except Exception:
                    pass
            try:
                self.container.stop(timeout=10)
                logger.info(f"Container {self.container.short_id} stopped.")
            except Exception as e:
                logger.warning(f"Error stopping container: {e}")
            try:
                self.container.remove(force=True)
                logger.info(f"Container {self.container.short_id} removed.")
            except Exception as e:
                logger.warning(f"Error removing container: {e}")
            self.container = None
            self.container_id = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
