"""Centralized environment variable resolution for agent types.

Each agent type (claude, codex, opencode) has its own namespaced env vars:
  Claude Code:  CC_API_KEY, CC_BASE_URL, CC_MODEL, CC_SMALL_FAST_MODEL
  Codex:        CODEX_API_KEY, CODEX_BASE_URL, CODEX_MODEL, CODEX_SMALL_FAST_MODEL
  OpenCode:     OPENCODE_API_KEY, OPENCODE_BASE_URL, OPENCODE_MODEL

Falls back to standard env vars (ANTHROPIC_*, OPENAI_*) when agent-specific
vars are not set.
"""

import os


def _get_env(*names: str) -> str:
    """Return the first non-empty env var value from the given names."""
    for name in names:
        val = os.environ.get(name, "").strip()
        if val:
            return val
    return ""


def _maybe_append_v1(url: str) -> str:
    """Append /v1 to an OpenAI-style base URL when a known proxy requires it.

    Currently only yunwu.ai is known to need /v1 appended.  Standard
    OpenAI endpoints (api.openai.com) already include /v1 or don't need it.
    """
    if not url:
        return url
    url = url.rstrip("/")
    if "yunwu.ai" in url and not url.endswith("/v1"):
        return url + "/v1"
    return url


# ── per-agent resolvers ──────────────────────────────────────────────

def resolve_claude_env() -> dict[str, str]:
    """Resolve env vars for Claude Code CLI → ANTHROPIC_* vars."""
    env: dict[str, str] = {}

    api_key = _get_env("CC_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
    if api_key:
        env["ANTHROPIC_API_KEY"] = api_key

    base_url = _get_env("CC_BASE_URL", "ANTHROPIC_BASE_URL")
    if base_url:
        env["ANTHROPIC_BASE_URL"] = base_url

    model = _get_env("CC_MODEL", "ANTHROPIC_MODEL")
    if model:
        env["ANTHROPIC_MODEL"] = model

    small = _get_env("CC_SMALL_FAST_MODEL", "ANTHROPIC_SMALL_FAST_MODEL")
    if small:
        env["ANTHROPIC_SMALL_FAST_MODEL"] = small

    return env


def resolve_codex_env() -> dict[str, str]:
    """Resolve env vars for OpenAI Codex CLI → OPENAI_* vars."""
    env: dict[str, str] = {}

    api_key = _get_env("CODEX_API_KEY", "OPENAI_API_KEY")
    if api_key:
        env["OPENAI_API_KEY"] = api_key
        env["CODEX_API_KEY"] = api_key        # CI alias expected by codex

    base_url = _get_env("CODEX_BASE_URL", "OPENAI_BASE_URL")
    if base_url:
        env["OPENAI_BASE_URL"] = _maybe_append_v1(base_url)

    model = _get_env("CODEX_MODEL", "OPENAI_MODEL")
    if model:
        env["OPENAI_MODEL"] = model

    small = _get_env("CODEX_SMALL_FAST_MODEL", "OPENAI_SMALL_FAST_MODEL")
    if small:
        env["OPENAI_SMALL_FAST_MODEL"] = small

    return env


def resolve_opencode_env() -> dict[str, str]:
    """Resolve env vars for OpenCode CLI.

    OpenCode is provider-agnostic.  The model format is ``provider/model-name``
    (e.g. ``openai/gpt-5.3-codex``).  Based on the provider prefix we set the
    appropriate standard API key / base-URL env vars that OpenCode reads
    automatically.

    When a custom OpenAI-compatible base URL is provided, the model is
    rewritten from ``openai/<name>`` to ``openai_compat/<name>`` so it
    matches the custom provider registered in opencode.json (the built-in
    "openai" provider rejects non-OpenAI model IDs).
    """
    env: dict[str, str] = {}

    model = _get_env("OPENCODE_MODEL")
    api_key = _get_env("OPENCODE_API_KEY")
    base_url = _get_env("OPENCODE_BASE_URL")

    is_anthropic = model.startswith("anthropic/")

    if is_anthropic:
        key = api_key or _get_env("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
        if key:
            env["ANTHROPIC_API_KEY"] = key
        url = base_url or _get_env("ANTHROPIC_BASE_URL")
        if url:
            env["ANTHROPIC_BASE_URL"] = url
    else:
        # Default: OpenAI-compatible provider
        key = api_key or _get_env("OPENAI_API_KEY")
        if key:
            env["OPENAI_API_KEY"] = key
        url = base_url or _get_env("OPENAI_BASE_URL")
        if url:
            url = _maybe_append_v1(url)
            env["OPENAI_BASE_URL"] = url
            # Rewrite model prefix from "openai/" to "openai_compat/" so it
            # matches the custom provider in opencode.json config.
            if model.startswith("openai/"):
                model = "openai_compat/" + model.split("/", 1)[1]

    if model:
        env["OPENCODE_MODEL"] = model

    return env


# ── public helpers ────────────────────────────────────────────────────

def resolve_env(agent_type: str) -> dict[str, str]:
    """Resolve environment variables for the given agent type."""
    if agent_type == "codex":
        return resolve_codex_env()
    elif agent_type == "opencode":
        return resolve_opencode_env()
    else:
        return resolve_claude_env()


def build_docker_exec_env_flags(agent_type: str) -> list[str]:
    """Build ``-e`` flags for ``docker exec`` with resolved env vars + HOME."""
    flags = ["-e", "HOME=/home/agent"]
    for key, val in resolve_env(agent_type).items():
        flags.extend(["-e", f"{key}={val}"])
    return flags
