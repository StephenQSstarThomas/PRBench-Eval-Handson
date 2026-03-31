#!/usr/bin/env python3
"""Generate comprehensive markdown report for PRBench-Eval results across all models."""
import json
from pathlib import Path
import yaml

DATA_DIR     = Path("/srv/home/shiqiu/PRBench-Eval/data/tasks")
RESULTS_BASE = Path("/srv/home/shiqiu/PRBench-Eval/results")
OUT_FILE     = Path("/srv/home/shiqiu/PRBench-Eval/results_report.md")

# Model configs: (result_dir_name, display_name, has_codex_tokens)
MODELS = [
    ("full_codex",    "Codex CLI",              True),
    ("full_glm",      "GLM (OpenCode)",          False),
    ("full_glm_4_7",  "GLM-4.7 (OpenCode)",     False),
    ("full_deepseek", "DeepSeek (OpenCode)",     False),
    ("full_kimi",     "Kimi (OpenCode)",         False),
]


def get_codex_tokens(task_res_dir: Path) -> dict | None:
    """Return cumulative token usage from the earliest codex session file."""
    session_files = sorted(task_res_dir.rglob(".codex/sessions/**/*.jsonl"))
    if not session_files:
        return None
    for jf in session_files:
        last_usage = None
        try:
            with open(jf) as fh:
                for line in fh:
                    d = json.loads(line)
                    if (d.get("type") == "event_msg"
                            and d.get("payload", {}).get("type") == "token_count"):
                        info = d["payload"].get("info") or {}
                        u = info.get("total_token_usage")
                        if u:
                            last_usage = u
        except Exception:
            continue
        if last_usage:
            return last_usage
    return None


def load_task_meta(data_dir: Path) -> dict:
    """Load task metadata from task.yaml. Returns {task_id: {title, author, year, doi}}."""
    meta = {}
    for td in sorted(data_dir.iterdir()):
        if not td.is_dir() or td.name.startswith("aaa") or td.name.startswith("bbb"):
            continue
        title, author, year, doi = "", "", "", ""
        task_yaml = td / "task.yaml"
        if task_yaml.exists():
            with open(task_yaml) as f:
                cfg = yaml.safe_load(f)
            p = cfg.get("paper", {})
            title  = p.get("title", "")
            author = p.get("author", "")
            year   = p.get("year", "")
            doi    = p.get("doi", "")
        meta[td.name] = {"title": title, "author": author, "year": year, "doi": doi}
    return meta


def load_model_results(model_dir: str, task_ids: list, has_tokens: bool) -> list:
    """Load eval results for all tasks under results/{model_dir}/."""
    res_dir = RESULTS_BASE / model_dir
    rows = []
    for task_id in task_ids:
        task_res_dir = res_dir / task_id
        report = task_res_dir / "eval_report.json"
        tokens = get_codex_tokens(task_res_dir) if has_tokens else None
        if not report.exists():
            rows.append({
                "task_id": task_id, "scores": {}, "overall": None,
                "time_s": None, "status": "NO RESULT", "tokens": tokens,
            })
            continue
        with open(report) as f:
            data = json.load(f)
        grading = data.get("grading", {})
        raw_scores = grading.get("scores", {})
        scores = {k: v.get("score") if isinstance(v, dict) else v
                  for k, v in raw_scores.items()}
        rows.append({
            "task_id": task_id, "scores": scores,
            "overall": grading.get("overall_score"),
            "time_s": data.get("time_used_seconds"),
            "status": "OK", "tokens": tokens,
        })
    return rows


# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt(v, fmt_str=".3f"):
    return f"{v:{fmt_str}}" if v is not None else "—"

def stats(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return {"mean": None, "min": None, "max": None, "n": 0}
    return {"mean": sum(v)/len(v), "min": min(v), "max": max(v), "n": len(v)}

def fmt_tokens(tok_dict):
    if not tok_dict:
        return "—"
    return f"{tok_dict.get('total_tokens', 0)/1e6:.2f}"

def sort_key(r):
    return (1, 0) if r["overall"] is None else (0, -r["overall"])

dim_short = {
    "methodology_understanding": "Methodology",
    "code_correctness":          "Code",
    "data_accuracy":             "Data Acc.",
    "completeness":              "Complete",
    "instruction_following":     "Instruction",
    "code_quality":              "Code Quality",
}


# ── Load canonical task list ──────────────────────────────────────────────────
task_meta = load_task_meta(DATA_DIR)
task_ids  = sorted(task_meta.keys())

lines = []
lines.append("# PRBench-Eval Comprehensive Results Report\n")
lines.append(
    "Models: **Codex CLI** (`full_codex`), **GLM + OpenCode** (`full_glm`), "
    "**GLM-4.7 + OpenCode** (`full_glm_4_7`), "
    "**DeepSeek + OpenCode** (`full_deepseek`), **Kimi + OpenCode** (`full_kimi`)\n"
)
lines.append(f"Canonical tasks: **{len(task_ids)}**  "
             "*(tasks from `data/tasks/`, excluding `aaa*`/`bbb*`)*\n")
lines.append("---\n")

all_model_rows = {}

# ── Per-model sections ────────────────────────────────────────────────────────
for model_key, model_name, has_tokens in MODELS:
    rows = load_model_results(model_key, task_ids, has_tokens)
    all_model_rows[model_key] = rows

    scored  = sorted([r for r in rows if r["overall"] is not None], key=sort_key)
    missing = [r for r in rows if r["overall"] is None]

    # Collect dimension names present in this model's results
    all_dims: list = []
    for r in rows:
        for k in r["scores"]:
            if k not in all_dims:
                all_dims.append(k)

    header_dims = [dim_short.get(d, d) for d in all_dims]
    token_col = " Tokens (M) |" if has_tokens else ""
    token_sep = "-----------:|" if has_tokens else ""

    lines.append(f"## {model_name} (`{model_key}`)\n")
    lines.append(
        f"Evaluated: **{len(scored)}** / {len(rows)} tasks "
        f"({len(missing)} missing)\n"
    )

    header = ("| # | Task | Year | " + " | ".join(header_dims)
              + " | **Overall** | Time (s) |" + token_col)
    sep    = ("|---|------|------|" + "|".join(["------"] * len(all_dims))
              + "|------------|----------:|" + token_sep)
    lines.append(header)
    lines.append(sep)

    for i, r in enumerate(scored, 1):
        dim_cells = " | ".join(fmt(r["scores"].get(d)) for d in all_dims)
        ov = fmt(r["overall"])
        t  = fmt(r["time_s"], ".0f") if r["time_s"] else "—"
        yr = task_meta.get(r["task_id"], {}).get("year", "")
        if has_tokens:
            tk = fmt_tokens(r.get("tokens"))
            lines.append(
                f"| {i} | `{r['task_id']}` | {yr} | {dim_cells} | **{ov}** | {t} | {tk} |"
            )
        else:
            lines.append(
                f"| {i} | `{r['task_id']}` | {yr} | {dim_cells} | **{ov}** | {t} |"
            )

    for r in missing:
        dim_cells = " | ".join("—" for _ in all_dims)
        yr = task_meta.get(r["task_id"], {}).get("year", "")
        if has_tokens:
            tk = fmt_tokens(r.get("tokens"))
            lines.append(
                f"| — | `{r['task_id']}` | {yr} | {dim_cells} | **N/A** | — | {tk} |"
            )
        else:
            lines.append(
                f"| — | `{r['task_id']}` | {yr} | {dim_cells} | **N/A** | — |"
            )

    lines.append("")

    # Per-model stats
    lines.append(f"### {model_name} — Summary Statistics\n")
    lines.append("*(Tasks with results only)*\n")
    stat_header = "| Metric | " + " | ".join(header_dims) + " | **Overall** |"
    stat_sep    = "|--------|" + "|".join(["-------"] * len(all_dims)) + "|------------|"
    lines.append(stat_header)
    lines.append(stat_sep)
    for label, key in [("Mean", "mean"), ("Min", "min"), ("Max", "max")]:
        cells = []
        for d in all_dims:
            s = stats([r["scores"].get(d) for r in scored])
            cells.append(fmt(s[key]))
        ov_s = stats([r["overall"] for r in scored])
        lines.append(f"| {label} | " + " | ".join(cells) + f" | **{fmt(ov_s[key])}** |")
    lines.append("")

    # Score distribution
    lines.append(f"### {model_name} — Score Distribution\n")
    bins = [
        (0.9, 1.01, "0.9 – 1.0"), (0.8, 0.9, "0.8 – 0.9"),
        (0.7, 0.8,  "0.7 – 0.8"), (0.6, 0.7, "0.6 – 0.7"),
        (0.5, 0.6,  "0.5 – 0.6"), (0.4, 0.5, "0.4 – 0.5"),
        (0.0, 0.4,  "< 0.4"),
    ]
    lines.append("| Score Range | Count | Tasks |")
    lines.append("|-------------|-------|-------|")
    for lo, hi, label in bins:
        matched = [r for r in scored if lo <= r["overall"] < hi]
        names = ", ".join(f"`{r['task_id']}`" for r in matched)
        lines.append(f"| {label} | {len(matched)} | {names or '—'} |")
    lines.append("")

    # Token stats (Codex only)
    if has_tokens:
        rows_with_tok = [r for r in rows if r.get("tokens")]
        lines.append(f"### {model_name} — Token Cost Statistics\n")
        lines.append("*(Work/task-solving session before eval runs)*\n")
        if rows_with_tok:
            lines.append(
                "| Task | Input (M) | Cached (M) | Output (M) | Reasoning (M) | **Total (M)** |"
            )
            lines.append(
                "|------|----------:|-----------:|-----------:|--------------:|--------------:|"
            )
            for r in sorted(rows_with_tok,
                            key=lambda x: x["tokens"].get("total_tokens", 0), reverse=True):
                u    = r["tokens"]
                inp  = u.get("input_tokens", 0) / 1e6
                cach = u.get("cached_input_tokens", 0) / 1e6
                out  = u.get("output_tokens", 0) / 1e6
                reas = u.get("reasoning_output_tokens", 0) / 1e6
                tot  = u.get("total_tokens", 0) / 1e6
                lines.append(
                    f"| `{r['task_id']}` | {inp:.2f} | {cach:.2f} | {out:.2f} | {reas:.2f} | **{tot:.2f}** |"
                )
            lines.append("")
            tots = [r["tokens"].get("total_tokens", 0) for r in rows_with_tok]
            inps = [r["tokens"].get("input_tokens", 0) for r in rows_with_tok]
            cacs = [r["tokens"].get("cached_input_tokens", 0) for r in rows_with_tok]
            outs = [r["tokens"].get("output_tokens", 0) for r in rows_with_tok]
            lines.append(
                f"**Tasks with token data: {len(rows_with_tok)} / {len(rows)}**  \n"
            )
            lines.append(f"- Total tokens across all tasks: {sum(tots)/1e6:.2f}M\n")
            lines.append(
                f"- Mean total: {sum(tots)/len(tots)/1e6:.2f}M  "
                f"(min {min(tots)/1e6:.2f}M, max {max(tots)/1e6:.2f}M)\n"
            )
            lines.append(
                f"- Mean input: {sum(inps)/len(inps)/1e6:.2f}M  "
                f"| Mean cached: {sum(cacs)/len(cacs)/1e6:.2f}M  "
                f"| Mean output: {sum(outs)/len(outs)/1e6:.2f}M\n"
            )
        else:
            lines.append("*No session token data found.*\n")
        lines.append("")

    lines.append("---\n")


# ── Cross-model comparison table ──────────────────────────────────────────────
lines.append("## Cross-Model Comparison\n")
lines.append(
    "Overall score per task across all models. "
    "**N/A** = not evaluated or no result.\n"
)

col_names = [name for _, name, _ in MODELS]
lines.append("| Task | Year | " + " | ".join(col_names) + " |")
lines.append("|------|------|" + "|".join(["-------"] * len(MODELS)) + "|")

lookup = {
    mk: {r["task_id"]: r["overall"] for r in all_model_rows[mk]}
    for mk, _, _ in MODELS
}

for task_id in task_ids:
    yr = task_meta[task_id]["year"]
    cells = []
    for mk, _, _ in MODELS:
        v = lookup[mk].get(task_id)
        cells.append(fmt(v) if v is not None else "N/A")
    lines.append(f"| `{task_id}` | {yr} | " + " | ".join(cells) + " |")

lines.append("")

lines.append("### Per-Model Overall Score Summary\n")
lines.append("| Model | Evaluated | Mean | Min | Max | ≥0.8 | ≥0.6 | <0.4 |")
lines.append("|-------|-----------|------|-----|-----|------|------|------|")
for mk, mname, _ in MODELS:
    vals = [r["overall"] for r in all_model_rows[mk] if r["overall"] is not None]
    if not vals:
        lines.append(f"| {mname} | 0 | — | — | — | — | — | — |")
        continue
    s   = stats(vals)
    n   = len(vals)
    n80 = sum(1 for v in vals if v >= 0.8)
    n60 = sum(1 for v in vals if v >= 0.6)
    n40 = sum(1 for v in vals if v < 0.4)
    lines.append(
        f"| {mname} | {n}/{len(task_ids)} | {fmt(s['mean'])} | {fmt(s['min'])} | {fmt(s['max'])} "
        f"| {n80} ({100*n80/n:.0f}%) | {n60} ({100*n60/n:.0f}%) | {n40} ({100*n40/n:.0f}%) |"
    )
lines.append("")

# ── Paper reference table ─────────────────────────────────────────────────────
lines.append("## Paper Reference\n")
lines.append("| Task | Title | Author(s) | Year | DOI |")
lines.append("|------|-------|-----------|------|-----|")
for task_id in task_ids:
    m = task_meta[task_id]
    short_author = m["author"].split(",")[0].split(" and ")[0] if m["author"] else "—"
    if m["author"] and ("," in m["author"] or " and " in m["author"]):
        short_author += " et al."
    lines.append(
        f"| `{task_id}` | {m['title']} | {short_author} | {m['year']} | {m['doi']} |"
    )
lines.append("")
lines.append("\n---\n*Generated by `gen_report.py`*\n")

OUT_FILE.write_text("\n".join(lines))
print(f"Written to {OUT_FILE}")
for mk, mname, _ in MODELS:
    rows = all_model_rows[mk]
    scored_vals = [r["overall"] for r in rows if r["overall"] is not None]
    s = stats(scored_vals)
    print(f"  {mname}: {len(scored_vals)}/{len(rows)} tasks, mean={fmt(s['mean'])}")
