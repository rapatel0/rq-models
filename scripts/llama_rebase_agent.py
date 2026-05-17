#!/usr/bin/env python3
"""Launch the standard llama.cpp upstream rebase agent workflow."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_REPO = "https://github.com/ggml-org/llama.cpp.git"
DEFAULT_RELEASE_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    capture: bool = True,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        text=True,
        input=input_text,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def repo_root() -> Path:
    out = run(["git", "rev-parse", "--show-toplevel"]).stdout.strip()
    return Path(out)


def parse_dockerfile(root: Path) -> tuple[str, str, str]:
    dockerfile = root / "docker" / "Dockerfile"
    text = dockerfile.read_text(encoding="utf-8")

    def arg(name: str, default: str = "") -> str:
        match = re.search(rf"^ARG\s+{re.escape(name)}=(\S+)\s*$", text, re.MULTILINE)
        return match.group(1) if match else default

    return (
        arg("LLAMA_CPP_REPO", DEFAULT_REPO),
        arg("LLAMA_CPP_REF"),
        arg("ROTORQUANT_PATCH"),
    )


def latest_release_tag(api_url: str) -> str:
    req = urllib.request.Request(
        api_url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "rq-models-llama-rebase-agent",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.load(resp)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to query latest llama.cpp release: {exc}") from exc

    tag = payload.get("tag_name")
    if not tag:
        raise RuntimeError("GitHub release response did not include tag_name")
    return str(tag)


def tag_commit(repo_url: str, ref: str) -> str:
    candidates = [f"refs/tags/{ref}^{{}}", f"refs/tags/{ref}", ref]
    for candidate in candidates:
        proc = run(["git", "ls-remote", repo_url, candidate], check=False)
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.split()[0]
    return "unknown"


def b_tag_number(tag: str) -> int | None:
    match = re.fullmatch(r"b(\d+)", tag)
    return int(match.group(1)) if match else None


def is_newer(candidate: str, current: str) -> bool:
    candidate_num = b_tag_number(candidate)
    current_num = b_tag_number(current)
    if candidate_num is not None and current_num is not None:
        return candidate_num > current_num
    return candidate != current


def git_is_dirty(root: Path) -> bool:
    return bool(run(["git", "status", "--porcelain"], cwd=root).stdout.strip())


def build_prompt(
    *,
    current_ref: str,
    target_ref: str,
    target_commit: str,
    patch_path: str,
    repo_url: str,
    push: bool,
) -> str:
    push_line = (
        "Commit and push the result to origin/main if all validation passes."
        if push
        else "Commit the result locally if all validation passes, but do not push."
    )
    return f"""You are updating rq-models' standard llama.cpp rebase workflow.

Goal:
Rebase the local RotorQuant patch from upstream llama.cpp {current_ref} to latest stable {target_ref}.

Inputs:
- Upstream repo: {repo_url}
- Target stable tag: {target_ref}
- Target commit: {target_commit}
- Current RotorQuant patch: {patch_path}
- Dockerfile pin to update: docker/Dockerfile ARG LLAMA_CPP_REF

Required workflow:
1. Start from a clean git worktree. Do not discard user changes.
2. Clone or fetch official upstream llama.cpp at {target_ref} into /tmp.
3. Apply the existing RotorQuant patch to understand the current patch surface.
4. Rebase or forward-port only the RotorQuant KV cache work onto upstream {target_ref}.
   Keep upstream's official MTP implementation and current `--spec-type draft-mtp` path.
   Do not revive the older fork-specific `--spec-type mtp` implementation except for runtime compatibility docs.
5. Regenerate docker/patches/llama-{target_ref}-rotorquant.patch.gz from the rebased source.
6. Update docker/Dockerfile to pin LLAMA_CPP_REF={target_ref} and the new patch filename.
7. Update README and docs/benchmarks with the source tag, target commit, validation results, and next homelab benchmark gate.
8. Validate:
   - gzip -dc docker/patches/llama-{target_ref}-rotorquant.patch.gz | git -C <fresh llama.cpp {target_ref} checkout> apply --check
   - docker buildx build --check -f docker/Dockerfile .
   - bash -n docker/entrypoint.sh
   - python3 -m py_compile scripts/mtp_probe.py scripts/llama_rebase_agent.py
   - docker compose --profile qwen36-27b-mtp-speed config
   - helm template rq ./k8s with qwen3.6-27b-mtp, q4_0 KV, draft 4, p-min 0.75, ubatch 32
   - local llama-server build if feasible on the current host
   - llama-server --help must advertise both `draft-mtp` and RotorQuant KV types:
     tbq3_0, tbq4_0, planar3_0, iso3_0, planar4_0, iso4_0
   - if the Qwen3.6 MTP GGUF is already cached locally, run scripts/mtp_probe.py against a local draft-mtp server
   - git diff --check
9. {push_line}

Final response:
- State the upstream tag and commit used.
- State whether the patch applies cleanly to a fresh upstream checkout.
- State which validations passed and which were skipped.
- State the commit hash, and push result if pushing was requested.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="only report current/latest state")
    parser.add_argument("--run-agent", action="store_true", help="launch the Codex rebase agent")
    parser.add_argument("--force", action="store_true", help="run even when current is latest")
    parser.add_argument("--target-ref", help="override latest release tag")
    parser.add_argument("--allow-dirty", action="store_true", help="allow agent launch with dirty git state")
    parser.add_argument("--push", action="store_true", help="tell the agent to push after a passing commit")
    parser.add_argument(
        "--output-dir",
        default=".agent-runs/llama-rebase",
        help="directory for generated prompts and final agent messages",
    )
    parser.add_argument("--codex-bin", default=os.environ.get("CODEX_BIN", "codex"))
    parser.add_argument("--model", default=os.environ.get("LLAMA_REBASE_CODEX_MODEL", ""))
    parser.add_argument("--release-api", default=DEFAULT_RELEASE_API)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    repo_url, current_ref, patch_name = parse_dockerfile(root)

    target_ref = args.target_ref or os.environ.get("LLAMA_REBASE_TARGET_REF")
    if not target_ref:
        target_ref = latest_release_tag(args.release_api)
    target_commit = tag_commit(repo_url, target_ref)
    current_commit = tag_commit(repo_url, current_ref) if current_ref else "unknown"

    status = {
        "repo": repo_url,
        "current_ref": current_ref,
        "current_commit": current_commit,
        "target_ref": target_ref,
        "target_commit": target_commit,
        "patch": patch_name,
        "needs_rebase": bool(args.force or is_newer(target_ref, current_ref)),
    }

    print(json.dumps(status, indent=2, sort_keys=True))

    if args.check:
        return 1 if status["needs_rebase"] else 0

    if not args.run_agent:
        return 0

    if not status["needs_rebase"]:
        print("llama.cpp pin is already current; not launching agent.")
        return 0

    if git_is_dirty(root) and not args.allow_dirty:
        print(
            "Refusing to launch rebase agent with a dirty git worktree. "
            "Commit/stash changes or pass --allow-dirty.",
            file=sys.stderr,
        )
        return 2

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = (root / args.output_dir / timestamp).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    patch_path = f"docker/patches/{patch_name}" if patch_name else "docker/patches/<current patch>"
    push = args.push or os.environ.get("LLAMA_REBASE_PUSH") == "1"
    prompt = build_prompt(
        current_ref=current_ref,
        target_ref=target_ref,
        target_commit=target_commit,
        patch_path=patch_path,
        repo_url=repo_url,
        push=push,
    )
    prompt_file = out_dir / "prompt.md"
    final_file = out_dir / "final.md"
    prompt_file.write_text(prompt, encoding="utf-8")

    cmd = [
        args.codex_bin,
        "--ask-for-approval",
        "never",
        "--sandbox",
        "danger-full-access",
        "--search",
        "exec",
        "--cd",
        str(root),
        "--output-last-message",
        str(final_file),
        "-",
    ]
    if args.model:
        cmd[1:1] = ["--model", args.model]

    print(f"Launching Codex rebase agent. Prompt: {prompt_file}")
    print(f"Final message will be written to: {final_file}")
    return run(cmd, cwd=root, check=False, capture=False, input_text=prompt).returncode


if __name__ == "__main__":
    raise SystemExit(main())
