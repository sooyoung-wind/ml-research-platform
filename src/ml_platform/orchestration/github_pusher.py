"""ML Research Platform — GitHub pusher module.

Takes a directory of generated code files, creates a new GitHub repository,
and pushes the code there.

Architecture:
  Generated Code Dir → git init → GitHub API (create repo) → git push → Repo URL

Uses:
  - GitHub REST API via httpx for async repo creation
  - Git CLI commands for init/add/commit/push (via asyncio subprocess)
  - Credentials from ~/.git-credentials or GITHUB_TOKEN env var
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GitHubPushConfig:
    """Configuration for the GitHub pusher.

    Attributes:
        username: GitHub username (auto-detected from git credentials if empty).
        token: GitHub personal access token or password.
        default_branch: Default branch name for new repositories.
        visibility: Repo visibility ("public" or "private").
        api_base_url: GitHub API base URL (change for GitHub Enterprise).
        git_credentials_path: Path to the git-credentials file.
        auto_readme: Whether to auto-generate a README if missing.
        git_user_name: Committer name (falls back to git config).
        git_user_email: Committer email (falls back to git config).
    """

    username: str = ""
    token: str = ""
    default_branch: str = "main"
    visibility: str = "public"
    api_base_url: str = "https://api.github.com"
    git_credentials_path: str = "~/.git-credentials"
    auto_readme: bool = True
    git_user_name: str = ""
    git_user_email: str = ""


@dataclass
class PushResult:
    """Result of a GitHub push operation.

    Attributes:
        success: Whether the push completed successfully.
        repo_url: URL of the GitHub repository.
        clone_url: Git clone URL for the repository.
        repo_name: Full repository name (e.g. ``user/repo-name``).
        commit_sha: SHA of the initial commit.
        error: Error message if the push failed.
        duration_seconds: Wall-clock time for the push operation.
        files_pushed: Number of files pushed to the repository.
    """

    success: bool = False
    repo_url: str = ""
    clone_url: str = ""
    repo_name: str = ""
    commit_sha: str = ""
    error: str = ""
    duration_seconds: float = 0.0
    files_pushed: int = 0


# ---------------------------------------------------------------------------
# Credential helpers
# ---------------------------------------------------------------------------


def _parse_git_credentials(path: str = "~/.git-credentials") -> tuple[str, str]:
    """Parse username and token from a git-credentials file.

    Only supports the HTTPS format: ``https://username:token@github.com``.

    Args:
        path: Filesystem path to the git-credentials file.

    Returns:
        A ``(username, token)`` tuple, or ``("", "")`` on failure.
    """
    cred_path = Path(path).expanduser()
    if not cred_path.is_file():
        return "", ""

    try:
        content = cred_path.read_text().strip()
        for line in content.splitlines():
            line = line.strip()
            if not line or "github.com" not in line:
                continue
            # https://username:token@github.com
            match = re.match(r"https://([^:***@]+)@github\.com", line)
            if match:
                return match.group(1), match.group(2)
    except OSError:
        pass

    return "", ""


async def _detect_git_user_name() -> str:
    """Detect git user.name from git config.

    Returns:
        The configured git user.name, or an empty string.
    """
    return (await _run_git(["config", "--global", "--get", "user.name"])).strip()


async def _detect_git_user_email() -> str:
    """Detect git user.email from git config.

    Returns:
        The configured git user.email, or an empty string.
    """
    return (await _run_git(["config", "--global", "--get", "user.email"])).strip()


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


async def _run_git(
    args: list[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> str:
    """Run a git command and return stdout.

    Args:
        args: Git sub-command and arguments (without the ``git`` prefix).
        cwd: Working directory for the command.
        env: Additional environment variables to merge.
        check: If True, raise ``RuntimeError`` on non-zero exit code.

    Returns:
        The command's standard output as a string.

    Raises:
        RuntimeError: If *check* is True and the command exits non-zero.
    """
    cmd = ["git"] + args
    merged_env = None
    if env:
        merged_env = {**os.environ, **env}

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=merged_env,
    )
    stdout, stderr = await proc.communicate()

    if check and proc.returncode != 0:
        err_msg = stderr.decode(errors="replace").strip()
        raise RuntimeError(
            f"git {' '.join(args)} failed (exit {proc.returncode}): {err_msg}"
        )

    return stdout.decode(errors="replace")


# ---------------------------------------------------------------------------
# Repo naming
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]")


def _make_repo_name(
    paper_id: str,
    paper_title: str | None = None,
    prefix: str = "ml-impl",
) -> str:
    """Generate a GitHub-friendly repository name.

    Examples:
        ``paper_id="2312.00752"`` → ``"ml-impl-2312-00752"``
        ``paper_id="2312.00752", title="Diffusion Models"`` →
        ``"ml-impl-diffusion-models-2312-00752"``

    Args:
        paper_id: Paper identifier.
        paper_title: Optional paper title used to make the name more descriptive.
        prefix: Prefix for the repository name.

    Returns:
        A slugified repository name string.
    """
    safe_id = paper_id.replace("/", "-").replace(".", "-")

    if paper_title:
        # Slugify the title
        slug = _SLUG_RE.sub("-", paper_title.lower().strip())
        slug = re.sub(r"-+", "-", slug).strip("-")
        # Truncate to keep repo name reasonable
        slug = slug[:60].rstrip("-")
        return f"{prefix}-{slug}-{safe_id}" if slug else f"{prefix}-{safe_id}"

    return f"{prefix}-{safe_id}"


def _make_description(
    paper_title: str | None = None,
    paper_id: str = "",
) -> str:
    """Generate a repository description.

    Args:
        paper_title: Optional paper title.
        paper_id: Optional paper identifier.

    Returns:
        A pipe-delimited description string.
    """
    parts: list[str] = []
    if paper_title:
        parts.append(f"Implementation of: {paper_title}")
    if paper_id:
        parts.append(f"Paper ID: {paper_id}")
    parts.append("Auto-generated by ML Research Platform")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class GitHubPusher:
    """Creates a GitHub repository and pushes generated code to it.

    Attributes:
        config: The active ``GitHubPushConfig`` used for this instance.

    Usage::

        pusher = GitHubPusher()
        result = await pusher.push(
            code_dir="/path/to/generated/code",
            paper_id="2312.00752",
            paper_title="Diffusion Models",
        )
        print(result.repo_url)
    """

    def __init__(self, push_config: GitHubPushConfig | None = None) -> None:
        """Initialize the GitHub pusher.

        Args:
            push_config: Optional push configuration. Defaults to a
                fresh ``GitHubPushConfig`` with auto-detected credentials.
        """
        self.config = push_config or GitHubPushConfig()
        self._resolve_credentials()

    def _resolve_credentials(self) -> None:
        """Resolve GitHub credentials from config, env, or git-credentials file."""
        # Try config first, then env var, then git-credentials file
        if not self.config.token:
            self.config.token = os.environ.get("GITHUB_TOKEN", "")

        if not self.config.username or not self.config.token:
            cred_user, cred_token = _parse_git_credentials(
                self.config.git_credentials_path
            )
            if not self.config.username:
                self.config.username = cred_user
            if not self.config.token:
                self.config.token = cred_token

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def push(
        self,
        code_dir: str | Path,
        *,
        paper_id: str,
        paper_title: str | None = None,
        repo_name: str | None = None,
        description: str | None = None,
        visibility: str | None = None,
        commit_message: str | None = None,
        files_to_include: Sequence[str] | None = None,
    ) -> PushResult:
        """Push a directory of generated code to a new GitHub repository.

        Args:
            code_dir: Local directory containing generated code files.
            paper_id: Paper identifier (used for repo naming).
            paper_title: Optional paper title (used for naming/description).
            repo_name: Override the auto-generated repo name.
            description: Override the auto-generated description.
            visibility: "public" or "private" (overrides config).
            commit_message: Custom commit message.
            files_to_include: If provided, only push these relative paths
                from code_dir. If None, push everything.

        Returns:
            ``PushResult`` with the repo URL and status.
        """
        start_time = time.time()
        code_path = Path(code_dir).resolve()

        if not code_path.is_dir():
            return PushResult(
                success=False,
                error=f"Code directory does not exist: {code_path}",
                duration_seconds=time.time() - start_time,
            )

        name = repo_name or _make_repo_name(paper_id, paper_title)
        desc = description or _make_description(paper_title, paper_id)
        vis = visibility or self.config.visibility

        user_name, user_email = await self._resolve_git_credentials()

        try:
            repo_info = await self._create_repo(name=name, description=desc, visibility=vis)
            clone_url = repo_info["clone_url"]
            repo_url = repo_info["html_url"]
            full_name = repo_info["full_name"]
            logger.info("Created remote repo: %s", repo_url)

            with tempfile.TemporaryDirectory(prefix="gh-push-") as tmpdir:
                work_dir = await self._prepare_local_repo(
                    code_path, Path(tmpdir) / "repo", files_to_include,
                    paper_id, paper_title, desc, user_name, user_email, commit_message,
                )
                await self._push_to_remote(work_dir, clone_url)
                sha = (await _run_git(["rev-parse", "HEAD"], cwd=work_dir)).strip()

            file_count = sum(1 for p in code_path.rglob("*") if p.is_file())
            logger.info("Pushed %d files to %s", file_count, repo_url)

            return PushResult(
                success=True,
                repo_url=repo_url,
                clone_url=clone_url,
                repo_name=full_name,
                commit_sha=sha,
                duration_seconds=time.time() - start_time,
                files_pushed=file_count,
            )

        except Exception as exc:
            logger.error("GitHub push failed: %s", exc)
            return PushResult(
                success=False,
                error=str(exc),
                duration_seconds=time.time() - start_time,
            )

    async def _prepare_local_repo(
        self,
        code_path: Path,
        work_dir: Path,
        files_to_include: Sequence[str] | None,
        paper_id: str,
        paper_title: str | None,
        description: str,
        user_name: str,
        user_email: str,
        commit_message: str | None,
    ) -> Path:
        """Prepare the local git repo by copying files and running git init/commit.

        Args:
            code_path: Source directory with generated code.
            work_dir: Destination working directory for the repo.
            files_to_include: Optional subset of files to include.
            paper_id: Paper identifier for README.
            paper_title: Optional paper title for README.
            description: Repository description for README.
            user_name: Git committer name.
            user_email: Git committer email.
            commit_message: Custom commit message.

        Returns:
            The work_dir Path with the prepared repo.
        """
        if files_to_include:
            work_dir.mkdir(parents=True)
            for rel_path in files_to_include:
                src = code_path / rel_path
                dst = work_dir / rel_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src.is_dir():
                    shutil.copytree(str(src), str(dst))
                else:
                    shutil.copy2(str(src), str(dst))
        else:
            shutil.copytree(str(code_path), str(work_dir))

        await self._maybe_add_readme(work_dir, paper_id, paper_title, description)
        await self._maybe_add_gitignore(work_dir)

        await _run_git(["init", "-b", self.config.default_branch], cwd=work_dir)
        await _run_git(["config", "user.name", user_name], cwd=work_dir)
        await _run_git(["config", "user.email", user_email], cwd=work_dir)
        await _run_git(["add", "."], cwd=work_dir)

        msg = commit_message or f"Initial commit: auto-generated implementation for {paper_id}"
        await _run_git(["commit", "-m", msg], cwd=work_dir)
        return work_dir

    async def _push_to_remote(self, work_dir: Path, clone_url: str) -> None:
        """Add remote origin and push to GitHub.

        Args:
            work_dir: Local git working directory.
            clone_url: GitHub clone URL for the remote.
        """
        auth_url = self._build_auth_url(clone_url)
        await _run_git(["remote", "add", "origin", auth_url], cwd=work_dir)
        await _run_git(
            ["push", "-u", "origin", self.config.default_branch],
            cwd=work_dir,
            env={"GIT_TERMINAL_PROMPT": "0"},
        )

    async def push_and_update_paper(
        self,
        code_dir: str | Path,
        paper_id: str,
        source: str,
        *,
        paper_title: str | None = None,
        visibility: str | None = None,
    ) -> PushResult:
        """Push code and update the paper record in the database.

        Convenience method that also sets the paper status to PUSHED
        and stores the GitHub URL.

        Args:
            code_dir: Local directory with generated code.
            paper_id: Paper identifier.
            source: Paper source identifier.
            paper_title: Optional paper title.
            visibility: "public" or "private".

        Returns:
            ``PushResult`` with the repo URL and status.
        """
        from ml_platform.models import PaperSource, ProcessingStatus
        from ml_platform.db import PapersDB

        result = await self.push(
            code_dir=code_dir,
            paper_id=paper_id,
            paper_title=paper_title,
            visibility=visibility,
        )

        if result.success:
            try:
                db = PapersDB()
                ps = PaperSource(source)
                db.update_status(paper_id, ps, ProcessingStatus.PUSHED)
                # Update code_url on the paper record
                paper = db.get_paper(paper_id, ps)
                if paper:
                    paper.code_url = result.repo_url
                    paper.status = ProcessingStatus.PUSHED
                    paper.updated_at = datetime.now()  # type: ignore[assignment]
                    db.upsert_paper(paper)
                logger.info("Updated paper %s status to PUSHED", paper_id)
            except Exception as db_exc:
                logger.warning("Failed to update paper in DB: %s", db_exc)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_auth_url(self, clone_url: str) -> str:
        """Inject credentials into a clone URL for pushing.

        Turns ``https://github.com/user/repo.git`` into
        ``https://user:token@github.com/user/repo.git``.

        Args:
            clone_url: The HTTPS clone URL.

        Returns:
            The clone URL with embedded credentials, or the original
            URL if credentials are not configured.
        """
        if not self.config.username or not self.config.token:
            return clone_url
        return clone_url.replace(
            "https://",
            f"https://{self.config.username}:{self.config.token}@",
        )

    async def _create_repo(
        self,
        *,
        name: str,
        description: str,
        visibility: str,
    ) -> dict:
        """Create a GitHub repository via the REST API.

        Args:
            name: Repository name.
            description: Repository description.
            visibility: "public" or "private".

        Returns:
            The JSON response dict from GitHub.

        Raises:
            RuntimeError: If no GitHub token is configured or the API
                returns an error.
        """
        if not self.config.token:
            raise RuntimeError(
                "No GitHub token available. Set GITHUB_TOKEN env var or "
                "configure ~/.git-credentials."
            )

        url = f"{self.config.api_base_url}/user/repos"
        headers = {
            "Authorization": f"token {self.config.token}",
            "Accept": "application/vnd.github.v3+json",
        }
        payload: dict = {
            "name": name,
            "description": description,
            "private": visibility == "private",
            "auto_init": False,  # We push our own content
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload, headers=headers)

        if resp.status_code == 201:
            return resp.json()

        if resp.status_code == 422:
            # Repo may already exist — try to get it
            detail = resp.json().get("message", "Unknown error")
            raise RuntimeError(f"Repository creation failed (422): {detail}")

        resp.raise_for_status()
        return resp.json()  # fallback

    async def _maybe_add_readme(
        self,
        work_dir: Path,
        paper_id: str,
        paper_title: str | None,
        description: str,
    ) -> None:
        """Add a README.md if one doesn't already exist.

        Args:
            work_dir: Repository working directory.
            paper_id: Paper identifier.
            paper_title: Optional paper title.
            description: Repository description for the README.
        """
        readme_path = work_dir / "README.md"
        if readme_path.exists():
            return

        title_line = paper_title or f"ML Implementation — {paper_id}"
        lines = [
            f"# {title_line}",
            "",
            f"{description}",
            "",
            f"**Paper ID:** `{paper_id}`",
            "",
        ]

        if paper_title:
            lines.extend([
                "## Overview",
                "",
                f"This repository contains an auto-generated implementation of "
                f"**{paper_title}** (paper ID: `{paper_id}`).",
                "",
                "Generated by the [ML Research Platform](https://github.com/sooyoung-wind/ml-research-platform).",
                "",
            ])

        lines.append("---\n*Auto-generated code — review before using in production.*\n")

        readme_path.write_text("\n".join(lines))
        logger.debug("Generated README.md")

    async def _maybe_add_gitignore(self, work_dir: Path) -> None:
        """Add a Python .gitignore if one doesn't already exist.

        Args:
            work_dir: Repository working directory.
        """
        gitignore_path = work_dir / ".gitignore"
        if gitignore_path.exists():
            return

        content = (
            "# Python\n"
            "__pycache__/\n"
            "*.py[cod]\n"
            "*.egg-info/\n"
            "dist/\n"
            "build/\n"
            ".eggs/\n"
            "\n"
            "# Virtual environments\n"
            ".venv/\n"
            "venv/\n"
            "\n"
            "# IDE\n"
            ".idea/\n"
            ".vscode/\n"
            "*.swp\n"
            "\n"
            "# OS\n"
            ".DS_Store\n"
            "Thumbs.db\n"
            "\n"
            "# Data / models\n"
            "*.pth\n"
            "*.ckpt\n"
            "*.pt\n"
            "data/\n"
            "wandb/\n"
        )
        gitignore_path.write_text(content)
        logger.debug("Generated .gitignore")


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


async def push_to_github(
    code_dir: str | Path,
    *,
    paper_id: str,
    paper_title: str | None = None,
    repo_name: str | None = None,
    visibility: str = "public",
    commit_message: str | None = None,
    config: GitHubPushConfig | None = None,
) -> PushResult:
    """One-shot function to push generated code to a new GitHub repository.

    Args:
        code_dir: Local directory with generated code.
        paper_id: Paper identifier.
        paper_title: Optional paper title.
        repo_name: Override auto-generated repo name.
        visibility: "public" or "private".
        commit_message: Custom commit message.
        config: Optional push configuration.

    Returns:
        ``PushResult`` with repo URL and status.
    """
    pusher = GitHubPusher(config)
    return await pusher.push(
        code_dir=code_dir,
        paper_id=paper_id,
        paper_title=paper_title,
        repo_name=repo_name,
        visibility=visibility,
        commit_message=commit_message,
    )
