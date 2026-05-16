"""ML Research Platform — DeepCode integration for paper-to-code generation.

Wraps DeepCode (deepcode-hku) multi-agent pipeline to generate code repos
from research papers within our discovery → processing → codegen workflow.

Architecture:
  Paper PDF → DeepCode Pipeline → Generated Code Repo → Optional GitHub Push

DeepCode internally handles:
  1. PDF → Markdown conversion (PyPDF2)
  2. Document segmentation (semantic chunks)
  3. Code planning (YAML plan)
  4. Reference code mining (top-5 GitHub repos)
  5. Code implementation (iterative, up to 800 rounds)
  6. Self-debugging with loop detection
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ml_platform.config import config


@dataclass
class CodeGenResult:
    """Result of a code generation run."""

    success: bool = False
    paper_id: str = ""
    paper_title: str = ""
    output_dir: str = ""  # Directory with generated code
    files_generated: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    error: str = ""
    pipeline_mode: str = "optimized"  # "optimized" | "comprehensive"
    model_used: str = ""


@dataclass
class DeepCodeConfig:
    """Configuration for DeepCode pipeline."""

    # LLM provider: "openai" | "anthropic" | "google"
    llm_provider: str = "openai"
    # Model name (provider-specific)
    model_name: str = "gpt-4o"
    # Planning model (can differ from implementation model)
    planning_model: str = ""
    # Enable CodeRAG reference mining (comprehensive mode)
    enable_indexing: bool = False
    # Enable document segmentation for long papers
    enable_segmentation: bool = True
    # Segmentation threshold (characters)
    segmentation_threshold: int = 50000
    # Max implementation iterations
    max_iterations: int = 100
    # Working directory for generated code
    output_base_dir: str = ""
    # API keys (read from env vars if not set)
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""


class DeepCodeRunner:
    """Wrapper around DeepCode's multi-agent pipeline.

    Provides a simple interface:
        runner = DeepCodeRunner(config)
        result = await runner.generate(paper_path, paper_id="2312.00752")

    Supports two modes:
        - optimized: Fast mode, no reference mining, lower cost
        - comprehensive: Full pipeline with CodeRAG, higher quality
    """

    def __init__(self, dc_config: DeepCodeConfig | None = None) -> None:
        self.config = dc_config or DeepCodeConfig()
        self._setup_dirs()
        self._write_secrets()

    def _setup_dirs(self) -> None:
        """Set up working directories."""
        base = self.config.output_base_dir or os.path.join(
            os.getcwd(), "data", "codegen"
        )
        self.output_dir = Path(base)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # DeepCode config directory
        self.config_dir = self.output_dir / ".deepcode"
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def _write_secrets(self) -> None:
        """Write DeepCode's required config YAML files."""
        import yaml  # noqa: delayed import

        # API keys from config or environment
        keys = {
            "openai": self.config.openai_api_key or os.environ.get("OPENAI_API_KEY", ""),
            "anthropic": self.config.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
            "google": self.config.google_api_key or os.environ.get("GOOGLE_API_KEY", "")
            or os.environ.get("GEMINI_API_KEY", ""),
        }

        # Write secrets file
        secrets = {}
        if keys["openai"]:
            secrets["openai"] = {"api_key": keys["openai"]}
        if keys["anthropic"]:
            secrets["anthropic"] = {"api_key": keys["anthropic"]}
        if keys["google"]:
            secrets["google"] = {"api_key": keys["google"]}

        secrets_path = self.config_dir / "mcp_agent.secrets.yaml"
        with open(secrets_path, "w") as f:
            yaml.dump(secrets, f, default_flow_style=False)

        # Write main config
        main_config = {
            "llm_provider": self.config.llm_provider,
        }
        config_path = self.config_dir / "mcp_agent.config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(main_config, f, default_flow_style=False)

    async def generate(
        self,
        paper_path: str,
        *,
        paper_id: str = "",
        paper_title: str = "",
        mode: str = "optimized",
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> CodeGenResult:
        """Generate code from a research paper.

        Args:
            paper_path: Path to the paper PDF or Markdown file.
            paper_id: Identifier for the paper (e.g., arXiv ID).
            paper_title: Title of the paper (for naming output).
            mode: "optimized" (fast) or "comprehensive" (full pipeline).
            progress_callback: Optional callback(progress_pct, message).

        Returns:
            CodeGenResult with paths to generated code.
        """
        start_time = time.time()
        result = CodeGenResult(
            paper_id=paper_id,
            paper_title=paper_title,
            pipeline_mode=mode,
        )

        if not os.path.exists(paper_path):
            result.error = f"Paper file not found: {paper_path}"
            result.duration_seconds = time.time() - start_time
            return result

        # Prepare output directory
        safe_name = paper_id.replace("/", "_").replace(".", "_") if paper_id else "unknown"
        paper_output = self.output_dir / safe_name
        paper_output.mkdir(parents=True, exist_ok=True)

        try:
            generated = await self._run_deepcode(
                paper_path=paper_path,
                output_dir=str(paper_output),
                mode=mode,
                progress_callback=progress_callback,
            )

            if generated:
                result.success = True
                result.output_dir = str(paper_output)
                result.files_generated = self._list_generated_files(paper_output)
                result.model_used = self.config.model_name or self.config.llm_provider
            else:
                result.error = "DeepCode pipeline returned no output"

        except Exception as e:
            result.error = f"DeepCode error: {e}"

        result.duration_seconds = time.time() - start_time
        return result

    async def _run_deepcode(
        self,
        paper_path: str,
        output_dir: str,
        mode: str,
        progress_callback: Callable | None = None,
    ) -> bool:
        """Run DeepCode's multi-agent pipeline programmatically.

        We call DeepCode's orchestration engine directly, changing CWD
        to a temp directory so it creates its deepcode_lab/ output there.
        """
        import yaml

        # Create a temp working directory for DeepCode
        # (DeepCode creates deepcode_lab/ in CWD)
        with tempfile.TemporaryDirectory(prefix="deepcode_") as tmpdir:
            # Copy paper to temp dir
            paper_name = Path(paper_path).name
            tmp_paper = os.path.join(tmpdir, paper_name)
            shutil.copy2(paper_path, tmp_paper)

            # Write config files in the temp dir
            # (DeepCode looks for mcp_agent.secrets.yaml in CWD)
            keys = {
                "openai": self.config.openai_api_key or os.environ.get("OPENAI_API_KEY", ""),
                "anthropic": self.config.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
                "google": self.config.google_api_key or os.environ.get("GOOGLE_API_KEY", "")
                or os.environ.get("GEMINI_API_KEY", ""),
            }

            secrets = {}
            for provider, key in keys.items():
                if key:
                    secrets[provider] = {"api_key": key}

            with open(os.path.join(tmpdir, "mcp_agent.secrets.yaml"), "w") as f:
                yaml.dump(secrets, f, default_flow_style=False)

            main_config = {"llm_provider": self.config.llm_provider}
            with open(os.path.join(tmpdir, "mcp_agent.config.yaml"), "w") as f:
                yaml.dump(main_config, f, default_flow_style=False)

            # Save current dir
            original_cwd = os.getcwd()

            try:
                os.chdir(tmpdir)

                # Import and run DeepCode's orchestration
                from workflows.agent_orchestration_engine import (
                    execute_multi_agent_research_pipeline,
                )
                from utils.llm_utils import get_preferred_llm_class

                # Simple logger
                import logging
                logger = logging.getLogger("deepcode")
                logger.setLevel(logging.INFO)

                enable_indexing = mode == "comprehensive"

                pipeline_result = await execute_multi_agent_research_pipeline(
                    input_source=tmp_paper,
                    logger=logger,
                    progress_callback=progress_callback,
                    enable_indexing=enable_indexing,
                )

                # Copy generated files from deepcode_lab/ to output_dir
                lab_dir = os.path.join(tmpdir, "deepcode_lab")
                if os.path.isdir(lab_dir):
                    # Copy all generated files
                    for item in os.listdir(lab_dir):
                        src = os.path.join(lab_dir, item)
                        dst = os.path.join(output_dir, item)
                        if os.path.isdir(src):
                            if os.path.exists(dst):
                                shutil.rmtree(dst)
                            shutil.copytree(src, dst)
                        else:
                            shutil.copy2(src, dst)
                    return True

                return False

            finally:
                os.chdir(original_cwd)

    @staticmethod
    def _list_generated_files(directory: Path) -> list[str]:
        """List all generated code files."""
        extensions = {".py", ".js", ".ts", ".jsx", ".tsx", ".sh", ".yaml", ".yml",
                      ".json", ".md", ".txt", ".toml", ".cfg", ".ini", ".ipynb"}
        files = []
        for root, _, filenames in os.walk(directory):
            for fn in filenames:
                if Path(fn).suffix in extensions:
                    rel = os.path.relpath(os.path.join(root, fn), directory)
                    files.append(rel)
        return sorted(files)


async def generate_code(
    paper_path: str,
    *,
    paper_id: str = "",
    paper_title: str = "",
    mode: str = "optimized",
    llm_provider: str = "openai",
    model_name: str = "gpt-4o",
    output_dir: str = "",
    progress_callback: Callable | None = None,
) -> CodeGenResult:
    """Convenience function for one-shot code generation.

    Args:
        paper_path: Path to paper PDF/Markdown.
        paper_id: Paper identifier.
        paper_title: Paper title.
        mode: "optimized" (fast) or "comprehensive" (full).
        llm_provider: "openai" | "anthropic" | "google".
        model_name: Specific model to use.
        output_dir: Where to save generated code.
        progress_callback: Progress callback.

    Returns:
        CodeGenResult with paths and status.
    """
    dc_config = DeepCodeConfig(
        llm_provider=llm_provider,
        model_name=model_name,
        output_base_dir=output_dir,
    )
    runner = DeepCodeRunner(dc_config)
    return await runner.generate(
        paper_path,
        paper_id=paper_id,
        paper_title=paper_title,
        mode=mode,
        progress_callback=progress_callback,
    )
