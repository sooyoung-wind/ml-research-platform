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
from typing import Callable

from ml_platform.config import config


@dataclass
class CodeGenResult:
    """Result of a code generation run.

    Attributes:
        success: Whether code generation succeeded.
        paper_id: Identifier of the source paper.
        paper_title: Title of the source paper.
        output_dir: Directory containing generated code.
        files_generated: List of relative paths to generated files.
        duration_seconds: Time taken for generation.
        error: Error message on failure.
        pipeline_mode: Pipeline mode used ("optimized" or "comprehensive").
        model_used: LLM model used for generation.
    """

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
    """Configuration for DeepCode pipeline.

    Attributes:
        llm_provider: LLM provider ("openai", "anthropic", "google", or "ollama").
        model_name: Provider-specific model name (e.g. "gpt-4o", "qwen3:8b").
        planning_model: Optional separate model for planning.
        enable_indexing: Enable CodeRAG reference mining (comprehensive mode).
        enable_segmentation: Enable document segmentation for long papers.
        segmentation_threshold: Character count threshold for segmentation.
        max_iterations: Maximum implementation iterations.
        output_base_dir: Working directory for generated code.
        openai_api_key: OpenAI API key (falls back to env var).
        anthropic_api_key: Anthropic API key (falls back to env var).
        google_api_key: Google API key (falls back to env var).
        ollama_base_url: Ollama server URL (default: http://localhost:11434).
    """

    # LLM provider: "openai" | "anthropic" | "google" | "ollama"
    # Default comes from ML_DEFAULT_LLM_PROVIDER env var (see .env.example)
    llm_provider: str = ""
    # Model name (provider-specific)
    # Default comes from ML_DEFAULT_LLM_MODEL env var
    model_name: str = ""
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
    # Ollama settings (local inference, no API key needed)
    ollama_base_url: str = "http://localhost:11434"


class DeepCodeRunner:
    """Wrapper around DeepCode's multi-agent pipeline.

    Provides a simple interface::

        runner = DeepCodeRunner(config)
        result = await runner.generate(paper_path, paper_id="2312.00752")

    Supports two modes:
        - optimized: Fast mode, no reference mining, lower cost
        - comprehensive: Full pipeline with CodeRAG, higher quality

    Attributes:
        config: The DeepCodeConfig used to initialise the runner.
        output_dir: Base output directory for generated code.
        config_dir: DeepCode configuration directory.
    """

    def __init__(self, dc_config: DeepCodeConfig | None = None) -> None:
        """Initialize the DeepCodeRunner.

        Args:
            dc_config: Configuration for the DeepCode pipeline.  Uses defaults
                if not provided.
        """
        self.config = dc_config or DeepCodeConfig()
        # Fill empty provider/model from platform config (.env)
        from ml_platform.config import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER

        if not self.config.llm_provider:
            self.config.llm_provider = DEFAULT_LLM_PROVIDER
        if not self.config.model_name:
            self.config.model_name = DEFAULT_LLM_MODEL
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
            secrets.setdefault("openai", {})["api_key"] = keys["openai"]
        if keys["anthropic"]:
            secrets.setdefault("anthropic", {})["api_key"] = keys["anthropic"]
        if keys["google"]:
            secrets.setdefault("google", {})["api_key"] = keys["google"]

        secrets_path = self.config_dir / "mcp_agent.secrets.yaml"
        with open(secrets_path, "w") as f:
            yaml.dump(secrets, f, default_flow_style=False)

        # Write main config
        main_config: dict = {
            "llm_provider": self.config.llm_provider,
        }

        # Ollama support for CWD-based execution
        if self.config.llm_provider == "ollama":
            main_config["default_model"] = "ollama"
            base_url = self.config.ollama_base_url.rstrip("/") + "/v1"
            main_config.setdefault("openai", {})["base_url"] = base_url
            main_config["openai"]["api_key"] = "ollama"
            main_config["openai"][
                "default_model"
            ] = self.config.model_name
            secrets.setdefault("openai", {})["api_key"] = "ollama"
            with open(secrets_path, "w") as f:
                yaml.dump(secrets, f, default_flow_style=False)

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
            CodeGenResult with paths to generated code and status.

        Raises:
            Exception: If the DeepCode pipeline encounters an unexpected
                error (caught and stored in result.error).
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

        Args:
            paper_path: Path to the paper file.
            output_dir: Directory to copy generated code into.
            mode: Pipeline mode ("optimized" or "comprehensive").
            progress_callback: Optional progress callback.

        Returns:
            True if generated files were found, False otherwise.

        Raises:
            Exception: Propagated from DeepCode's orchestration engine.
        """
        with tempfile.TemporaryDirectory(prefix="deepcode_") as tmpdir:
            self._prepare_deepcode_input(paper_path, tmpdir)

            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                pipeline_result = await self._execute_deepcode_pipeline(
                    tmpdir, tmpdir, mode, progress_callback,
                )
                return self._copy_generated_output(tmpdir, output_dir)
            finally:
                os.chdir(original_cwd)

    def _prepare_deepcode_input(self, paper_path: str, tmpdir: str) -> None:
        """Copy paper file and write config files into the temp directory.

        Args:
            paper_path: Path to the source paper file.
            tmpdir: Temporary working directory for DeepCode.
        """
        import yaml

        paper_name = Path(paper_path).name
        tmp_paper = os.path.join(tmpdir, paper_name)
        shutil.copy2(paper_path, tmp_paper)

        # Build mcp-agent secrets YAML (nested format expected by mcp_agent)
        secrets: dict = {}
        openai_key = (
            self.config.openai_api_key
            or os.environ.get("OPENAI_API_KEY", "")
        )
        anthropic_key = (
            self.config.anthropic_api_key
            or os.environ.get("ANTHROPIC_API_KEY", "")
        )
        google_key = (
            self.config.google_api_key
            or os.environ.get("GOOGLE_API_KEY", "")
            or os.environ.get("GEMINI_API_KEY", "")
        )

        if openai_key:
            secrets.setdefault("openai", {})["api_key"] = openai_key
        if anthropic_key:
            secrets.setdefault("anthropic", {})["api_key"] = anthropic_key
        if google_key:
            secrets.setdefault("google", {})["api_key"] = google_key

        with open(os.path.join(tmpdir, "mcp_agent.secrets.yaml"), "w") as f:
            yaml.dump(secrets, f, default_flow_style=False)

        # Build mcp-agent config YAML
        main_config: dict = {}
        if self.config.llm_provider:
            main_config["llm_provider"] = self.config.llm_provider

        # Ollama support: set openai.base_url to Ollama endpoint
        if self.config.llm_provider == "ollama":
            main_config["default_model"] = "ollama"
            base_url = self.config.ollama_base_url.rstrip("/") + "/v1"
            main_config.setdefault("openai", {})["base_url"] = base_url
            main_config["openai"]["api_key"] = "ollama"
            main_config["openai"][
                "default_model"
            ] = self.config.model_name
            # Write dummy openai key in secrets so get_api_keys finds it
            secrets.setdefault("openai", {})["api_key"] = "ollama"
            with open(os.path.join(tmpdir, "mcp_agent.secrets.yaml"), "w") as f:
                yaml.dump(secrets, f, default_flow_style=False)

        with open(os.path.join(tmpdir, "mcp_agent.config.yaml"), "w") as f:
            yaml.dump(main_config, f, default_flow_style=False)

    async def _execute_deepcode_pipeline(
        self,
        paper_dir: str,
        tmpdir: str,
        mode: str,
        progress_callback: Callable | None,
    ) -> None:
        """Import and run the DeepCode orchestration pipeline.

        Args:
            paper_dir: Directory containing the paper file.
            tmpdir: Temporary working directory (CWD for DeepCode).
            mode: Pipeline mode ("optimized" or "comprehensive").
            progress_callback: Optional progress callback.
        """
        import logging as _logging
        from workflows.agent_orchestration_engine import (
            execute_multi_agent_research_pipeline,
        )

        dc_logger = _logging.getLogger("deepcode")
        dc_logger.setLevel(_logging.INFO)

        paper_name = [f for f in os.listdir(paper_dir)
                       if f.endswith((".pdf", ".md"))][0]
        tmp_paper = os.path.join(tmpdir, paper_name)
        enable_indexing = mode == "comprehensive"

        await execute_multi_agent_research_pipeline(
            input_source=tmp_paper,
            logger=dc_logger,
            progress_callback=progress_callback,
            enable_indexing=enable_indexing,
        )

    @staticmethod
    def _copy_generated_output(tmpdir: str, output_dir: str) -> bool:
        """Copy generated files from deepcode_lab/ to the output directory.

        Args:
            tmpdir: Temporary directory containing deepcode_lab/.
            output_dir: Target directory for generated files.

        Returns:
            True if files were copied, False if no output was found.
        """
        lab_dir = os.path.join(tmpdir, "deepcode_lab")
        if not os.path.isdir(lab_dir):
            return False

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

    @staticmethod
    def _list_generated_files(directory: Path) -> list[str]:
        """List all generated code files.

        Args:
            directory: Directory to scan for generated files.

        Returns:
            Sorted list of relative file paths with recognised extensions.
        """
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
    llm_provider: str = "",
    model_name: str = "",
    output_dir: str = "",
    progress_callback: Callable | None = None,
) -> CodeGenResult:
    """Convenience function for one-shot code generation.

    Args:
        paper_path: Path to paper PDF/Markdown.
        paper_id: Paper identifier.
        paper_title: Paper title.
        mode: "optimized" (fast) or "comprehensive" (full).
        llm_provider: "openai" | "anthropic" | "google" | "ollama".
        model_name: Specific model to use (e.g. "gpt-4o", "qwen3:8b").
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
