from __future__ import annotations

import json
import logging
from typing import Any, Callable

import httpx

from ml_platform.analysis.evidence import extract_evidence_sentences, run_self_correction
from ml_platform.analysis.models import AnalysisStatus, FiveW1H, PaperAnalysis, StrengthWeakness
from ml_platform.analysis.prompts import build_analysis_messages
from ml_platform.analysis.reference_chain import extract_references_from_parsed
from ml_platform.config import api_config

logger = logging.getLogger(__name__)


class PaperAnalyzer:
    """Orchestrates paper analysis: LLM extraction → evidence → self-correction.

    Usage:
        analyzer = PaperAnalyzer(model="gemma4:31b-cloud")
        result = await analyzer.analyze(paper)
    """

    def __init__(
        self,
        model: str | None = None,
        max_chars: int = 30000,
        enable_self_correction: bool = True,
        progress_callback: Callable[[str, float, str], Any] | None = None,
    ):
        """Initialize the analyzer.

        Args:
            model: Ollama model name. Defaults to OLLAMA_DEFAULT_MODEL from config.
            max_chars: Maximum paper text length for LLM input.
            enable_self_correction: Whether to run self-correction after initial analysis.
            progress_callback: Optional callback(stage, pct, message) for progress updates.
        """
        self.model = model or api_config.OLLAMA_DEFAULT_MODEL
        self.base_url = api_config.OLLAMA_BASE_URL
        self.max_chars = max_chars
        self.enable_self_correction = enable_self_correction
        self.progress_callback = progress_callback

    def _progress(self, stage: str, pct: float, msg: str) -> None:
        """Emit progress update."""
        if self.progress_callback:
            self.progress_callback(stage, pct, msg)
        logger.info("[%s] %.0f%% — %s", stage, pct, msg)

    async def analyze(self, paper: Any) -> PaperAnalysis:
        """Run full analysis pipeline on a paper.

        Args:
            paper: A Paper model object with parsed_content populated.

        Returns:
            Complete PaperAnalysis with 5W1H, evidence, and references.

        Raises:
            ValueError: If paper has no text content.
            RuntimeError: If LLM analysis fails.
        """
        paper_id = paper.paper_id
        self._progress("analyze", 0, f"Starting analysis for {paper_id}")

        # 1. Get paper text
        text = self._get_paper_text(paper)
        if not text:
            raise ValueError(f"No text content available for paper {paper_id}")
        self._progress("analyze", 10, f"Extracted {len(text)} chars of text")

        # 2. LLM analysis (5W1H + S/W)
        self._progress("analyze", 20, "Running 5W1H extraction via Ollama...")
        raw_analysis = await self._call_ollama(text)
        self._progress("analyze", 50, "5W1H extraction complete")

        # 3. Parse LLM response into PaperAnalysis
        analysis = self._parse_response(raw_analysis, paper_id)
        self._progress("analyze", 55, "Parsed analysis response")

        # 4. Extract references
        self._progress("analyze", 60, "Extracting references...")
        analysis.references = extract_references_from_parsed(paper.parsed_content)
        self._progress("analyze", 65, f"Found {len(analysis.references)} references")

        # 5. Extract evidence sentences
        self._progress("analyze", 70, "Extracting evidence sentences...")
        analysis.evidence = extract_evidence_sentences(analysis, text)
        self._progress("analyze", 80, f"Found {len(analysis.evidence)} evidence items")

        # 6. Self-correction (optional)
        if self.enable_self_correction:
            self._progress("analyze", 85, "Running self-correction...")
            analysis = await run_self_correction(analysis, text, model=self.model)
            self._progress("analyze", 95, "Self-correction complete")

        analysis.model_used = self.model
        if analysis.status == AnalysisStatus.PENDING:
            analysis.status = AnalysisStatus.COMPLETED

        self._progress("analyze", 100, "Analysis complete")
        return analysis

    def _get_paper_text(self, paper: Any) -> str:
        """Extract full text from a paper's parsed_content.

        Args:
            paper: Paper object with parsed_content dict.

        Returns:
            Concatenated text content, or empty string.
        """
        if not paper.parsed_content:
            return ""

        # GROBID-parsed: sections with text
        if "sections" in paper.parsed_content:
            sections = paper.parsed_content.get("sections", [])
            parts = []
            for sec in sections:
                if isinstance(sec, dict):
                    parts.append(sec.get("text", ""))
                elif isinstance(sec, str):
                    parts.append(sec)
            text = "\n\n".join(p for p in parts if p)
            # Add abstract if available
            abstract = paper.abstract or ""
            if abstract:
                text = f"Abstract: {abstract}\n\n{text}"
            return text

        # PyPDF2-parsed: raw text
        if "full_text" in paper.parsed_content:
            return paper.parsed_content["full_text"]
        if "raw_text" in paper.parsed_content:
            return paper.parsed_content["raw_text"]

        # Fallback: abstract only
        return paper.abstract or ""

    async def _call_ollama(self, paper_text: str) -> str:
        """Call Ollama chat API for analysis.

        Args:
            paper_text: Full text of the paper.

        Returns:
            Raw LLM response string (expected JSON).

        Raises:
            RuntimeError: If the API call fails.
        """
        messages = build_analysis_messages(paper_text, max_chars=self.max_chars)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 4096},
        }

        try:
            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(f"{self.base_url}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()

            content = data.get("message", {}).get("content", "")
            return content

        except httpx.TimeoutException:
            raise RuntimeError(f"Ollama API timeout (model: {self.model}, text: {len(paper_text)} chars)")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Ollama API error: {e.response.status_code} — {e.response.text[:200]}")
        except Exception as e:
            raise RuntimeError(f"Ollama API call failed: {e}")

    def _parse_response(self, raw: str, paper_id: str) -> PaperAnalysis:
        """Parse raw LLM JSON response into PaperAnalysis model.

        Handles various LLM output quirks (markdown fences, extra text,
        flat vs nested structure, alternative field names).

        Args:
            raw: Raw LLM response string.
            paper_id: Paper identifier.

        Returns:
            PaperAnalysis object.

        Raises:
            RuntimeError: If response cannot be parsed.
        """
        # Strip markdown code fences
        text = raw.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # Try to find JSON object in text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try with strict=False to handle invalid escape sequences
            try:
                data = json.loads(text.replace("\\", "\\\\"))
            except json.JSONDecodeError:
                # Last resort: try to fix common JSON issues
                import re
                # Remove trailing commas before } or ]
                text = re.sub(r',\s*([}\]])', r'\1', text)
                # Fix unescaped newlines in strings
                text = re.sub(r'(?<!\\)\n(?=[^"]*"[^"]*$)', '\\n', text)
                try:
                    data = json.loads(text)
                except json.JSONDecodeError as e:
                    logger.error("Failed to parse LLM response as JSON: %s", e)
                    logger.debug("Raw response: %s", raw[:500])
                    raise RuntimeError(f"Failed to parse LLM response: {e}") from e

        try:
            data = self._normalize_response(data)
            five_w1h = FiveW1H(**data.get("five_w1h", {}))
            sw = StrengthWeakness(**data.get("sw", {}))
            return PaperAnalysis(
                paper_id=paper_id,
                five_w1h=five_w1h,
                sw=sw,
                summary=data.get("summary", ""),
                key_contributions=data.get("key_contributions", []),
                methodology_type=data.get("methodology_type", ""),
                domain=data.get("domain", ""),
                status=AnalysisStatus.PENDING,
            )
        except Exception as e:
            logger.error("Failed to build PaperAnalysis: %s", e)
            logger.debug("Parsed data keys: %s", list(data.keys()) if isinstance(data, dict) else type(data))
            raise RuntimeError(f"Failed to build PaperAnalysis from LLM response: {e}") from e

    def _normalize_response(self, data: dict) -> dict:
        """Normalize LLM response to expected schema.

        Handles common variations:
        - Flat structure (who/what/when at top level instead of nested in five_w1h)
        - Alternative field names (strengths_weaknesses → sw, etc.)
        - Missing nested objects

        Args:
            data: Parsed JSON dict from LLM.

        Returns:
            Normalized dict matching the expected schema.
        """
        # Ensure five_w1h exists and has all fields
        w1h_fields = {"who", "what", "when", "where", "why", "how"}
        if "five_w1h" not in data or not isinstance(data.get("five_w1h"), dict):
            # Maybe flat structure — collect w1h fields from top level
            flat_w1h = {k: data.pop(k) for k in w1h_fields if k in data}
            if flat_w1h:
                data["five_w1h"] = flat_w1h
            else:
                data["five_w1h"] = {}
        else:
            # Ensure all fields are strings
            for field in w1h_fields:
                if field not in data["five_w1h"] or data["five_w1h"][field] is None:
                    data["five_w1h"][field] = ""

        # Ensure sw exists
        sw_aliases = ["sw", "strengths_weaknesses", "strengths_and_weaknesses", "analysis"]
        sw_data = None
        for alias in sw_aliases:
            if alias in data and isinstance(data[alias], dict):
                sw_data = data.pop(alias) if alias != "sw" else data[alias]
                break

        if sw_data is None:
            # Maybe flat structure
            sw_flat = {}
            for key in ["strengths", "weaknesses", "future_work", "improvements"]:
                if key in data:
                    sw_flat[key] = data.pop(key)
            if sw_flat:
                sw_data = sw_flat

        if sw_data is not None:
            data["sw"] = sw_data
        elif "sw" not in data:
            data["sw"] = {}

        # Ensure list fields are lists
        for list_field in ["strengths", "weaknesses", "future_work"]:
            if list_field in data["sw"] and isinstance(data["sw"][list_field], str):
                data["sw"][list_field] = [data["sw"][list_field]]

        if "key_contributions" in data and isinstance(data["key_contributions"], str):
            data["key_contributions"] = [data["key_contributions"]]

        return data
