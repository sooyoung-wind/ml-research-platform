from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from ml_platform.analysis.models import AnalysisStatus, EvidenceItem, PaperAnalysis
from ml_platform.analysis.prompts import build_correction_messages
from ml_platform.config import api_config

logger = logging.getLogger(__name__)


def extract_evidence_sentences(
    analysis: PaperAnalysis,
    paper_text: str,
) -> list[EvidenceItem]:
    """Match analysis claims to evidence sentences in the original paper text.
    
    For each key claim in the analysis, find the most relevant sentence
    in the paper text that supports it.
    
    Args:
        analysis: The paper analysis with claims.
        paper_text: Full text of the paper.
        
    Returns:
        List of EvidenceItem objects linking claims to evidence.
    """
    evidence = []
    sentences = _split_sentences(paper_text)
    
    # Extract claims from FiveW1H
    claims = [
        ("who", analysis.five_w1h.who),
        ("what", analysis.five_w1h.what),
        ("when", analysis.five_w1h.when),
        ("where", analysis.five_w1h.where),
        ("why", analysis.five_w1h.why),
        ("how", analysis.five_w1h.how),
    ]
    
    for claim_type, claim_text in claims:
        if not claim_text or claim_text.startswith("Not specified"):
            continue
        
        best_match = _find_best_match(claim_text, sentences)
        if best_match:
            evidence.append(EvidenceItem(
                claim_type=claim_type,
                claim_text=claim_text,
                evidence_text=best_match,
                confidence=_compute_confidence(claim_text, best_match),
            ))
    
    # Extract evidence for strengths
    for strength in analysis.sw.strengths:
        best_match = _find_best_match(strength, sentences)
        if best_match:
            evidence.append(EvidenceItem(
                claim_type="strength",
                claim_text=strength,
                evidence_text=best_match,
                confidence=_compute_confidence(strength, best_match),
            ))
    
    # Extract evidence for weaknesses
    for weakness in analysis.sw.weaknesses:
        best_match = _find_best_match(weakness, sentences)
        if best_match:
            evidence.append(EvidenceItem(
                claim_type="weakness",
                claim_text=weakness,
                evidence_text=best_match,
                confidence=_compute_confidence(weakness, best_match),
            ))
    
    return evidence


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Simple sentence splitter
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def _find_best_match(claim: str, sentences: list[str]) -> str | None:
    """Find the sentence that best matches a claim using keyword overlap.
    
    Args:
        claim: The analysis claim text.
        sentences: List of sentences from the paper.
        
    Returns:
        Best matching sentence, or None if no good match found.
    """
    claim_words = set(claim.lower().split())
    # Remove common stopwords for better matching
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as',
                 'this', 'that', 'these', 'those', 'it', 'its', 'we', 'our', 'their',
                 'and', 'or', 'but', 'not', 'no', 'has', 'have', 'had', 'can', 'could',
                 'will', 'would', 'shall', 'should', 'may', 'might', 'must'}
    claim_words -= stopwords
    
    if not claim_words:
        return None
    
    best_score = 0.0
    best_sentence = None
    
    for sentence in sentences:
        sent_words = set(sentence.lower().split()) - stopwords
        if not sent_words:
            continue
        
        overlap = claim_words & sent_words
        score = len(overlap) / max(len(claim_words), 1)
        
        if score > best_score:
            best_score = score
            best_sentence = sentence
    
    # Only return if there's reasonable overlap
    if best_score >= 0.15:
        return best_sentence
    return None


def _compute_confidence(claim: str, evidence: str) -> float:
    """Compute a confidence score for a claim-evidence pair.
    
    Based on keyword overlap ratio.
    """
    claim_words = set(claim.lower().split())
    evidence_words = set(evidence.lower().split())
    
    if not claim_words:
        return 0.0
    
    overlap = claim_words & evidence_words
    return min(len(overlap) / len(claim_words), 1.0)


async def run_self_correction(
    analysis: PaperAnalysis,
    paper_text: str,
    model: str | None = None,
) -> PaperAnalysis:
    """Run self-correction loop: verify analysis against paper text.
    
    Sends the analysis back to the LLM for verification and potential correction.
    
    Args:
        analysis: Initial paper analysis.
        paper_text: Original paper text for verification.
        model: Ollama model to use. Defaults to config default.
        
    Returns:
        Corrected PaperAnalysis (or original if no corrections needed).
    """
    model = model or api_config.OLLAMA_DEFAULT_MODEL
    base_url = api_config.OLLAMA_BASE_URL
    
    analysis_json = analysis.model_dump_json()
    messages = build_correction_messages(
        paper_excerpt=paper_text[:15000],
        current_analysis=analysis_json,
    )
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 4096},
    }
    
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(f"{base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            
        content = data.get("message", {}).get("content", "")
        content = _strip_json_markers(content)
        
        correction = json.loads(content)
        
        if not correction.get("corrections_needed", False):
            analysis.self_correction_applied = True
            analysis.correction_notes = "Analysis verified, no corrections needed."
            analysis.status = AnalysisStatus.COMPLETED
            return analysis
        
        # Apply corrections
        if "corrected_five_w1h" in correction:
            from ml_platform.analysis.models import FiveW1H
            analysis.five_w1h = FiveW1H(**correction["corrected_five_w1h"])
        if "corrected_sw" in correction:
            from ml_platform.analysis.models import StrengthWeakness
            analysis.sw = StrengthWeakness(**correction["corrected_sw"])
        if "corrected_summary" in correction:
            analysis.summary = correction["corrected_summary"]
        
        analysis.self_correction_applied = True
        analysis.correction_notes = correction.get("notes", "Corrections applied via self-correction.")
        analysis.status = AnalysisStatus.COMPLETED
        
        logger.info("Self-correction applied: %s", analysis.correction_notes)
        return analysis
        
    except Exception as e:
        logger.warning("Self-correction failed: %s", e)
        analysis.status = AnalysisStatus.COMPLETED
        analysis.correction_notes = f"Self-correction skipped: {e}"
        return analysis


def _strip_json_markers(text: str) -> str:
    """Remove markdown code fences and other markers from LLM output."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()
