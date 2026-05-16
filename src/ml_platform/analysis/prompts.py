from __future__ import annotations

ANALYSIS_SYSTEM_PROMPT = """You are a research paper analysis expert. You extract structured information from academic papers using the 5W1H framework.
You respond ONLY with valid JSON matching the specified schema. No markdown, no explanation, just JSON."""

ANALYSIS_PROMPT_TEMPLATE = """Analyze the following research paper text and extract structured information.

Respond with a JSON object matching this exact schema:

{{
  "five_w1h": {{
    "who": "Researchers, affiliations, research groups involved",
    "what": "Research problem, key contributions, main claims",
    "when": "Publication timeline and temporal context",
    "where": "Venue, journal, or institution",
    "why": "Motivation, existing limitations, gap being addressed",
    "how": "Methodology, datasets, architecture, experimental setup"
  }},
  "sw": {{
    "strengths": ["list of paper strengths"],
    "weaknesses": ["list of paper weaknesses or limitations"],
    "future_work": ["improvement directions, both author-stated and derived"]
  }},
  "summary": "One-paragraph summary of the paper",
  "key_contributions": ["list of key contributions as bullet points"],
  "methodology_type": "One of: empirical, theoretical, survey, hybrid",
  "domain": "Research domain (e.g. computer vision, NLP, optimization)"
}}

CRITICAL RULES:
1. Each field must be based on EXPLICIT content from the paper.
2. For strengths/weaknesses, provide 3-7 items each.
3. For future_work, include both author-stated and your derived suggestions.
4. Be specific — avoid vague statements like "novel approach" without details.
5. If information is not available, write "Not specified in the paper."

PAPER TEXT:
---
{paper_text}
---

Respond with JSON only:"""

SELF_CORRECTION_PROMPT = """You are reviewing a paper analysis for accuracy and completeness.

ORIGINAL PAPER TEXT (excerpt):
---
{paper_excerpt}
---

CURRENT ANALYSIS:
---
{current_analysis}
---

Check the analysis against the paper text. For each field:
1. Is the claim accurate? Does it match what the paper actually says?
2. Is anything important missing?
3. Are there any hallucinated claims not supported by the text?

Respond with a JSON object:
{{
  "corrections_needed": true/false,
  "corrected_five_w1h": {{ ... }},  // only if corrections needed
  "corrected_sw": {{ ... }},          // only if corrections needed
  "corrected_summary": "...",         // only if corrections needed
  "notes": "Explanation of what was corrected and why"
}}

If the analysis is accurate, respond with: {{"corrections_needed": false, "notes": "Analysis verified."}}"""

# Few-shot example for the analysis prompt
FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": """Analyze this paper excerpt:

---
Denoising Diffusion Probabilistic Models. Ho et al., NeurIPS 2020.
We present high quality image synthesis results using diffusion probabilistic models. Our models are parameterized as a Markov chain that reverses a diffusion process. We show that our training objective is a weighted variational bound on the data likelihood. We demonstrate that our models can synthesize high-fidelity images competitive with GANs.
---
""",    
    },
    {
        "role": "assistant",
        "content": """{{
  "five_w1h": {{
    "who": "Jonathan Ho, Ajay Jain, Pieter Abbeel (UC Berkeley)",
    "what": "High quality image synthesis using diffusion probabilistic models, demonstrating competitive results with GANs",
    "when": "NeurIPS 2020",
    "where": "NeurIPS 2020, UC Berkeley",
    "why": "Existing generative models (GANs) are difficult to train and can suffer from mode collapse; diffusion models offer a more stable training alternative",
    "how": "Parameterized as a Markov chain reversing a diffusion process, trained with a weighted variational bound on data likelihood"
  }},
  "sw": {{
    "strengths": ["Competitive image quality with GANs", "Stable training without adversarial dynamics", "Theoretical foundation via variational bound"],
    "weaknesses": ["Slow sampling requiring many diffusion steps", "Limited to image generation in this work"],
    "future_work": ["Accelerating sampling with fewer steps", "Extending to other modalities", "Combining with guidance mechanisms for controlled generation"]
  }},
  "summary": "This paper introduces denoising diffusion probabilistic models (DDPM) that generate high-fidelity images by learning to reverse a gradual noising process, achieving quality competitive with GANs through stable training.",
  "key_contributions": ["Demonstrated high-quality image synthesis via diffusion models", "Connected training objective to weighted variational bound", "Showed competitive results with GANs without adversarial training"],
  "methodology_type": "empirical",
  "domain": "computer vision"
}}""",
    },
]


def build_analysis_messages(paper_text: str, max_chars: int = 30000) -> list[dict]:
    """Build Ollama chat messages for paper analysis.
    
    Args:
        paper_text: Full text of the paper.
        max_chars: Maximum characters to include (truncation limit).
        
    Returns:
        List of message dicts for Ollama chat API.
    """
    truncated = paper_text[:max_chars]
    if len(paper_text) > max_chars:
        truncated += f"\n\n[TRUNCATED: showing first {max_chars} of {len(paper_text)} characters]"
    
    messages = [{"role": "system", "content": ANALYSIS_SYSTEM_PROMPT}]
    
    # Add few-shot examples
    for example in FEW_SHOT_EXAMPLES:
        messages.append(example)
    
    messages.append({
        "role": "user",
        "content": ANALYSIS_PROMPT_TEMPLATE.format(paper_text=truncated),
    })
    
    return messages


def build_correction_messages(
    paper_excerpt: str,
    current_analysis: str,
    max_chars: int = 15000,
) -> list[dict]:
    """Build Ollama chat messages for self-correction review.
    
    Args:
        paper_excerpt: Paper text excerpt for verification.
        current_analysis: JSON string of current analysis.
        max_chars: Maximum characters for excerpt.
        
    Returns:
        List of message dicts for Ollama chat API.
    """
    truncated = paper_excerpt[:max_chars]
    
    return [
        {"role": "system", "content": "You are a JSON-only responder. Output valid JSON with no markdown."},
        {
            "role": "user",
            "content": SELF_CORRECTION_PROMPT.format(
                paper_excerpt=truncated,
                current_analysis=current_analysis,
            ),
        },
    ]
