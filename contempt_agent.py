"""
contempt_agent.py
-----------------
Agentic contempt detection pipeline for the BAIT backend.
Written by: Afreen Sorathiya | Summer 2026 | Sustaining PEACE Project

This module slots into the existing BAIT backend alongside llm_analyzer.py.
It receives a transcript (already extracted by the existing pipeline) and runs
it through a 5-agent LLM pipeline to detect contemptuous speech.

Usage in main.py:
    from contempt_agent import analyze_transcript_for_contempt
    result = analyze_transcript_for_contempt(transcript)
"""

import os
import json
import logging
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Logging ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Model Setup (lazy-loaded) ──────────────────────────────────────────────────
_llm_strong = None
_llm_fast   = None

def _get_models():
    """Lazy-load models so they pick up the API key after dotenv loads."""
    global _llm_strong, _llm_fast
    if _llm_strong is None:
        api_key     = os.getenv("OPENAI_API_KEY", "")
        _llm_strong = ChatOpenAI(model="gpt-4o",      temperature=0, api_key=api_key)
        _llm_fast   = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    return _llm_strong, _llm_fast

# ── Text Splitter ──────────────────────────────────────────────────────────────
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", "! ", "? ", " "]
)

# ── Helper ─────────────────────────────────────────────────────────────────────
def _call_agent(llm, system_prompt: str, user_content: str) -> str:
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
    return llm.invoke(messages).content.strip()

# ── Agent 1: Context ───────────────────────────────────────────────────────────
def _context_agent(chunk: str, llm) -> str:
    system = """You are a context analysis agent for YouTube video transcripts.
Briefly identify:
1. Who appears to be speaking
2. Who or what is being addressed or discussed (the target)
3. The general topic or situation
Do NOT judge tone or intent yet. Be concise."""
    return _call_agent(llm, system, f"Transcript chunk:\n{chunk}")

# ── Agent 2: Semantics ─────────────────────────────────────────────────────────
def _semantics_agent(chunk: str, llm) -> str:
    system = """You are a semantics analysis agent for YouTube video transcripts.
Extract only the LITERAL meaning of the text:
- What is the speaker literally claiming or saying?
- Ignore tone, sarcasm, or subtext for now.
Be concise."""
    return _call_agent(llm, system, f"Transcript chunk:\n{chunk}")

# ── Agent 3: Tone ──────────────────────────────────────────────────────────────
def _tone_agent(chunk: str, llm) -> str:
    system = """You are a tone and rhetoric analysis agent for YouTube video transcripts.
Identify:
1. Overall tone (neutral / sarcastic / mocking / dismissive / aggressive / respectful)
2. Specific words or phrases signalling contempt, sarcasm, mockery, or superiority
3. Whether there is a gap between what is said and how it is said
Quote the exact phrases that carry the tone signal."""
    return _call_agent(llm, system, f"Transcript chunk:\n{chunk}")

# ── Run Agents 1-3 in Parallel ─────────────────────────────────────────────────
def _run_parallel_agents(chunk: str, llm_fast):
    with ThreadPoolExecutor(max_workers=3) as executor:
        fc = executor.submit(_context_agent,   chunk, llm_fast)
        fs = executor.submit(_semantics_agent, chunk, llm_fast)
        ft = executor.submit(_tone_agent,      chunk, llm_fast)
        return fc.result(), fs.result(), ft.result()

# ── Agent 4: Decision ──────────────────────────────────────────────────────────
def _decision_agent(chunk: str, context: str, semantics: str, tone: str, llm_strong) -> Dict:
    system = """You are a contempt detection decision agent for YouTube video transcripts.
You receive the original chunk plus three analyses: context, literal meaning, and tone.

Your job:
1. Compare literal meaning vs tone — is there a gap? (gap = contempt/sarcasm signal)
2. Decide: is this chunk contemptuous? YES / NO / UNCERTAIN
3. Give a confidence score 0-100 indicating how certain you are about the verdict
4. Give a contempt score 0-100 indicating the intensity of contempt in the text:
   - 0-20 = little or no contempt
   - 21-40 = low contempt
   - 41-60 = moderate contempt
   - 61-80 = high contempt
   - 81-100 = very high contempt
5. Explain your reasoning in 2-3 sentences
6. List the specific phrases that triggered your verdict
Contempt = mockery, dismissiveness, or superiority directed at a person or group.
NOT the same as anger, blunt criticism, or factual reporting about contemptuous acts.

Respond ONLY in this exact JSON format:
{
  "verdict": "YES" | "NO" | "UNCERTAIN",
  "confidence": <0-100>,
  "contempt_score": <0-100>,
  "reasoning": "<2-3 sentences>",
  "key_phrases": ["<phrase1>", "<phrase2>"]
}"""
    user = f"Chunk: {chunk}\n\nContext: {context}\nLiteral meaning: {semantics}\nTone analysis: {tone}"
    raw = _call_agent(llm_strong, system, user)
    try:
        return json.loads(raw.replace("```json", "").replace("```", "").strip())
    except json.JSONDecodeError:
        return {"verdict": "UNCERTAIN", "confidence": 0, "reasoning": raw, "key_phrases": []}

# ── Agent 5: Critic (conditional) ─────────────────────────────────────────────
def _critic_agent(chunk: str, decision: Dict, llm_strong) -> Dict:
    system = """You are a critic agent reviewing a contempt detection verdict for a YouTube transcript.
Check for these common mistakes:
- Confusing contempt with anger or strong disagreement
- Confusing contempt with blunt but honest criticism
- Missing subtle or implicit contempt
- Flagging a reporter DESCRIBING contemptuous speech as contemptuous themselves

Respond ONLY in this JSON format:
{
  "revised_verdict": "YES" | "NO" | "UNCERTAIN",
  "revised_confidence": <0-100>,
  "critique": "<1-2 sentences>"
}"""
    user = f"Chunk: {chunk}\n\nPrevious verdict: {decision['verdict']}\nPrevious confidence: {decision['confidence']}\nPrevious reasoning: {decision['reasoning']}"
    raw = _call_agent(llm_strong, system, user)
    try:
        return json.loads(raw.replace("```json", "").replace("```", "").strip())
    except json.JSONDecodeError:
        return {"revised_verdict": decision["verdict"], "revised_confidence": decision["confidence"], "critique": raw}

# ── Per-chunk Pipeline ─────────────────────────────────────────────────────────
def _analyze_chunk(chunk: str, chunk_num: int, confidence_threshold: int = 55) -> Dict:
    llm_strong, llm_fast = _get_models()
    logger.info(f"  contempt_agent: chunk {chunk_num}...")
    context, semantics, tone = _run_parallel_agents(chunk, llm_fast)
    decision = _decision_agent(chunk, context, semantics, tone, llm_strong)
    critic_output = None
    if decision.get("confidence", 0) < confidence_threshold:
        critic_output    = _critic_agent(chunk, decision, llm_strong)
        final_verdict    = critic_output["revised_verdict"]
        final_confidence = critic_output["revised_confidence"]
    else:
        final_verdict    = decision["verdict"]
        final_confidence = decision["confidence"]
    return {
        "chunk_num":        chunk_num,
        "final_verdict":    final_verdict,
        "final_confidence": final_confidence,
        "key_phrases":      decision.get("key_phrases", []),
        "reasoning":        decision.get("reasoning", ""),
        "agents":           {"context": context, "semantics": semantics, "tone": tone},
        "decision_agent":   decision,
        "critic_agent":     critic_output,
    }

# ── Main Public Function ───────────────────────────────────────────────────────
def analyze_transcript_for_contempt(
    transcript: str,
    confidence_threshold: int = 55
) -> Dict[str, Any]:
    """
    Main entry point for the BAIT backend.

    Receives a transcript string (already extracted by the existing pipeline)
    and returns a structured contempt detection result.

    Args:
        transcript:           The full transcript text.
        confidence_threshold: Chunks below this trigger the Critic Agent (default 55).

    Returns:
        {
            "verdict":         "YES" | "NO" | "UNCERTAIN",
            "confidence":      <int 0-100>,
            "vote_agreement":  <int 0-100>,
            "key_phrases":     [<str>, ...],
            "chunk_breakdown": {"YES": n, "NO": n, "UNCERTAIN": n},
            "chunk_results":   [ ... full per-chunk detail ... ]
        }
    """
    logger.info("contempt_agent: starting analysis...")
    chunks = _splitter.split_text(transcript.strip())
    if not chunks:
        return {"verdict": "UNCERTAIN", "confidence": 0, "vote_agreement": 0,
                "key_phrases": [], "chunk_breakdown": {}, "chunk_results": []}

    logger.info(f"contempt_agent: {len(chunks)} chunk(s) to analyze.")
    chunk_results = [_analyze_chunk(chunk, i, confidence_threshold) for i, chunk in enumerate(chunks, 1)]

    verdicts         = [r["final_verdict"] for r in chunk_results]
    counts           = Counter(verdicts)
    top_label, top_n = counts.most_common(1)[0]
    avg_confidence   = round(sum(r["final_confidence"] for r in chunk_results) / len(chunk_results))
    vote_agreement   = round(top_n / len(chunk_results) * 100)

    all_phrases: List[str] = []
    for r in chunk_results:
        all_phrases.extend(r.get("key_phrases", []))
    top_phrases = list(dict.fromkeys(all_phrases))[:6]

    logger.info(f"contempt_agent: verdict={top_label}, confidence={avg_confidence}, agreement={vote_agreement}%")

    return {
        "verdict":         top_label,
        "confidence":      avg_confidence,
        "vote_agreement":  vote_agreement,
        "key_phrases":     top_phrases,
        "chunk_breakdown": dict(counts),
        "chunk_results":   chunk_results
    }
