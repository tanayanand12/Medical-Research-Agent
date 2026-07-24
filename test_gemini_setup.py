"""
test_gemini_setup.py
- LLM  → Gemini API (offloads the slow Llama3 synthesis)  
- Embed → Ollama nomic-embed-text (fast local, no rate limits, same 768d)
"""
import os, time
from dotenv import load_dotenv # type: ignore
load_dotenv()

import litellm # type: ignore

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

LLM_CASCADE = [
    "gemini/gemini-2.5-flash-lite",
    "gemini/gemini-2.5-flash",
]


def try_llm(models, retries=3, wait=5):
    for model in models:
        for attempt in range(1, retries + 1):
            try:
                print(f"  [{attempt}/{retries}] Trying {model}...")
                resp = litellm.completion(
                    model=model,
                    messages=[{"role": "user", "content": "Reply with only the word: OK"}],
                    api_key=GEMINI_KEY,
                    timeout=30,
                )
                text = resp.choices[0].message.content.strip()
                print(f"  ✅ {model} → {text!r}")
                return model
            except litellm.exceptions.ServiceUnavailableError:
                print(f"  ⚠️  503 (attempt {attempt}) — waiting {wait}s...")
                if attempt < retries:
                    time.sleep(wait)
            except Exception as e:
                print(f"  ❌ {model}: {type(e).__name__}: {str(e)[:100]}")
                break
    return None


def try_ollama_embed(retries=2):
    for attempt in range(1, retries + 1):
        try:
            print(f"  [{attempt}/{retries}] Ollama nomic-embed-text...")
            emb = litellm.embedding(
                model="ollama/nomic-embed-text",
                input=["metformin lactic acidosis"],
                api_base=OLLAMA_URL,
            )
            dim = len(emb.data[0]["embedding"])
            print(f"  ✅ ollama/nomic-embed-text → {dim}d vector")
            return dim
        except Exception as e:
            print(f"  ❌ Ollama embed failed: {str(e)[:120]}")
            if attempt < retries:
                time.sleep(2)
    return None


print("=" * 55)
print("  Medical Research Agent — provider check")
print("=" * 55)

print("\n[1] LLM — Gemini API (synthesis, query expansion)")
llm = try_llm(LLM_CASCADE)

print("\n[2] Embeddings — Ollama local (chunk indexing)")
dim = try_ollama_embed()

print("\n" + "=" * 55)
if llm and dim:
    print("  ✅ PASS")
    print(f"\n  Confirmed working config:")
    print(f"  DEFAULT_LLM_MODEL={llm}")
    print(f"  DEFAULT_EMBEDDING_MODEL=ollama/nomic-embed-text")
    print(f"  Embedding dimensions: {dim}d")
    print(f"\n  Expected speedup vs full-Ollama:")
    print(f"  Synthesis: ~600s → ~5s  (Gemini Flash-Lite)")
    print(f"  Embeddings: unchanged  (already fast locally)")
else:
    issues = []
    if not llm:
        issues.append("Gemini LLM — check GEMINI_API_KEY")
    if not dim:
        issues.append("Ollama embed — is Ollama running? (ollama serve)")
    for i in issues:
        print(f"  ❌ {i}")
print("=" * 55)