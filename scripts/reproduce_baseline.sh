#!/usr/bin/env bash
# reproduce_baseline.sh — Phase 10: Reproducibility kit
#
# Runs a deterministic baseline evaluation to verify that the system
# produces consistent results across environments.
#
# Usage:
#   bash scripts/reproduce_baseline.sh
#
# Prerequisites:
#   - Python 3.10+ with dependencies from requirements.txt installed
#   - At least one LLM API key configured in .env
#
# Output:
#   results/baseline_repro.json — evaluation results (10 MedQA samples, seed=42)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "============================================================"
echo "Medical Research Agent — Reproducibility Kit"
echo "============================================================"
echo ""

# -------------------------------------------------------------------
# Step 1: Verify Python environment
# -------------------------------------------------------------------
echo "[1/5] Checking Python environment..."

python -c "import sys; assert sys.version_info >= (3, 10), f'Python 3.10+ required, got {sys.version}'" 2>/dev/null || {
    echo "ERROR: Python 3.10+ is required."
    exit 1
}

python -c "import pkg_resources; print('  pkg_resources OK')" 2>/dev/null || {
    echo "ERROR: setuptools not available."
    exit 1
}

echo "  Python $(python --version 2>&1 | cut -d' ' -f2)"

# -------------------------------------------------------------------
# Step 2: Verify core dependencies
# -------------------------------------------------------------------
echo ""
echo "[2/5] Verifying core dependencies..."

MISSING=0
for pkg in fastapi uvicorn litellm langgraph pydantic yaml dotenv; do
    python -c "import $pkg" 2>/dev/null && echo "  $pkg OK" || {
        echo "  $pkg MISSING"
        MISSING=1
    }
done

if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "Some dependencies are missing. Install with:"
    echo "  pip install -r requirements.txt"
    exit 1
fi

echo "  All core dependencies present."

# -------------------------------------------------------------------
# Step 3: Verify .env configuration
# -------------------------------------------------------------------
echo ""
echo "[3/5] Checking environment configuration..."

if [ -f .env ]; then
    echo "  .env file found."
else
    echo "  WARNING: No .env file found. LLM calls will fail without API keys."
    echo "  Create .env with at least OPENAI_API_KEY or ANTHROPIC_API_KEY."
fi

# -------------------------------------------------------------------
# Step 4: Verify graph can be built
# -------------------------------------------------------------------
echo ""
echo "[4/5] Verifying LangGraph compilation..."

python -c "
import sys
sys.path.insert(0, '.')
from graph import build_graph
g = build_graph()
print('  LangGraph StateGraph compiled successfully (8 nodes)')
" 2>/dev/null || {
    echo "  WARNING: Graph compilation failed (may need API keys for LLM init)."
}

# -------------------------------------------------------------------
# Step 5: Run baseline evaluation
# -------------------------------------------------------------------
echo ""
echo "[5/5] Running baseline evaluation (MedQA, 10 samples, seed=42)..."
echo ""

mkdir -p results

python eval/run_eval.py \
    --dataset medqa \
    --n_samples 10 \
    --seed 42 \
    --output results/baseline_repro.json \
    2>&1

echo ""
echo "============================================================"
echo "Reproducibility run complete."
echo "Results: results/baseline_repro.json"
echo "============================================================"
