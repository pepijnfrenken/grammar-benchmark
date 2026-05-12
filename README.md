# LLM Grammar Bench

Benchmark framework for evaluating AI models on grammatical error correction (GEC) using the [BEA-2019](https://www.cl.cam.ac.uk/research/nl/bea2019st/) shared task dataset (W&I + LOCNESS).

Evaluates models across CEFR proficiency levels (A, B, C, N) with ERRANT, GLEU, and BERTScore metrics.

## Quick Start

```bash
# Install with local model support
pip install llmgrammarbench[huggingface]

# Run a benchmark with a local model
llm-grammar-bench run --model hf:google/flan-t5-small --max-sentences 50
```

## Installation

Requires Python ≥ 3.11.

```bash
# Core only (no backends)
pip install llmgrammarbench

# With specific backends
pip install llmgrammarbench[openai]        # OpenAI (GPT-4o, GPT-4)
pip install llmgrammarbench[anthropic]     # Anthropic (Claude)
pip install llmgrammarbench[huggingface]   # Local models (T5, BART, Llama)
pip install llmgrammarbench[openrouter]    # OpenRouter gateway

# All backends
pip install llmgrammarbench[all]
```

## Usage

### Single model

```bash
# Local HuggingFace model
llm-grammar-bench run --model hf:google/flan-t5-small

# API model (requires env var)
llm-grammar-bench run --model openai:gpt-4o

# With options
llm-grammar-bench run \
  --model hf:grammarly/coedit-large \
  --max-sentences 100 \
  --strategy seq2seq \
  --metrics errant,gleu,bertscore \
  --output results/
```

### Configuration file

Define providers, models, and evaluation settings in a YAML config:

```yaml
# config.yaml
providers:
  openai:
    provider_type: openai
    api_key: "${OPENAI_API_KEY}"
  groq:
    provider_type: openai_compatible
    api_key: "${GROQ_API_KEY}"
    base_url: "https://api.groq.com/openai/v1"

models:
  gpt4o:
    provider: openai
    model: gpt-4o
    nickname: "GPT-4o"
  llama-70b:
    provider: groq
    model: llama-3.3-70b-versatile
    nickname: "Llama 3.3 70B"

evaluation:
  metrics: [errant, gleu, bertscore]
  beta: 0.5
  output_dir: results/
```

Use the config:

```bash
# Run a configured model by its key
llm-grammar-bench --config config.yaml run --model gpt4o

# List configured models
llm-grammar-bench --config config.yaml list-models
```

### Batch benchmarking

Run multiple models from a config file sequentially:

```bash
llm-grammar-bench --config config.yaml run \
  --models gpt4o,llama-70b,claude-opus \
  --max-sentences 200
```

Each model produces its own output file in the configured output directory.

### Model specification

Models can be referenced in three ways:

| Style | Example | Requires |
|-------|---------|----------|
| Config key | `--model gpt4o` | `--config` |
| Shorthand | `--model openai:gpt-4o` | Nothing |
| HuggingFace default | `--model google/flan-t5-small` | Nothing |

Shorthand prefixes: `hf:`, `openai:`, `anthropic:`, `openrouter:`

## Provider Types

Six provider types are supported, each mapping to a backend:

| Type | Backend | Required package |
|------|---------|-----------------|
| `openai` | OpenAI native API | `llmgrammarbench[openai]` |
| `anthropic` | Anthropic native API | `llmgrammarbench[anthropic]` |
| `huggingface` | Local transformers models | `llmgrammarbench[huggingface]` |
| `openrouter` | OpenRouter gateway | `llmgrammarbench[openrouter]` |
| `openai_compatible` | Any OpenAI-compatible endpoint | `llmgrammarbench[openai]` |
| `anthropic_compatible` | Any Anthropic-compatible endpoint | `llmgrammarbench[anthropic]` |

The `openai_compatible` type works with Groq, Together, Ollama, vLLM, and any service exposing an OpenAI-compatible `/v1/chat/completions` endpoint. The `anthropic_compatible` type works with AWS Bedrock, GCP Vertex, and any service exposing an Anthropic-compatible `/v1/messages` endpoint.

Four built-in providers (`openai`, `anthropic`, `openrouter`, `huggingface`) are always available — no need to declare them in the config unless overriding their base URL or API key.

## Correction Strategies

| Strategy | Description |
|----------|-------------|
| `seq2seq` | Direct correction — sends text, returns corrected version |
| `edit_based` | Model outputs JSON edit spans `[{start, end, text}]` applied to source |
| `few_shot` | Prepends N example correction pairs before the target sentence |

Strategy usage:

```bash
llm-grammar-bench run --model hf:google/flan-t5-small --strategy edit_based
```

## Metrics

Three metrics from [`gec-metrics`](https://github.com/gotutiyan/gec-metrics):

| Metric | Measures |
|--------|----------|
| **ERRANT** | Edit-based accuracy — precision/recall of error corrections |
| **GLEU** | N-gram overlap between hypothesis and references |
| **BERTScore** | Semantic similarity using BERT embeddings |

All metrics compute corpus-level F-scores and per-CEFR breakdowns.

## Dataset

Uses the [BEA-2019 W&I + LOCNESS](https://huggingface.co/datasets/jvamvas/peer_wi-locness) dataset (2,819 validation sentences) with CEFR level annotations:

| Level | Sentences | Description |
|-------|-----------|-------------|
| A | 819 | Beginner |
| B | 922 | Intermediate |
| C | 571 | Advanced |
| N | 507 | Native (LOCNESS) |

Filter by CEFR level in config:

```yaml
datasets:
  bea2019:
    type: bea2019
    split: validation
    cefr_filter: [A, B]  # Only beginner and intermediate
```

## CLI Reference

```
Usage: llm-grammar-bench [OPTIONS] COMMAND [ARGS]...

Options:
  --config PATH  Path to YAML config file
  --version      Show version
  --help         Show help

Commands:
  run            Run a benchmark with one or more models
  list-models    List available backends and configured models
  list-datasets  List available datasets
```

### `run` options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | — | Single model key or shorthand spec |
| `--models` | — | Comma-separated model keys (requires `--config`) |
| `--dataset` | `bea2019` | Dataset name |
| `--split` | `validation` | Dataset split |
| `--strategy` | `seq2seq` | Correction strategy |
| `--metrics` | `errant,gleu,bertscore` | Metrics to compute |
| `--output` | `results/` | Output directory or file path |
| `--max-sentences` | all | Limit sentences for quick tests |
| `--beta` | `0.5` | Beta for ERRANT F-score |

## Example Results

Batch benchmark comparing two models on 200 A-level sentences:

```
$ llm-grammar-bench --config configs/default.yaml run --models t5-small,coedit-large --max-sentences 200

  Model 1/2: t5-small
  ===================
  Corpus-level scores:
    errant: F=0.0902
    gleu: F=0.1430
    bertscore: F=0.6430

  Model 2/2: coedit-large
  =======================
  Corpus-level scores:
    errant: F=0.2431
    gleu: F=0.3895
    bertscore: F=0.8810
```

## Development

```bash
git clone <repo>
cd llmGrammarBench

# Install dev dependencies
uv sync --all-extras

# Run checks
uv run ty check .          # Type checking
uv run ruff check .        # Linting
uv run ruff format --check .  # Formatting
uv run pytest              # Tests (99 tests, 73% coverage)
```
