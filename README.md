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
  huggingface:
    provider_type: huggingface
    api_key: "${HF_TOKEN:-}"  # optional; authenticates Hub downloads when set

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
  max_workers: 8        # concurrent API requests; local models stay sequential
  rate_limit: 2.0       # optional max API calls per second
  api_sampling:
    sample_size: 200     # API models only; null runs the full dataset
    stratify_by: cefr    # preserve CEFR distribution
    seed: 0              # deterministic sample for comparable runs
```

Secrets can be kept in a local `.env` file next to the config file. `.env` is gitignored:

```bash
HF_TOKEN=hf_...
OPENCODE_API_KEY=...
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
  --models gpt4o,llama-70b,claude-opus
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
| `--sample-size` | config/all | Stratified sample size; useful for API models |
| `--stratify-by` | `cefr` | Metadata field used for stratified sampling |
| `--sample-seed` | `0` | Deterministic sampling seed |
| `--max-workers` | `8` for APIs | Concurrent API requests; local models run sequentially |
| `--rate-limit` | config/none | Max API calls per second |
| `--beta` | `0.5` | Beta for ERRANT F-score |

## Example Results

These smoke-test results use the BEA-2019 validation split, `seq2seq` strategy, a
200-sentence stratified sample, and default metric settings.

| Model | Backend | Runtime | ERRANT F | GLEU F | BERTScore F |
|-------|---------|--------:|---------:|-------:|------------:|
| `google/flan-t5-small` | HuggingFace local | 36.9s | 0.0741 | 0.3486 | 0.8013 |
| `grammarly/coedit-large` | HuggingFace local | 120.3s | 0.5010 | 0.6569 | 0.9235 |
| `qwen3.7-plus` | OpenCode Go | 797.8s | 0.4091 | 0.6511 | 0.9102 |

Per-CEFR breakdown:

| Model | CEFR | ERRANT F | GLEU F | BERTScore F |
|-------|------|---------:|-------:|------------:|
| `google/flan-t5-small` | A | 0.1228 | 0.0000 | 0.7727 |
| `google/flan-t5-small` | B | 0.0816 | 0.2867 | 0.8136 |
| `google/flan-t5-small` | C | 0.0185 | 0.4549 | 0.8191 |
| `google/flan-t5-small` | N | 0.0249 | 0.5032 | 0.8051 |
| `grammarly/coedit-large` | A | 0.4917 | 0.5585 | 0.8891 |
| `grammarly/coedit-large` | B | 0.5589 | 0.6176 | 0.9344 |
| `grammarly/coedit-large` | C | 0.4611 | 0.7337 | 0.9452 |
| `grammarly/coedit-large` | N | 0.4326 | 0.7415 | 0.9344 |
| `qwen3.7-plus` | A | 0.4728 | 0.6240 | 0.8898 |
| `qwen3.7-plus` | B | 0.4003 | 0.6034 | 0.9091 |
| `qwen3.7-plus` | C | 0.3577 | 0.7011 | 0.9236 |
| `qwen3.7-plus` | N | 0.3599 | 0.7016 | 0.9297 |

In this sample, `grammarly/coedit-large` is the strongest practical baseline: it
has the highest corpus-level scores and runs locally without API cost.

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
uv run pytest              # Tests
```
