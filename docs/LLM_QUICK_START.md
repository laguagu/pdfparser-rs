# LLM Vision - Quick Start

## 5-Minute Setup

### 1. Get API Key

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# OR Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 2. Enable Feature

```bash
# Edit Cargo.toml - add to [dependencies]:
reqwest = { version = "0.11", features = ["json"] }
base64 = "0.22"
```

### 3. Set Environment

```bash
USE_LLM_VISION=1
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
```

### 4. Run

```bash
cargo run --features llm-vision --bin parser your.pdf
```

## When to Use LLM vs OCR

### Use OCR (Fast, Free):

- ✅ Clean typed text
- ✅ Simple scanned documents
- ✅ High volume processing
- ✅ Real-time needs

### Use LLM (Slow, Paid, Better Quality):

- ✅ Complex charts/graphs
- ✅ Multi-column tables
- ✅ Infographics
- ✅ Diagram relationships
- ✅ When accuracy >> cost

## Cost Estimate

**OpenAI GPT-4o Vision:**

- ~$0.01 per image (1024x1024)
- 100 pages with 5 images each = ~$5

**Anthropic Claude Vision:**

- Similar pricing

**Local Models (Free):**

- LLaVA, Ollama vision models
- No API costs but needs GPU

## Best Practice: Hybrid

```bash
# Try OCR first, use LLM only for complex content
OCR_OVER_TEXT=1     # Use fast OCR
USE_LLM_VISION=1    # Enable LLM as fallback
LLM_THRESHOLD=0.7   # Only use LLM if OCR quality < 0.7
```

## Troubleshooting

**API Key Error:**

```bash
echo $OPENAI_API_KEY  # Check it's set
```

**Rate Limits:**

```bash
MAX_CONCURRENCY=1  # Process one page at a time
```

**High Costs:**

```bash
USE_LLM_VISION=0          # Disable for testing
LLM_ONLY_COMPLEX=1        # Only use for complex images
```

## See Full Guide

→ `docs/LLM_INTEGRATION.md` for complete documentation
