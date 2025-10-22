# LLM Integration Guide

## Overview

This guide explains how to integrate LLM models (OpenAI, Anthropic, local models) for superior image and table extraction when OCR quality is insufficient.

## Why Use LLM Instead of OCR?

**OCR Limitations:**

- Poor quality with complex layouts
- Struggles with charts, diagrams, infographics
- Misses context and relationships
- Bad at table structure recognition

**LLM Advantages:**

- ✅ Understands visual context
- ✅ Better chart/diagram interpretation
- ✅ Preserves table structure
- ✅ Can extract insights and relationships
- ✅ Handles multiple languages naturally

## Architecture Changes

### Current Flow (OCR-based)

```
PDF → Extract Images → Tesseract OCR → Raw Text
PDF → Extract Tables → Camelot → Structured Data
```

### Proposed Flow (LLM-based)

```
PDF → Extract Images → LLM Vision API → Structured Markdown
PDF → Render Page → LLM Vision API → Full Page Analysis
```

## Implementation Guide

### 1. Add Dependencies to `Cargo.toml`

```toml
[dependencies]
# Existing dependencies...
reqwest = { version = "0.11", features = ["json", "multipart"] }
base64 = "0.22"
serde_json = "1"

[features]
default = []
llm-vision = ["reqwest", "base64"]  # Optional feature flag
```

### 2. Create LLM Module (`src/llm.rs`)

````rust
//! LLM Vision Integration for Image and Table Extraction
//!
//! Supports: OpenAI GPT-4 Vision, Anthropic Claude Vision, Local models

use anyhow::{bail, Result};
use base64::{engine::general_purpose, Engine};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokio::fs;

#[derive(Debug, Clone)]
pub struct LLMConfig {
    pub provider: LLMProvider,
    pub api_key: String,
    pub model: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub image_prompt: String,
    pub table_prompt: String,
}

#[derive(Debug, Clone)]
pub enum LLMProvider {
    OpenAI,
    Anthropic,
    Local { endpoint: String },
}

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            provider: LLMProvider::OpenAI,
            api_key: std::env::var("OPENAI_API_KEY").unwrap_or_default(),
            model: std::env::var("LLM_MODEL")
                .unwrap_or_else(|_| "gpt-4o".to_string()),
            max_tokens: 4096,
            temperature: 0.0, // Deterministic for data extraction
            image_prompt: default_image_prompt(),
            table_prompt: default_table_prompt(),
        }
    }
}

/// Extract text and structure from image using LLM vision
pub async fn analyze_image_with_llm(
    image_path: &Path,
    config: &LLMConfig,
) -> Result<ImageAnalysis> {
    // Read and encode image
    let image_data = fs::read(image_path).await?;
    let base64_image = general_purpose::STANDARD.encode(&image_data);

    match &config.provider {
        LLMProvider::OpenAI => analyze_with_openai(&base64_image, config).await,
        LLMProvider::Anthropic => analyze_with_anthropic(&base64_image, config).await,
        LLMProvider::Local { endpoint } => {
            analyze_with_local(&base64_image, endpoint, config).await
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ImageAnalysis {
    pub content_type: ContentType,
    pub markdown_text: String,
    pub structured_data: Option<serde_json::Value>,
    pub confidence: f32,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum ContentType {
    Decorative,
    Simple,
    Table,
    Chart,
    Diagram,
    Mixed,
}

async fn analyze_with_openai(
    base64_image: &str,
    config: &LLMConfig,
) -> Result<ImageAnalysis> {
    let client = reqwest::Client::new();

    let payload = serde_json::json!({
        "model": config.model,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": config.image_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": format!("data:image/jpeg;base64,{}", base64_image)
                    }
                }
            ]
        }],
        "max_tokens": config.max_tokens,
        "temperature": config.temperature
    });

    let response = client
        .post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", config.api_key))
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .await?;

    if !response.status().is_success() {
        let error_text = response.text().await?;
        bail!("OpenAI API error: {}", error_text);
    }

    let result: serde_json::Value = response.json().await?;
    let markdown_text = result["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();

    // Parse the response to extract content type and structure
    parse_llm_response(&markdown_text)
}

async fn analyze_with_anthropic(
    base64_image: &str,
    config: &LLMConfig,
) -> Result<ImageAnalysis> {
    let client = reqwest::Client::new();

    let payload = serde_json::json!({
        "model": config.model,
        "max_tokens": config.max_tokens,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image
                    }
                },
                {
                    "type": "text",
                    "text": config.image_prompt
                }
            ]
        }]
    });

    let response = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", &config.api_key)
        .header("anthropic-version", "2023-06-01")
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .await?;

    if !response.status().is_success() {
        let error_text = response.text().await?;
        bail!("Anthropic API error: {}", error_text);
    }

    let result: serde_json::Value = response.json().await?;
    let markdown_text = result["content"][0]["text"]
        .as_str()
        .unwrap_or("")
        .to_string();

    parse_llm_response(&markdown_text)
}

async fn analyze_with_local(
    base64_image: &str,
    endpoint: &str,
    config: &LLMConfig,
) -> Result<ImageAnalysis> {
    // For local models (LLaVA, Ollama, etc.)
    let client = reqwest::Client::new();

    let payload = serde_json::json!({
        "model": config.model,
        "prompt": config.image_prompt,
        "images": [base64_image],
        "stream": false
    });

    let response = client
        .post(endpoint)
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .await?;

    if !response.status().is_success() {
        let error_text = response.text().await?;
        bail!("Local LLM API error: {}", error_text);
    }

    let result: serde_json::Value = response.json().await?;
    let markdown_text = result["response"]
        .as_str()
        .unwrap_or("")
        .to_string();

    parse_llm_response(&markdown_text)
}

fn parse_llm_response(markdown: &str) -> Result<ImageAnalysis> {
    // Parse the structured markdown response
    let content_type = if markdown.contains("[DECORATIVE]") {
        ContentType::Decorative
    } else if markdown.contains("[TABLE]") {
        ContentType::Table
    } else if markdown.contains("[CHART:") {
        ContentType::Chart
    } else if markdown.contains("[DIAGRAM:") {
        ContentType::Diagram
    } else if markdown.contains("[MIXED]") {
        ContentType::Mixed
    } else {
        ContentType::Simple
    };

    // Extract structured data if it's a table
    let structured_data = if matches!(content_type, ContentType::Table) {
        extract_table_from_markdown(markdown).ok()
    } else {
        None
    };

    Ok(ImageAnalysis {
        content_type,
        markdown_text: markdown.to_string(),
        structured_data,
        confidence: 0.9, // Could be extracted from response metadata
    })
}

fn extract_table_from_markdown(markdown: &str) -> Result<serde_json::Value> {
    // Parse markdown table into structured JSON
    // This is simplified - you'd want a proper markdown table parser
    let lines: Vec<&str> = markdown.lines()
        .skip_while(|l| !l.starts_with("|"))
        .take_while(|l| l.starts_with("|") || l.is_empty())
        .filter(|l| !l.contains("---"))  // Skip separator line
        .collect();

    if lines.is_empty() {
        bail!("No table found in markdown");
    }

    let headers: Vec<String> = lines[0]
        .split('|')
        .skip(1)
        .take_while(|s| !s.is_empty())
        .map(|s| s.trim().to_string())
        .collect();

    let rows: Vec<Vec<String>> = lines.iter()
        .skip(1)
        .map(|line| {
            line.split('|')
                .skip(1)
                .take_while(|s| !s.is_empty())
                .map(|s| s.trim().to_string())
                .collect()
        })
        .collect();

    Ok(serde_json::json!({
        "headers": headers,
        "rows": rows
    }))
}

fn default_image_prompt() -> String {
    r#"Analyze this image for document search and data extraction.

## Output Format Requirements:
1. Start with content type: [DECORATIVE], [SIMPLE], [TABLE], [CHART: type], [DIAGRAM: type], or [MIXED]
2. For trivial/decorative images: One-line description only
3. For data-rich content: Follow structured format below

## For Tables:
**Content Type:** [TABLE]
**Title:** [Extract title if present]

```markdown
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
````

## For Charts:

**Content Type:** [CHART: Bar/Line/Pie]
**Title:** [Chart title]
**X-Axis:** [Label]
**Y-Axis:** [Label]

**Data:**

- Series 1: [Values]
- Series 2: [Values]

## For Diagrams:

**Content Type:** [DIAGRAM: Flowchart/Network/etc]
**Description:** [Structure and relationships]
**Text Elements:** [All text labels]

## Guidelines:

- Extract ALL text exactly as shown
- Preserve relationships and structure
- Mark unclear elements as [unclear]
- Include units, labels, and legends
- For simple images, be brief
- For data tables/charts, be comprehensive

Output ONLY the structured analysis in markdown format."#.to_string()
}

fn default_table_prompt() -> String {
r#"Extract this table with perfect structure preservation.

## Requirements:

1. Identify all column headers
2. Extract all rows maintaining order
3. Note any merged cells
4. Preserve data types (numbers, text, dates)
5. Include any footnotes or captions

## Output Format:

```markdown
| Header 1 | Header 2 | Header 3 |
| -------- | -------- | -------- |
| Cell 1   | Cell 2   | Cell 3   |
```

**Notes:** [Any special formatting or merged cells]
**Source:** [Any caption or source information]

Be precise. Extract every cell exactly."#.to_string()
}

````

### 3. Modify `src/lib.rs` to Support LLM

Add at the top of `lib.rs`:

```rust
#[cfg(feature = "llm-vision")]
pub mod llm;

#[cfg(feature = "llm-vision")]
use llm::{analyze_image_with_llm, LLMConfig};
````

Add to `ParserConfig`:

```rust
pub struct ParserConfig {
    // ... existing fields ...

    #[cfg(feature = "llm-vision")]
    pub use_llm_vision: bool,

    #[cfg(feature = "llm-vision")]
    pub llm_config: Option<LLMConfig>,
}
```

Update `process_page` function to use LLM when enabled:

```rust
// 4) OCR embedded images OR use LLM vision
if config.ocr_images {
    if let Ok(img_paths) = extract_embedded_images(pdf, page).await {
        for img in &img_paths {
            #[cfg(feature = "llm-vision")]
            let txt = if config.use_llm_vision && config.llm_config.is_some() {
                // Use LLM instead of OCR
                match llm::analyze_image_with_llm(img, config.llm_config.as_ref().unwrap()).await {
                    Ok(analysis) => analysis.markdown_text,
                    Err(_) => {
                        // Fallback to OCR
                        ocr_image_file(img, &config.ocr_lang, &config.ocr_psm).await?
                    }
                }
            } else {
                ocr_image_file(img, &config.ocr_lang, &config.ocr_psm).await?
            };

            #[cfg(not(feature = "llm-vision"))]
            let txt = ocr_image_file(img, &config.ocr_lang, &config.ocr_psm).await?;

            // ... rest of processing ...
        }
    }
}
```

### 4. Environment Variables Configuration

Add to your `.env` or set before running:

```bash
# LLM Configuration
USE_LLM_VISION=1                          # Enable LLM vision (default: 0)
LLM_PROVIDER=openai                       # openai, anthropic, or local
OPENAI_API_KEY=sk-...                     # Your OpenAI API key
LLM_MODEL=gpt-4o                          # Model to use
LLM_MAX_TOKENS=4096                       # Max response tokens
LLM_TEMPERATURE=0.0                       # 0=deterministic, 1=creative

# For Anthropic
ANTHROPIC_API_KEY=sk-ant-...
LLM_MODEL=claude-3-5-sonnet-20241022

# For local models (Ollama, LLaVA, etc.)
LLM_PROVIDER=local
LLM_ENDPOINT=http://localhost:11434/api/generate
LLM_MODEL=llava:13b
```

### 5. Build and Run with LLM Support

```bash
# Build with LLM feature
cargo build --release --features llm-vision

# Run with environment variables
USE_LLM_VISION=1 OPENAI_API_KEY=sk-... cargo run --features llm-vision --bin parser test_pdfs/example.pdf
```

## Advanced: Custom Prompts

Create `prompts/image_analysis.txt`:

````markdown
# Image Analysis for Document Extraction

## Classification (Choose ONE):

- **[SKIP]**: Decorative element, logo, or page number
- **[TEXT]**: Simple paragraph or caption
- **[TABLE]**: Tabular data
- **[CHART]**: Graph, plot, or visualization
- **[DIAGRAM]**: Flowchart, architecture, or schematic
- **[FORM]**: Form fields or structured input

## For [SKIP] and [TEXT]:

Provide brief description (1-2 sentences max).

## For [TABLE]:

### Format:

```markdown
**Title:** [Table caption/title]
**Description:** [Brief context]

| Column 1 | Column 2 | Column 3 |
| -------- | -------- | -------- |
| Value 1  | Value 2  | Value 3  |

**Notes:**

- [Footnotes]
- [Data source]
- [Special formatting]
```
````

## For [CHART]:

### Format:

```markdown
**Chart Type:** [Bar/Line/Pie/Scatter/Area/etc.]
**Title:** [Chart title]
**Description:** [What the chart shows]

**Axes:**

- X-Axis: [Label] ([Unit])
- Y-Axis: [Label] ([Unit])

**Data Series:**

1. [Series Name]
   - [Data point 1]: [Value] [Unit]
   - [Data point 2]: [Value] [Unit]

**Key Insights:**

- [Trend 1]
- [Notable value]

**Legend:** [If applicable]
**Source:** [If shown]
```

## For [DIAGRAM]:

### Format:

```markdown
**Diagram Type:** [Flowchart/Network/Architecture/etc.]
**Title:** [Diagram title]
**Description:** [Purpose and structure]

**Components:**

1. [Component 1]: [Description]
2. [Component 2]: [Description]

**Relationships:**

- [Component A] → [Component B]: [Relationship type]

**Text Labels:** [All visible text]
**Source:** [If shown]
```

## Quality Requirements:

1. ✅ Extract ALL visible text exactly
2. ✅ Preserve numerical precision
3. ✅ Maintain structure and hierarchy
4. ✅ Include units, labels, legends
5. ✅ Mark unclear elements: [unclear: partial_text...]
6. ✅ Use consistent formatting
7. ❌ Do NOT interpret or summarize data
8. ❌ Do NOT add information not in image

## Output:

- Start with classification tag
- Follow the format for that type
- Use markdown for structure
- Be precise and complete

````

Load custom prompt:

```rust
let custom_prompt = tokio::fs::read_to_string("prompts/image_analysis.txt").await?;
config.llm_config.as_mut().unwrap().image_prompt = custom_prompt;
````

## Cost Optimization

LLM vision APIs can be expensive. Optimize with:

### 1. Selective Processing

```rust
// Only use LLM for images that OCR fails on
let ocr_result = ocr_image_file(img, lang, psm).await?;
if ocr_result.len() < 50 || ocr_result.matches(char::is_alphanumeric).count() < 10 {
    // OCR quality poor, use LLM
    let llm_result = analyze_image_with_llm(img, llm_config).await?;
    final_text = llm_result.markdown_text;
} else {
    // OCR quality good, use it
    final_text = ocr_result;
}
```

### 2. Image Preprocessing

```rust
// Resize large images before sending to LLM
async fn optimize_image_for_llm(path: &Path) -> Result<Vec<u8>> {
    let img = image::open(path)?;

    // Resize if larger than 2048px
    let img = if img.width() > 2048 || img.height() > 2048 {
        img.resize(2048, 2048, image::imageops::FilterType::Lanczos3)
    } else {
        img
    };

    // Convert to JPEG with compression
    let mut buffer = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut buffer);
    img.write_to(&mut cursor, image::ImageFormat::Jpeg)?;

    Ok(buffer)
}
```

### 3. Caching

```rust
use std::collections::HashMap;
use sha2::{Sha256, Digest};

// Cache LLM responses by image hash
static LLM_CACHE: Lazy<Mutex<HashMap<String, ImageAnalysis>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

async fn analyze_image_cached(path: &Path, config: &LLMConfig) -> Result<ImageAnalysis> {
    // Calculate image hash
    let img_data = fs::read(path).await?;
    let hash = format!("{:x}", Sha256::digest(&img_data));

    // Check cache
    if let Some(cached) = LLM_CACHE.lock().unwrap().get(&hash) {
        return Ok(cached.clone());
    }

    // Call LLM
    let result = analyze_image_with_llm(path, config).await?;

    // Store in cache
    LLM_CACHE.lock().unwrap().insert(hash, result.clone());

    Ok(result)
}
```

## Performance Comparison

| Method            | Speed       | Quality              | Cost      | Best For                           |
| ----------------- | ----------- | -------------------- | --------- | ---------------------------------- |
| **Tesseract OCR** | ⚡⚡⚡ Fast | ⭐⭐ Good            | 💰 Free   | Clean text, scanned documents      |
| **GPT-4o Vision** | ⚡ Slow     | ⭐⭐⭐⭐⭐ Excellent | 💰💰💰 $$ | Complex charts, diagrams, tables   |
| **Claude Vision** | ⚡ Slow     | ⭐⭐⭐⭐⭐ Excellent | 💰💰💰 $$ | Detailed analysis, structured data |
| **Local LLaVA**   | ⚡⚡ Medium | ⭐⭐⭐ Good          | 💰 Free   | Privacy-sensitive documents        |

## Recommended Strategy

```rust
// Hybrid approach: Fast OCR first, LLM fallback for complex content
pub async fn smart_image_extraction(
    image_path: &Path,
    ocr_config: &OCRConfig,
    llm_config: &LLMConfig,
) -> Result<String> {
    // Step 1: Try OCR (fast)
    let ocr_text = ocr_image_file(image_path, &ocr_config.lang, &ocr_config.psm).await?;

    // Step 2: Quality check
    let quality_score = assess_ocr_quality(&ocr_text);

    if quality_score > 0.7 {
        // OCR quality is good
        return Ok(ocr_text);
    }

    // Step 3: Check if image is data-rich (worth LLM cost)
    if is_likely_data_rich(image_path).await? {
        // Use LLM for better extraction
        let analysis = analyze_image_with_llm(image_path, llm_config).await?;
        return Ok(analysis.markdown_text);
    }

    // Step 4: Fallback to OCR even if quality is poor
    Ok(ocr_text)
}

fn assess_ocr_quality(text: &str) -> f32 {
    let total_chars = text.len() as f32;
    if total_chars == 0.0 {
        return 0.0;
    }

    let alphanum_count = text.chars().filter(|c| c.is_alphanumeric()).count() as f32;
    let word_count = text.split_whitespace().count() as f32;

    // Quality score based on alphanumeric ratio and word count
    let alphanum_ratio = alphanum_count / total_chars;
    let has_words = (word_count > 3.0) as u8 as f32;

    (alphanum_ratio * 0.7) + (has_words * 0.3)
}

async fn is_likely_data_rich(image_path: &Path) -> Result<bool> {
    // Use image analysis to detect tables, charts, diagrams
    // This could be a simple CNN classifier or heuristics
    let img = image::open(image_path)?;

    // Heuristic: Images with high contrast and structured elements
    // are likely data-rich
    let (width, height) = (img.width(), img.height());

    // Large images are more likely to be data-rich
    Ok(width > 800 && height > 600)
}
```

## Testing

```bash
# Test with OpenAI
USE_LLM_VISION=1 OPENAI_API_KEY=sk-... cargo run --features llm-vision --bin parser test_pdfs/difficult-charts.pdf

# Compare OCR vs LLM
cargo run --bin parser test_pdfs/example.pdf > ocr_output.json
USE_LLM_VISION=1 cargo run --features llm-vision --bin parser test_pdfs/example.pdf > llm_output.json
diff ocr_output.json llm_output.json
```

## Summary

**To integrate LLM vision:**

1. ✅ Add `reqwest` and `base64` to dependencies
2. ✅ Create `src/llm.rs` module (code provided above)
3. ✅ Modify `src/lib.rs` to support `use_llm_vision` flag
4. ✅ Set environment variables: `USE_LLM_VISION=1`, `OPENAI_API_KEY=...`
5. ✅ Build with `--features llm-vision`
6. ✅ Use hybrid approach for cost optimization

**When to use LLM:**

- Complex charts and infographics
- Multi-column tables with poor OCR
- Diagrams with relationships
- Documents where structure matters

**When to use OCR:**

- Clean typed/printed text
- Scanned documents
- Cost-sensitive applications
- Real-time processing needs

The hybrid approach gives you the best of both worlds! 🚀
