//! PDF Parser Library - Shared parsing logic
//!
//! This library contains the core PDF parsing functionality that can be used by:
//! - API server (main.rs)
//! - CLI tool (bin/parser.rs)

use anyhow::{bail, Result};
use futures::stream::{self, StreamExt};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet, fmt::Write as FmtWrite, path::Path, process::Stdio, sync::Arc,
};
use tokio::{fs, process::Command, sync::Semaphore};

// Note: roxmltree used only when SVG_TEXT=1
// use roxmltree imported inline in extract_svg_text function

// ============================================================================
// Configuration Constants - Default values (overridable via environment)
// ============================================================================

/// Maximum number of pages to process concurrently
/// Env: MAX_CONCURRENCY (default: 4 for single-user speed)
/// Note: Lower to 2 if running multiple containers or limited memory
const DEFAULT_MAX_CONCURRENCY: usize = 4;

/// Minimum characters for text layer to be considered valid
/// Env: TEXT_MIN_CHARS (default: 20)
const DEFAULT_TEXT_MIN_CHARS: usize = 20;

/// Run OCR on page even when text layer exists (captures charts/diagrams)
/// Env: OCR_OVER_TEXT (default: true for maximum accuracy)
const DEFAULT_OCR_OVER_TEXT: bool = true;

/// Extract and OCR embedded images separately
/// Env: OCR_IMAGES (default: true)
const DEFAULT_OCR_IMAGES: bool = true;

/// Extract tables using Camelot (Python library)
/// Env: EXTRACT_TABLES (default: true)
const DEFAULT_EXTRACT_TABLES: bool = true;

/// Extract SVG vector text (slower, usually not needed)
/// Env: SVG_TEXT (default: false)
const DEFAULT_SVG_TEXT: bool = false;

/// DPI for rendering PDF pages to images for OCR (higher = better quality, slower)
/// Env: OCR_DPI (default: 300)
const DEFAULT_OCR_DPI: &str = "300";

/// Tesseract language codes (comma-separated, e.g., "fin+eng")
/// Common codes: eng, rus, fin, deu, fra, spa, ita, por, nld, swe, nor, dan
/// Env: OCR_LANG (default: fin+eng)
const DEFAULT_OCR_LANG: &str = "fin+eng";

/// Tesseract page segmentation mode
/// 3 = Fully automatic page segmentation (default)
/// 6 = Uniform block of text
/// 11 = Sparse text, find as much text as possible
/// Env: OCR_PSM (default: 3)
const DEFAULT_OCR_PSM: &str = "3";

// ============================================================================
// Public Types (exposed to users of this library)
// ============================================================================

#[derive(Serialize)]
pub struct TableOut {
    pub table_index: usize,
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<String>>,
}

#[derive(Serialize)]
pub struct PageOut {
    pub page: u32,
    pub source: &'static str,
    pub text: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tables: Vec<TableOut>,
}

#[derive(Serialize)]
pub struct ParseResult {
    pub pages: Vec<PageOut>,
}

// ============================================================================
// Markdown Rendering
// ============================================================================

/// Render a parsed PDF into a Markdown string optimized for LLM consumption.
/// Output is clean, structured, and preserves all text and table data.
pub fn render_markdown(result: &ParseResult) -> String {
    let mut buf = String::new();

    for (page_idx, page) in result.pages.iter().enumerate() {
        // Page separator
        if page_idx > 0 {
            buf.push_str("\n---\n\n");
        }

        // Page heading
        let _ = writeln!(&mut buf, "## Page {}\n", page.page);

        // Main text content
        if page.text.trim().is_empty() {
            buf.push_str("_No text content_\n\n");
        } else {
            buf.push_str(page.text.trim_end());
            buf.push_str("\n\n");
        }

        // Tables
        if !page.tables.is_empty() {
            for (table_idx, table) in page.tables.iter().enumerate() {
                let _ = writeln!(
                    &mut buf,
                    "### Table {}\n",
                    table_idx + 1,
                );

                buf.push_str(&table_to_markdown(table));
                buf.push('\n');
            }
        }
    }

    // Ensure single newline at end
    buf.trim_end().to_string() + "\n"
}

// ============================================================================
// Configuration
// ============================================================================

pub struct ParserConfig {
    pub max_concurrency: usize,
    pub text_min_chars: usize,
    pub ocr_over_text: bool,
    pub ocr_images: bool,
    pub extract_tables: bool,
    pub svg_text: bool,
    pub ocr_dpi: String,
    pub ocr_lang: String,
    pub ocr_psm: String,
}

impl Default for ParserConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

impl ParserConfig {
    /// Load configuration from environment variables with fallback to defaults
    pub fn from_env() -> Self {
        Self {
            max_concurrency: env_usize("MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY),
            text_min_chars: env_usize("TEXT_MIN_CHARS", DEFAULT_TEXT_MIN_CHARS),
            ocr_over_text: env_bool("OCR_OVER_TEXT", DEFAULT_OCR_OVER_TEXT),
            ocr_images: env_bool("OCR_IMAGES", DEFAULT_OCR_IMAGES),
            extract_tables: env_bool("EXTRACT_TABLES", DEFAULT_EXTRACT_TABLES),
            svg_text: env_bool("SVG_TEXT", DEFAULT_SVG_TEXT),
            ocr_dpi: env_string("OCR_DPI", DEFAULT_OCR_DPI),
            ocr_lang: env_string("OCR_LANG", DEFAULT_OCR_LANG),
            ocr_psm: env_string("OCR_PSM", DEFAULT_OCR_PSM),
        }
    }

    /// Override OCR language at runtime (useful for API query parameters)
    pub fn with_lang(mut self, lang: String) -> Self {
        self.ocr_lang = lang;
        self
    }

    /// Override OCR DPI at runtime (useful for API query parameters)
    pub fn with_dpi(mut self, dpi: String) -> Self {
        self.ocr_dpi = dpi;
        self
    }
}

// ============================================================================
// Environment Variable Helpers
// ============================================================================

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_bool(key: &str, default: bool) -> bool {
    std::env::var(key)
        .ok()
        .map(|v| matches!(v.to_lowercase().as_str(), "true" | "1" | "yes" | "on"))
        .unwrap_or(default)
}

fn env_string(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

// ============================================================================
// Main Parsing Function
// ============================================================================

/// Parse a PDF file and extract text, images, and tables
pub async fn parse_pdf(pdf_path: &Path, config: ParserConfig) -> Result<ParseResult> {
    // Get page count
    let pages = pdf_pages(pdf_path).await?;

    // Process pages with bounded concurrency
    let pdf_arc = Arc::new(pdf_path.to_path_buf());
    let sem = Arc::new(Semaphore::new(config.max_concurrency));

    let mut out_pages = stream::iter(1..=pages)
        .map(|page| {
            let pdf = pdf_arc.clone();
            let sem = sem.clone();
            let cfg = ParserConfig {
                max_concurrency: config.max_concurrency,
                text_min_chars: config.text_min_chars,
                ocr_over_text: config.ocr_over_text,
                ocr_images: config.ocr_images,
                extract_tables: config.extract_tables,
                svg_text: config.svg_text,
                ocr_dpi: config.ocr_dpi.clone(),
                ocr_lang: config.ocr_lang.clone(),
                ocr_psm: config.ocr_psm.clone(),
            };

            async move {
                let _permit = sem.acquire().await.expect("semaphore");
                process_page(&pdf, page, &cfg).await
            }
        })
        .buffer_unordered(config.max_concurrency)
        .collect::<Vec<_>>()
        .await;

    out_pages.sort_by_key(|p| p.page);
    Ok(ParseResult { pages: out_pages })
}

// ============================================================================
// Internal Functions
// ============================================================================

async fn process_page(pdf: &Path, page: u32, config: &ParserConfig) -> PageOut {
    // 1) Extract text layer
    let base_text = pdftotext_page(pdf, page).await.unwrap_or_default();
    let has_text = base_text.chars().filter(|c| !c.is_whitespace()).count() > config.text_min_chars;

    let mut final_text: String;
    let mut src: &'static str = "text";

    if has_text {
        final_text = base_text.clone();
    } else {
        // No text → OCR full page
        if let Ok(ocr_txt) = ocr_page_bitmap(
            pdf,
            page,
            &config.ocr_dpi,
            &config.ocr_lang,
            &config.ocr_psm,
        )
        .await
        {
            final_text = ocr_txt;
            src = "ocr";
        } else {
            final_text = base_text;
            src = "ocr";
        }
    }

    // 2) OCR over text layer (captures charts/diagrams)
    if config.ocr_over_text && has_text {
        if let Ok(ocr_txt2) = ocr_page_bitmap(
            pdf,
            page,
            &config.ocr_dpi,
            &config.ocr_lang,
            &config.ocr_psm,
        )
        .await
        {
            let (merged, added) = merge_and_count(&final_text, &ocr_txt2);
            if added > 0 {
                final_text = merged;
                src = "text+ocr";
            }
        }
    }

    // 3) SVG vector text extraction (if enabled)
    if config.svg_text {
        if let Ok(svg_txt) = extract_svg_text(pdf, page).await {
            let (merged, added) = merge_and_count(&final_text, &svg_txt);
            if added > 0 {
                final_text = merged;
                if src == "text" {
                    src = "text+ocr";
                }
            }
        }
    }

    // 4) OCR embedded images
    if config.ocr_images {
        if let Ok(img_paths) = extract_embedded_images(pdf, page).await {
            for img in &img_paths {
                if let Ok(txt) = ocr_image_file(img, &config.ocr_lang, &config.ocr_psm).await {
                    let trimmed = txt.trim();
                    if !trimmed.is_empty() {
                        let (merged, added) = merge_and_count(&final_text, &txt);
                        if added > 0 {
                            final_text = merged;
                            if src == "text" {
                                src = "text+ocr";
                            }
                        }
                    }
                }
            }
            for img in img_paths {
                let _ = fs::remove_file(&img).await;
            }
        }
    }

    // 5) Extract tables
    let mut tables = Vec::new();
    if config.extract_tables {
        if let Ok(extracted_tables) = extract_tables_from_page(pdf, page).await {
            tables = extracted_tables;
        }
    }

    // 6) Normalize whitespace for LLM
    final_text = normalize_whitespace(&final_text);

    PageOut {
        page,
        source: src,
        text: final_text,
        tables,
    }
}

// Text merging and deduplication
fn merge_and_count(base: &str, add: &str) -> (String, usize) {
    let mut seen: HashSet<String> = base
        .lines()
        .map(normalize_line)
        .filter(|l| !l.is_empty())
        .collect();
    let mut out = String::new();
    out.push_str(base.trim_end());
    let mut added = 0usize;

    for line in add.lines() {
        let key = normalize_line(line);
        if key.is_empty() {
            continue;
        }
        if !seen.contains(&key) {
            let ln = if out.is_empty() {
                line.to_string()
            } else {
                format!("\n{}", line)
            };
            added += ln.len();
            out.push_str(&ln);
            seen.insert(key);
        }
    }
    (out, added)
}

fn normalize_line(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Normalize whitespace for LLM consumption - aggressively removes layout whitespace
fn normalize_whitespace(text: &str) -> String {
    // Compile regex once (in real code, use lazy_static or once_cell for production)
    let multi_space = Regex::new(r" {2,}").unwrap();

    let mut result: Vec<String> = Vec::new();
    let mut block: Vec<String> = Vec::new();

    for line in text.lines() {
        let trimmed_end = line.trim_end();
        if trimmed_end.is_empty() {
            if !block.is_empty() {
                result.extend(dedent_block(&block));
                block.clear();
            }

            // Only add one blank line between blocks
            if !result.last().is_some_and(|l| l.is_empty()) {
                result.push(String::new());
            }
        } else {
            // Aggressively collapse multiple spaces into single space
            let cleaned_line = multi_space.replace_all(trimmed_end, " ").to_string();
            block.push(cleaned_line);
        }
    }

    if !block.is_empty() {
        result.extend(dedent_block(&block));
    }

    // Remove trailing blank lines
    while result.last().is_some_and(|l| l.is_empty()) {
        result.pop();
    }

    // Remove leading blank lines
    while result.first().is_some_and(|l| l.is_empty()) {
        result.remove(0);
    }

    // Collapse multiple consecutive blank lines into single blank line
    let mut cleaned: Vec<String> = Vec::new();
    let mut last_was_empty = false;
    for line in result {
        if line.is_empty() {
            if !last_was_empty {
                cleaned.push(line);
                last_was_empty = true;
            }
        } else {
            cleaned.push(line);
            last_was_empty = false;
        }
    }

    cleaned.join("\n")
}

fn dedent_block(block: &[String]) -> Vec<String> {
    let min_indent = block
        .iter()
        .filter_map(|line| {
            let stripped = line.trim_start_matches(|c| c == ' ' || c == '\t');
            if stripped.is_empty() {
                None
            } else {
                Some(line.len() - stripped.len())
            }
        })
        .min()
        .unwrap_or(0);

    block
        .iter()
        .map(|line| {
            let stripped = line.trim_start_matches(|c| c == ' ' || c == '\t');
            let leading = line.len() - stripped.len();
            let remove = if min_indent > 0 {
                min_indent.min(leading)
            } else if leading >= 4 {
                leading
            } else {
                0
            };

            if remove > 0 && line.len() >= remove {
                line[remove..].to_string()
            } else {
                line.clone()
            }
        })
        .collect()
}

// ============================================================================
// External Tool Wrappers
// ============================================================================

async fn pdf_pages(pdf: &Path) -> Result<u32> {
    let out = Command::new("pdfinfo")
        .arg(pdf)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .await?;
    if !out.status.success() {
        bail!("pdfinfo failed");
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("Pages:") {
            return Ok(rest.trim().parse()?);
        }
    }
    bail!("could not parse page count");
}

async fn pdftotext_page(pdf: &Path, page: u32) -> Result<String> {
    let out = Command::new("pdftotext")
        .args(["-layout", "-enc", "UTF-8"])
        .args(["-f", &page.to_string(), "-l", &page.to_string()])
        .arg(pdf)
        .arg("-")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .await?;
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

async fn ocr_page_bitmap(
    pdf: &Path,
    page: u32,
    dpi: &str,
    lang: &str,
    psm: &str,
) -> Result<String> {
    let png_path = render_page_png(pdf, page, dpi).await?;
    let out = Command::new("tesseract")
        .arg(&png_path)
        .arg("stdout")
        .args(["-l", lang])
        .args(["--psm", psm])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .await?;
    let _ = fs::remove_file(&png_path).await;
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

async fn render_page_png(pdf: &Path, page: u32, dpi: &str) -> Result<std::path::PathBuf> {
    let tmp_dir = std::env::temp_dir();
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let base = tmp_dir.join(format!("pp-{}-{}", timestamp, page));
    let base_str = base.to_string_lossy().to_string();
    let status = Command::new("pdftoppm")
        .args(["-r", dpi, "-png"])
        .args(["-f", &page.to_string(), "-l", &page.to_string()])
        .arg(pdf)
        .arg(&base_str)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await?;
    if !status.success() {
        bail!("pdftoppm failed");
    }
    Ok(tmp_dir.join(format!("pp-{}-{}-1.png", timestamp, page)))
}

async fn extract_embedded_images(pdf: &Path, page: u32) -> Result<Vec<std::path::PathBuf>> {
    let tmp_dir = std::env::temp_dir();
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let prefix_name = format!("img-{}-p{}", timestamp, page);
    let prefix_path = tmp_dir.join(&prefix_name);
    let prefix_str = prefix_path.to_string_lossy().to_string();

    let status = Command::new("pdfimages")
        .args(["-f", &page.to_string(), "-l", &page.to_string()])
        .args(["-png"])
        .arg(pdf)
        .arg(&prefix_str)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await?;
    if !status.success() {
        bail!("pdfimages failed");
    }

    let mut imgs = Vec::new();
    let mut entries = fs::read_dir(&tmp_dir).await?;
    let pattern = format!("{}-", prefix_name);
    while let Some(e) = entries.next_entry().await? {
        if let Some(name) = e.file_name().to_str() {
            if name.starts_with(&pattern) && name.ends_with(".png") {
                imgs.push(e.path());
            }
        }
    }
    Ok(imgs)
}

async fn ocr_image_file(img: &Path, lang: &str, psm: &str) -> Result<String> {
    let out = Command::new("tesseract")
        .arg(img)
        .arg("stdout")
        .args(["-l", lang])
        .args(["--psm", psm])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .await?;
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Extract vector text from PDF by converting to SVG
/// Requires mutool (from mupdf-tools package)
async fn extract_svg_text(pdf: &Path, page: u32) -> Result<String> {
    use std::cmp::Ordering;

    let out = Command::new("mutool")
        .args(["draw", "-F", "svg", "-o", "-"])
        .arg(pdf)
        .arg(page.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .await?;

    if !out.status.success() {
        bail!("mutool failed");
    }

    let svg_str = String::from_utf8_lossy(&out.stdout);

    // Parse SVG using roxmltree
    let doc = roxmltree::Document::parse(&svg_str)?;
    let mut texts: Vec<(String, f64, f64)> = Vec::new();

    // Extract <text> elements with their coordinates
    extract_svg_text_recursive(doc.root(), &mut texts);

    // Sort by Y coordinate (row), then X (column)
    texts.sort_by(|a, b| match a.1.partial_cmp(&b.1) {
        Some(Ordering::Equal) => a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal),
        Some(ord) => ord,
        None => Ordering::Equal,
    });

    Ok(texts
        .into_iter()
        .map(|(t, _, _)| t)
        .collect::<Vec<_>>()
        .join("\n"))
}

fn extract_svg_text_recursive(node: roxmltree::Node, acc: &mut Vec<(String, f64, f64)>) {
    if node.is_element() && node.tag_name().name() == "text" {
        let mut x = 0.0;
        let mut y = 0.0;
        if let Some(xv) = node.attribute("x") {
            x = xv.parse().unwrap_or(0.0);
        }
        if let Some(yv) = node.attribute("y") {
            y = yv.parse().unwrap_or(0.0);
        }
        let txt = node.text().unwrap_or("").trim();
        if !txt.is_empty() {
            acc.push((txt.to_string(), y, x));
        }
    }
    for child in node.children() {
        extract_svg_text_recursive(child, acc);
    }
}

async fn extract_tables_from_page(pdf: &Path, page: u32) -> Result<Vec<TableOut>> {
    let script = r#"
import json
import sys

try:
    import camelot
except Exception:
    print(json.dumps([]))
    sys.exit(0)

pdf_path = sys.argv[1]
page = sys.argv[2]
flavors = [f.strip() for f in sys.argv[3].split(',') if f.strip()]
min_chars = int(sys.argv[4])

def clean_table(table):
    data = [[str(cell).strip() for cell in row] for row in table.df.values.tolist()]
    rows = [row for row in data if any(cell for cell in row)]
    if not rows:
        return None

    keep_indices = [idx for idx in range(len(rows[0])) if any(row[idx] for row in rows)]
    if not keep_indices:
        return None

    cleaned = [[row[idx] for idx in keep_indices] for row in rows]
    text_chars = sum(len(cell) for row in cleaned for cell in row)
    if text_chars < min_chars:
        return None

    return cleaned

results = []
seen = set()

for flavor in flavors:
    try:
        tables = camelot.read_pdf(pdf_path, pages=page, flavor=flavor, suppress_stdout=True)
    except Exception:
        continue

    for table in tables:
        cleaned = clean_table(table)
        if cleaned is None:
            continue

        key = tuple(tuple(row) for row in cleaned)
        if key in seen:
            continue

        seen.add(key)
        results.append({
            'rows': len(cleaned),
            'cols': len(cleaned[0]) if cleaned else 0,
            'data': cleaned,
            'flavor': flavor,
        })

for idx, tbl in enumerate(results):
    tbl['table_index'] = idx

print(json.dumps(results))
"#;

    let python = if cfg!(target_os = "windows") {
        "python"
    } else {
        "python3"
    };
    let out = Command::new(python)
        .args([
            "-c",
            script,
            &pdf.display().to_string(),
            &page.to_string(),
            "lattice,stream",
            "12",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .await?;

    if out.stdout.is_empty() {
        return Ok(Vec::new());
    }

    #[derive(Deserialize, Default)]
    struct RawTable {
        table_index: usize,
        rows: usize,
        cols: usize,
        data: Vec<Vec<String>>,
        #[allow(dead_code)]
        #[serde(default)]
        flavor: Option<String>,
    }

    let mut tables: Vec<RawTable> =
        serde_json::from_slice(&out.stdout).unwrap_or_else(|_| Vec::new());

    tables.retain(|t| {
        t.rows > 0
            && t.cols > 0
            && t.data
                .iter()
                .any(|row| row.iter().any(|cell| !cell.trim().is_empty()))
    });

    Ok(tables
        .into_iter()
        .map(|t| TableOut {
            table_index: t.table_index,
            rows: t.rows,
            cols: t.cols,
            data: t.data,
        })
        .collect())
}

// ============================================================================
// Helper Functions
// ============================================================================

fn table_to_markdown(table: &TableOut) -> String {
    let column_count = table.data.iter().map(|row| row.len()).max().unwrap_or(0);

    if column_count == 0 {
        return "_Empty table_\n".to_string();
    }

    let mut out = String::new();

    let rows: Vec<Vec<String>> = table
        .data
        .iter()
        .map(|row| {
            (0..column_count)
                .map(|idx| {
                    row.get(idx)
                        .map(|s| s.trim().to_string())
                        .unwrap_or_default()
                })
                .collect()
        })
        .collect();

    let mut header_idx = 0usize;
    let mut best_score = (0usize, 0usize, 0usize); // (non-empty cells, alpha cells, total length)

    for (idx, row) in rows.iter().enumerate() {
        let mut non_empty = 0usize;
        let mut with_alpha = 0usize;
        let mut total_len = 0usize;
        for cell in row {
            if !cell.trim().is_empty() {
                non_empty += 1;
                if cell.chars().any(|c| c.is_alphabetic()) {
                    with_alpha += 1;
                }
                total_len += cell.chars().count();
            }
        }
        let score = (non_empty, with_alpha, total_len);
        if score > best_score {
            best_score = score;
            header_idx = idx;
        }
    }

    let header = rows
        .get(header_idx)
        .cloned()
        .unwrap_or_else(|| vec![String::new(); column_count]);

    let mut header_cells = Vec::with_capacity(column_count);
    for col in 0..column_count {
        let cell = header.get(col).cloned().unwrap_or_default();
        let escaped = escape_markdown_cell(&cell);
        if escaped.is_empty() {
            header_cells.push(format!("Column {}", col + 1));
        } else {
            header_cells.push(escaped);
        }
    }
    let _ = writeln!(&mut out, "| {} |", header_cells.join(" | "));

    let separators = vec!["---"; column_count].join(" | ");
    let _ = writeln!(&mut out, "| {} |", separators);

    for (idx, row) in rows.into_iter().enumerate() {
        if idx <= header_idx {
            continue;
        }
        if row.iter().all(|cell| cell.trim().is_empty()) {
            continue;
        }

        let mut cells = Vec::with_capacity(column_count);
        for col in 0..column_count {
            let escaped = escape_markdown_cell(row.get(col).unwrap_or(&String::new()));
            cells.push(escaped);
        }
        let _ = writeln!(&mut out, "| {} |", cells.join(" | "));
    }

    out
}

fn escape_markdown_cell(cell: &str) -> String {
    let mut cleaned = cell.replace('\r', " ");
    cleaned = cleaned.replace('\n', "<br>");
    cleaned = cleaned.replace('|', "\\|");
    cleaned.trim().to_string()
}
