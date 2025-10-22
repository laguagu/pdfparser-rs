//! PDF Parser Library - Shared parsing logic
//!
//! This library contains the core PDF parsing functionality that can be used by:
//! - API server (main.rs)
//! - CLI tool (bin/parser.rs)

use anyhow::{bail, Result};
use futures::stream::{self, StreamExt};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    env,
    path::Path,
    process::Stdio,
    sync::Arc,
};
use tokio::{fs, process::Command, sync::Semaphore};

// Note: roxmltree used only when SVG_TEXT=1
// use roxmltree imported inline in extract_svg_text function

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
        Self {
            max_concurrency: env_var_usize("MAX_CONCURRENCY", 4),
            text_min_chars: env_var_usize("TEXT_MIN_CHARS", 20),
            ocr_over_text: env_var_bool("OCR_OVER_TEXT", true),
            ocr_images: env_var_bool("OCR_IMAGES", true),
            extract_tables: env_var_bool("EXTRACT_TABLES", true),
            svg_text: env_var_bool("SVG_TEXT", false), // Default: false for speed
            ocr_dpi: env::var("OCR_DPI").unwrap_or_else(|_| "300".to_string()),
            ocr_lang: env::var("OCR_LANG").unwrap_or_else(|_| "fin+eng".to_string()),
            ocr_psm: env::var("OCR_PSM").unwrap_or_else(|_| "3".to_string()),
        }
    }
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
        if let Ok(ocr_txt) = ocr_page_bitmap(pdf, page, &config.ocr_dpi, &config.ocr_lang, &config.ocr_psm).await {
            final_text = ocr_txt;
            src = "ocr";
        } else {
            final_text = base_text;
            src = "ocr";
        }
    }

    // 2) OCR over text layer (captures charts/diagrams)
    if config.ocr_over_text && has_text {
        if let Ok(ocr_txt2) = ocr_page_bitmap(pdf, page, &config.ocr_dpi, &config.ocr_lang, &config.ocr_psm).await {
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
    s.trim()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Normalize whitespace for LLM consumption
fn normalize_whitespace(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let mut result = Vec::new();
    let mut prev_empty = false;

    for line in lines {
        let trimmed = line.trim_end();
        let is_empty = trimmed.is_empty();

        // Skip consecutive empty lines
        if is_empty && prev_empty {
            continue;
        }

        result.push(trimmed.to_string());
        prev_empty = is_empty;
    }

    // Remove trailing empty lines
    while result.last().map_or(false, |l| l.is_empty()) {
        result.pop();
    }

    result.join("\n")
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

async fn ocr_page_bitmap(pdf: &Path, page: u32, dpi: &str, lang: &str, psm: &str) -> Result<String> {
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
        .arg(&page.to_string())
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
    texts.sort_by(|a, b| {
        match a.1.partial_cmp(&b.1) {
            Some(Ordering::Equal) => a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal),
            Some(ord) => ord,
            None => Ordering::Equal,
        }
    });
    
    Ok(texts.into_iter().map(|(t, _, _)| t).collect::<Vec<_>>().join("\n"))
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
import camelot, json, sys
try:
    tables = camelot.read_pdf(sys.argv[1], pages=sys.argv[2], flavor=sys.argv[3], suppress_stdout=True)
    result = []
    for i, table in enumerate(tables):
        data = table.df.values.tolist()
        result.append({
            'table_index': i,
            'rows': len(data),
            'cols': len(data[0]) if data else 0,
            'data': [[str(cell) for cell in row] for row in data]
        })
    print(json.dumps(result))
except:
    print(json.dumps([]))
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
            "lattice",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .await?;

    if out.stdout.is_empty() {
        return Ok(Vec::new());
    }

    #[derive(Deserialize)]
    struct RawTable {
        table_index: usize,
        rows: usize,
        cols: usize,
        data: Vec<Vec<String>>,
    }

    let tables: Vec<RawTable> = serde_json::from_slice(&out.stdout).unwrap_or_else(|_| Vec::new());

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

fn env_var_usize(key: &str, def: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(def)
}

fn env_var_bool(key: &str, def: bool) -> bool {
    match std::env::var(key) {
        Ok(v) => matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "on"),
        Err(_) => def,
    }
}
