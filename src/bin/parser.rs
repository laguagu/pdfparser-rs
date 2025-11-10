//! PDF Parser CLI Tool
//!
//! Simple command-line tool for parsing PDFs.
//!
//! Usage:
//!   parser <file.pdf>           # Full parsing (text + OCR + tables)
//!   parser --no-ocr <file.pdf>  # Fast mode (text + tables only, no OCR)
//!
//! Environment variables:
//!   OCR_IMAGES=1        # OCR embedded images (default: true)
//!   EXTRACT_TABLES=1    # Extract tables (default: true)
//!   SVG_TEXT=1          # Extract SVG vector text (default: false)
//!   OCR_DPI=300         # OCR resolution
//!   OCR_LANG=fin+eng    # OCR languages

use pdfparser_rs::{parse_pdf, render_markdown, ParserConfig};
use std::env;
use std::path::PathBuf;
use std::time::Instant;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();

    // Parse arguments
    let (pdf_path, no_ocr) = if args.len() == 2 {
        (PathBuf::from(&args[1]), false)
    } else if args.len() == 3 && args[1] == "--no-ocr" {
        (PathBuf::from(&args[2]), true)
    } else {
        eprintln!("Usage:");
        eprintln!("  {} <file.pdf>", args[0]);
        eprintln!("  {} --no-ocr <file.pdf>  (fast mode)", args[0]);
        std::process::exit(1);
    };

    if !pdf_path.exists() {
        eprintln!("❌ File not found: {:?}", pdf_path);
        std::process::exit(1);
    }

    // Configure parser
    let mut config = ParserConfig::default();

    if no_ocr {
        println!("⚡ FAST MODE: OCR disabled");
        config.ocr_over_text = false;
        config.ocr_images = false;
    }

    println!("📄 Parsing: {:?}", pdf_path);
    println!("⚙️  Config:");
    println!("   OCR over text: {}", config.ocr_over_text);
    println!("   OCR images: {}", config.ocr_images);
    println!("   Extract tables: {}", config.extract_tables);
    println!("   SVG text: {}", config.svg_text);
    println!("   Concurrency: {}", config.max_concurrency);

    // Parse PDF
    let start = Instant::now();
    let result = parse_pdf(&pdf_path, config).await?;
    let elapsed = start.elapsed();

    // Save Markdown (only format we need for LLMs)
    let md_path = pdf_path.with_extension("md");
    let markdown = render_markdown(&result);
    tokio::fs::write(&md_path, markdown).await?;

    // Print summary
    println!("\n✅ Parsed {} pages in {:.2}s", result.pages.len(), elapsed.as_secs_f64());
    for page in &result.pages {
        let lines = page.text.lines().count();
        let tables = page.tables.len();
        println!(
            "   Page {}: {} lines, {} tables (source: {})",
            page.page, lines, tables, page.source
        );
    }
    println!("📝 Output: {:?}", md_path);

    Ok(())
}
