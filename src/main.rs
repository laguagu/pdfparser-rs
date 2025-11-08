//! PDF Parser API Server
//!
//! REST API that accepts PDF uploads and returns parsed JSON.
//! Uses the shared pdfparser_rs library for all parsing logic.

use axum::{
    extract::{Multipart, Query, Request},
    http::{header, HeaderValue, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use pdfparser_rs::{parse_pdf, render_markdown, ParserConfig};
use serde::Deserialize;
use std::env;
use tempfile::tempdir;
use tokio::{fs, net::TcpListener};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Check if API key is set
    let api_key = env::var("API_KEY").ok();
    let has_api_key = api_key.is_some();

    // Load and display configuration
    let config = ParserConfig::default();

    let app = Router::new()
        .route("/parse", post(parse_handler))
        .route("/health", axum::routing::get(health_check))
        .layer(middleware::from_fn(move |req, next| {
            let key = api_key.clone();
            api_key_middleware(req, next, key)
        }));

    let addr = "0.0.0.0:3000";

    println!("🚀 PDF Parser API");
    println!("📡 Listening on http://{}/parse", addr);
    if has_api_key {
        println!("🔐 API Key authentication enabled");
    } else {
        println!("⚠️  No API_KEY set - running without authentication");
    }
    println!("\n⚙️  Configuration:");
    println!("   • Max concurrency: {}", config.max_concurrency);
    println!("   • OCR over text: {}", config.ocr_over_text);
    println!("   • OCR images: {}", config.ocr_images);
    println!("   • Extract tables: {}", config.extract_tables);
    println!("   • OCR language: {}", config.ocr_lang);
    println!("   • OCR DPI: {}", config.ocr_dpi);
    println!("\n💡 Override with query params: ?lang=eng&dpi=150");
    println!("📝 POST your PDF files here!");

    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

/// Health check endpoint
async fn health_check() -> &'static str {
    "OK"
}

/// API key authentication middleware
async fn api_key_middleware(
    req: Request,
    next: Next,
    expected_key: Option<String>,
) -> Result<Response, StatusCode> {
    // Skip auth for health check
    if req.uri().path() == "/health" {
        return Ok(next.run(req).await);
    }

    // If no API key configured, allow all requests
    let Some(expected) = expected_key else {
        return Ok(next.run(req).await);
    };

    // Check X-API-Key header
    let provided_key = req
        .headers()
        .get("X-API-Key")
        .and_then(|v| v.to_str().ok());

    match provided_key {
        Some(key) if key == expected => Ok(next.run(req).await),
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}

/// API handler for PDF parsing
async fn parse_handler(
    query: Option<Query<ParseParams>>,
    mut mp: Multipart,
) -> Result<Response, (StatusCode, String)> {
    // Receive PDF file
    let mut pdf_data: Vec<u8> = Vec::new();
    while let Some(field) = mp.next_field().await.map_err(map_error)? {
        if field.name() == Some("file") {
            let data = field.bytes().await.map_err(map_error)?;
            pdf_data.extend_from_slice(&data);
        }
    }

    if pdf_data.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "No PDF file uploaded".into()));
    }

    // Save to temp file
    let dir = tempdir().map_err(map_error)?;
    let pdf_path = dir.path().join("upload.pdf");
    fs::write(&pdf_path, &pdf_data).await.map_err(map_error)?;

    // Parse with configuration from environment + optional query params
    let params = query.map(|Query(p)| p).unwrap_or_default();
    let mut config = ParserConfig::default();

    // Allow runtime overrides via query parameters
    if let Some(lang) = params.lang.clone() {
        config = config.with_lang(lang);
    }
    if let Some(dpi) = params.dpi.clone() {
        config = config.with_dpi(dpi);
    }

    let result = parse_pdf(&pdf_path, config).await.map_err(map_error)?;
    let wants_markdown = params
        .format
        .as_deref()
        .map(|fmt| fmt.eq_ignore_ascii_case("markdown") || fmt.eq_ignore_ascii_case("md"))
        .unwrap_or(false);

    if wants_markdown {
        let markdown = render_markdown(&result);
        let headers = [(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/markdown; charset=utf-8"),
        )];
        Ok((headers, markdown).into_response())
    } else {
        Ok(Json(result).into_response())
    }
}

#[derive(Default, Deserialize)]
struct ParseParams {
    #[serde(default)]
    format: Option<String>,
    /// Override OCR language (e.g., "eng", "fin+eng", "deu")
    #[serde(default)]
    lang: Option<String>,
    /// Override OCR DPI (e.g., "150", "300", "600")
    #[serde(default)]
    dpi: Option<String>,
}

fn map_error<E: std::fmt::Display>(e: E) -> (StatusCode, String) {
    let msg = e.to_string().to_lowercase();
    let status = if msg.contains("pdf") && (msg.contains("invalid") || msg.contains("failed")) {
        StatusCode::UNPROCESSABLE_ENTITY
    } else {
        StatusCode::INTERNAL_SERVER_ERROR
    };
    (status, e.to_string())
}
