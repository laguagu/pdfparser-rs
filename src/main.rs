//! PDF Parser API Server
//!
//! REST API that accepts PDF uploads and returns parsed JSON.
//! Uses the shared pdfparser_rs library for all parsing logic.

use axum::{extract::Multipart, http::StatusCode, routing::post, Json, Router};
use pdfparser_rs::{parse_pdf, ParserConfig, ParseResult};
use tempfile::tempdir;
use tokio::{fs, net::TcpListener};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let app = Router::new().route("/parse", post(parse_handler));
    let addr = "0.0.0.0:3000";
    
    println!("🚀 PDF Parser API");
    println!("📡 Listening on http://{}/parse", addr);
    println!("📝 POST your PDF files here!");
    
    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

/// API handler for PDF parsing
async fn parse_handler(mut mp: Multipart) -> Result<Json<ParseResult>, (StatusCode, String)> {
    // Receive PDF file
    let mut pdf_data: Vec<u8> = Vec::new();
    while let Some(field) = mp.next_field().await.map_err(to_500)? {
        if field.name() == Some("file") {
            let data = field.bytes().await.map_err(to_500)?;
            pdf_data.extend_from_slice(&data);
        }
    }

    if pdf_data.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "No PDF file uploaded".into()));
    }

    // Save to temp file
    let dir = tempdir().map_err(to_500)?;
    let pdf_path = dir.path().join("upload.pdf");
    fs::write(&pdf_path, &pdf_data).await.map_err(to_500)?;

    // Parse with default configuration
    let config = ParserConfig::default();
    let result = parse_pdf(&pdf_path, config).await.map_err(to_500)?;

    Ok(Json(result))
}

fn to_500<E: std::fmt::Display>(e: E) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
}
