# PDF Parser API

Fast PDF parsing with Rust. Extract text, OCR images, parse tables. Optimized for LLM consumption.

## Structure

```
src/
├── lib.rs          # Core parsing logic
├── main.rs         # REST API server
└── bin/parser.rs   # CLI tool
```

## Quick Start

```bash
# Build
cargo build --release

# API Server
cargo run  # → http://0.0.0.0:3000/parse

# CLI
cargo run --bin parser file.pdf              # Full parsing
cargo run --bin parser -- --no-ocr file.pdf  # Fast mode (10x faster)
```

## Features

- **Text extraction** (pdftotext) - Native text layer
- **OCR** (tesseract) - Scanned pages
- **Table extraction** (camelot) - Structured tables
- **Image OCR** - Embedded images
- **SVG text** (optional) - Vector text with mutool
- **LLM-ready** - Normalized whitespace, structured JSON

## Configuration

```bash
OCR_OVER_TEXT=1      # OCR charts/diagrams (default: true)
OCR_IMAGES=1         # OCR embedded images (default: true)
EXTRACT_TABLES=1     # Extract tables (default: true)
SVG_TEXT=1           # SVG vector text (default: false)
OCR_DPI=300          # Resolution
OCR_LANG=fin+eng     # Languages
MAX_CONCURRENCY=4    # Parallel processing
```

## API Usage

```bash
curl -X POST http://localhost:3000/parse -F "file=@document.pdf" -o output.json
```

## Performance

```bash
# Fastest (no OCR)
./target/release/parser --no-ocr document.pdf

# Disable features
OCR_OVER_TEXT=0 OCR_IMAGES=0 cargo run
```

## Dependencies

```bash
# Ubuntu/Debian
apt-get install -y poppler-utils tesseract-ocr python3 python3-pip
pip3 install camelot-py[cv] opencv-python

# macOS
brew install poppler tesseract python3
pip3 install camelot-py[cv] opencv-python

# Windows
choco install poppler tesseract python
pip install camelot-py[cv] opencv-python
```

## Deployment (OpenShift/Rahti)

```dockerfile
FROM rust:1.75 as builder
COPY . /app
WORKDIR /app
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y poppler-utils tesseract-ocr python3 python3-pip \\
    && pip3 install camelot-py[cv] opencv-python
COPY --from=builder /app/target/release/pdfparser-rs /usr/local/bin/
EXPOSE 3000
CMD ["pdfparser-rs"]
```

## Testing

```bash
cargo run --bin parser test_pdfs/example.pdf
cargo run --bin parser test_pdfs/wef-1-page.pdf  # Has tables
```

## Future: LLM Integration

See `docs/LLM_INTEGRATION.md` for using GPT-4 Vision/Claude for superior image extraction.

---

**License:** MIT | **Docs:** See CHEATSHEET.md
