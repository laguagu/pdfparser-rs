# PDF Parser

Fast multimodal PDF parser in Rust. Extracts text, OCR, and tables as clean markdown for LLMs.

## Quick Start

```bash
cargo build --release
cargo run --release --bin parser test.pdf  # → test.md
```

## Features

- **Text extraction** - pdftotext native layer
- **OCR** - Tesseract (rus+fin+eng)
- **Tables** - Camelot markdown tables
- **LLM-ready** - Clean output, no duplicates

## Configuration

Edit `src/lib.rs` lines 23-53:

```rust
const OCR_LANG: &str = "rus+fin+eng";  // Language
const OCR_DPI: &str = "300";           // Quality
const EXTRACT_TABLES: bool = true;     // Tables
```

## Dependencies

```bash
# Ubuntu/Debian
apt-get install poppler-utils tesseract-ocr tesseract-ocr-{rus,fin,eng} python3-pip
pip3 install camelot-py[cv] opencv-python

# macOS
brew install poppler tesseract tesseract-lang python3
pip3 install camelot-py[cv] opencv-python
```

## Docker

```bash
# Generate API key
export API_KEY=$(openssl rand -base64 32)

# Build & run
docker build -t pdfparser .
docker run -p 3000:3000 -e API_KEY=$API_KEY pdfparser

# Test
curl -X POST http://localhost:3000/parse \
  -H "X-API-Key: $API_KEY" \
  -F "file=@test.pdf"
```

## REST API

```bash
# Start server
cargo run --release  # → :3000

# Parse to JSON
curl -X POST http://localhost:3000/parse -F "file=@doc.pdf"

# Parse to Markdown
curl "http://localhost:3000/parse?format=markdown" -F "file=@doc.pdf"

# With API key
API_KEY=secret cargo run --release
curl -H "X-API-Key: secret" -F "file=@doc.pdf" http://localhost:3000/parse
```

## Output Example

```markdown
## Page 1

Economy Profile 2/2 Working Age Population (Millions)
Japan 98.4

### Table 1

| Part no. | ID | Description |
| --- | --- | --- |
| 1750 | 3AFP9073612 | COPPER BAR |
```

## License

MIT
