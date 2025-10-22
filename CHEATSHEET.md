# PDF Parser - Quick Reference

## CLI

```bash
parser file.pdf              # Full
parser --no-ocr file.pdf     # Fast (10x)
```

## API

```bash
cargo run  # → http://0.0.0.0:3000/parse
curl -X POST http://localhost:3000/parse -F "file=@doc.pdf" -o out.json
```

## Config

```bash
OCR_OVER_TEXT=1 OCR_IMAGES=1 EXTRACT_TABLES=1 SVG_TEXT=0
OCR_DPI=300 OCR_LANG=fin+eng MAX_CONCURRENCY=4
```

## Speed

```bash
./target/release/parser --no-ocr file.pdf  # Fastest
OCR_OVER_TEXT=0 OCR_IMAGES=0 cargo run    # Fast API
MAX_CONCURRENCY=1 cargo run               # Low memory
```

## Install

```bash
# Ubuntu: apt-get install poppler-utils tesseract-ocr python3-pip
# pip3 install camelot-py[cv] opencv-python
```

---

Full docs: README.md | LLM: docs/LLM_INTEGRATION.md
