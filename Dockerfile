# Multi-stage build for minimal final image
FROM rust:1.90-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy source
COPY Cargo.toml Cargo.lock ./
COPY src ./src

# Build release binary
RUN cargo build --release --locked

# Runtime stage with all PDF tools
FROM rust:1.90-slim

# Remove Rust toolchain to save space
RUN rustup self uninstall -y

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-fin \
    tesseract-ocr-eng \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for table extraction
RUN pip3 install --no-cache-dir camelot-py[cv] opencv-python --break-system-packages

# Copy binary from builder
COPY --from=builder /app/target/release/pdfparser-rs /usr/local/bin/pdfparser-rs

# Create non-root user for OpenShift compatibility
RUN useradd -m -u 1001 -s /bin/bash appuser && \
    mkdir -p /tmp/pdfparser && \
    chown -R 1001:0 /tmp/pdfparser && \
    chmod -R g=u /tmp/pdfparser

# Switch to non-root user
USER 1001

# Set temp directory
ENV TMPDIR=/tmp/pdfparser

# Expose port
EXPOSE 3000

# Run the API server
CMD ["/usr/local/bin/pdfparser-rs"]
