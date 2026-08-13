# === Build stage ===
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    clang \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Clone BitNet repo (includes patched llama.cpp)
RUN git clone --recursive https://github.com/microsoft/BitNet.git .

# Fix const-correctness bug in ggml-bitnet-mad.cpp (clang treats as hard error)
RUN sed -i 's/int8_t \* y_col = y + col \* by;/const int8_t * y_col = y + col * by;/' src/ggml-bitnet-mad.cpp

# Install BitNet dependencies (requires Python <=3.12 for torch~=2.2.1)
RUN pip install --no-cache-dir -r requirements.txt

# Copy local pre-converted GGUF model
COPY ggml-model-i2_s.gguf models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf

# Build via setup_env.py (generates TL2 kernels, detects existing GGUF, builds binaries)
RUN python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s || \
    (cat /build/logs/compile.log 2>/dev/null; cat /build/logs/convert.log 2>/dev/null; exit 1) && \
    test -f /build/build/bin/llama-cli && echo "Build OK"

# Ensure llama-server is also built (setup_env.py may not include it as a target)
RUN test -f /build/build/bin/llama-server || \
    cmake --build /build/build --target llama-server -j$(nproc)

# Collect shared libraries — use wildcards to catch versioned names (e.g. libllama.so.1)
RUN mkdir -p /build/collected_libs && \
    find /build/build -name "libllama.so*" -exec cp -a {} /build/collected_libs/ \; && \
    find /build/build -name "libggml*.so*" -exec cp -a {} /build/collected_libs/ \;

# === Runtime stage ===
FROM python:3.13-slim

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN groupadd --system bitnet && useradd --system --gid bitnet --create-home bitnet

WORKDIR /app

# Copy binaries + shared libs
COPY --from=builder /build/build/bin/llama-cli /usr/local/bin/llama-cli
COPY --from=builder /build/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=builder /build/collected_libs/ /usr/local/lib/
RUN ldconfig

# Copy model
COPY --from=builder /build/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf /app/models/model.gguf

# Install FastAPI + httpx for proxying to llama-server
RUN pip install --no-cache-dir fastapi==0.115.12 uvicorn==0.34.0 httpx==0.28.1 pydantic==2.11.1

COPY app.py /app/app.py
COPY download.zip /app/static/download.zip
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV MODEL_PATH=/app/models/model.gguf
ENV MODEL_ID=bitnet-b1.58-2B-4T
ENV LLAMA_SERVER_PORT=8080
ENV BITNET_THREADS=4
ENV BITNET_CTX_SIZE=4096

EXPOSE 8010

USER bitnet

ENTRYPOINT ["/app/entrypoint.sh"]
