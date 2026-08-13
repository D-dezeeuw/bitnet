# syntax=docker/dockerfile:1
#
# === Build stage ===
#
# Both stages are pinned to the same Debian suite. The runtime copies compiled
# .so files out of the builder, so a suite mismatch means a glibc/libstdc++
# mismatch at load time rather than a build error.
FROM python:3.11-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    clang \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Pinned: an unpinned clone means the image changes without a commit here, and
# the patch below is silently sensitive to upstream renaming a line.
ARG BITNET_REF=0b341e582afbf9e1011f24744b554c96a3477eb5
RUN git clone --recursive https://github.com/microsoft/BitNet.git . \
    && git checkout "${BITNET_REF}" \
    && git submodule update --init --recursive

# Fix const-correctness in ggml-bitnet-mad.cpp (clang treats it as a hard
# error). `sed` does not fail when its pattern is absent, so assert the match
# explicitly -- otherwise an upstream rename turns this into a silent no-op and
# the failure surfaces much later as a confusing compile error.
RUN grep -q 'int8_t \* y_col = y + col \* by;' src/ggml-bitnet-mad.cpp \
    || (echo "ERROR: const-correctness patch no longer matches upstream" >&2; exit 1)
RUN sed -i 's/int8_t \* y_col = y + col \* by;/const int8_t * y_col = y + col * by;/' src/ggml-bitnet-mad.cpp \
    && grep -q 'const int8_t \* y_col' src/ggml-bitnet-mad.cpp

# Generate the LUT kernel header. src/CMakeLists.txt compiles
# ggml-bitnet-lut.cpp unconditionally and that file #includes
# bitnet-lut-kernels.h, so this step is required even though TL2 itself is
# disabled below and inference runs on I2_S kernels.
#
# These are exactly the arguments setup_env.py passes for BitNet-b1.58-2B-4T on
# x86_64. codegen_tl2.py imports only argparse/os/configparser -- no torch.
RUN python utils/codegen_tl2.py \
      --model bitnet_b1_58-3B \
      --BM 160,320,320 --BK 96,96,96 --bm 32,32,32 \
    && test -s include/bitnet-lut-kernels.h

# setup_env.py is deliberately bypassed. It would (a) require the 1.2 GB GGUF
# to be present in this stage purely to run an existence check, since
# prepare_model() skips conversion when the file already exists, and (b) force
# `pip install -r requirements.txt`, whose only consumer is the HF->GGUF
# converter that never runs for a pre-converted model. That is ~2 GB of torch
# for nothing. cmake is invoked directly with the same flags instead, plus a
# target list and -j: upstream's `cmake --build build` builds every example
# single-threaded.
RUN cmake -B build \
      -DBITNET_X86_TL2=OFF \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_COMMON=ON \
      -DLLAMA_BUILD_SERVER=ON \
      -DLLAMA_BUILD_TOOLS=ON \
      -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_TESTS=OFF

RUN cmake --build build --config Release -j"$(nproc)" --target llama-server llama-cli \
    && test -f build/bin/llama-server \
    && test -f build/bin/llama-cli

# Collect shared libraries -- wildcards catch versioned names (libllama.so.1).
RUN mkdir -p /build/collected_libs \
    && find /build/build -name "libllama.so*" -exec cp -a {} /build/collected_libs/ \; \
    && find /build/build -name "libggml*.so*" -exec cp -a {} /build/collected_libs/ \;

#
# === Runtime stage ===
#
FROM python:3.13-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system bitnet && useradd --system --gid bitnet --create-home bitnet

WORKDIR /app

COPY --from=builder /build/build/bin/llama-cli /usr/local/bin/llama-cli
COPY --from=builder /build/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=builder /build/collected_libs/ /usr/local/lib/
RUN ldconfig

# Fail the build, not the container, if a shared library did not come across.
RUN ldd /usr/local/bin/llama-server \
    && ! ldd /usr/local/bin/llama-server | grep -q "not found"

# The model is copied straight from the build context into the runtime stage.
# It is not needed to compile anything, so putting it in the builder only meant
# a model change invalidated the expensive kernel build cache.
COPY ggml-model-i2_s.gguf /app/models/model.gguf

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app.py /app/app.py
COPY static/ /app/static/
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# /download serves from here. Nothing is baked in: mount a file at
# /app/downloads/download.zip to enable it, and the route 404s when absent.
RUN mkdir -p /app/downloads && chown bitnet:bitnet /app/downloads

ENV MODEL_PATH=/app/models/model.gguf \
    MODEL_ID=bitnet-b1.58-2B-4T \
    LLAMA_SERVER_PORT=8080 \
    BITNET_THREADS=4 \
    BITNET_CTX_SIZE=4096 \
    BITNET_STATIC_DIR=/app/static \
    BITNET_DOWNLOAD_PATH=/app/downloads/download.zip
# BITNET_API_KEY is intentionally unset: set it at run time to require
# authentication on every /v1 route. Leaving it unset serves an open endpoint.

EXPOSE 8010

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8010/health || exit 1

USER bitnet

ENTRYPOINT ["/app/entrypoint.sh"]
