#!/bin/bash
set -euo pipefail

# Load .env if present. Values act as defaults: anything already exported wins,
# so `BITNET_THREADS=8 ./start.sh` still overrides the file.
#
# The file is parsed, not sourced. Sourcing would execute whatever it contains,
# and this file holds the API key -- a config file should not be a shell script.
if [ -f .env ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line#"${line%%[![:space:]]*}"}"        # strip leading whitespace
    case "$line" in ''|'#'*) continue ;; esac
    case "$line" in *=*) ;; *) continue ;; esac
    key="${line%%=*}"
    val="${line#*=}"
    key="${key#export }"
    key="${key%"${key##*[![:space:]]}"}"           # strip trailing whitespace
    # A regex, not a case glob: in glob syntax `[A-Za-z0-9_]*` is a single
    # character class followed by a wildcard, so it accepts "BAD-KEY" and the
    # export below then fails and takes the whole script down under `set -e`.
    [[ $key =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    case "$val" in                                 # strip matched quotes
      \"*\") val="${val#\"}"; val="${val%\"}" ;;
      \'*\') val="${val#\'}"; val="${val%\'}" ;;
    esac
    [ -n "${!key+set}" ] || export "$key=$val"
  done < .env
  echo "Loaded configuration from .env"
fi

IMAGE="${IMAGE:-bitnet-2b-api}"
CONTAINER="${CONTAINER:-bitnet-2b}"
NETWORK="${NETWORK:-nginx-proxy-manager_default}"
STATIC_IP="${STATIC_IP:-172.22.0.25}"
API_PORT="${BITNET_API_PORT:-8010}"
MODEL_FILE="ggml-model-i2_s.gguf"
# Pinned for the same reason the upstream source is: the HF repo is a moving
# target (last updated 2025-12-17, well after its April 2025 release), so an
# unpinned download can silently change the model under a rebuild.
MODEL_REPO="microsoft/bitnet-b1.58-2B-4T-gguf"
MODEL_REVISION="a1f2f1c765812aa8af3f6eda4a313707064bba15"
MODEL_SHA256="4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162"

# Publish a host port for local testing: HOST_PORT=8010 ./start.sh
# Left empty by default -- the container is reached through the reverse proxy.
HOST_PORT="${HOST_PORT:-}"

if [ ! -f "$MODEL_FILE" ]; then
  echo "ERROR: $MODEL_FILE not found in current directory." >&2
  echo "Download it first:" >&2
  echo "  huggingface-cli download $MODEL_REPO $MODEL_FILE \\" >&2
  echo "    --revision $MODEL_REVISION --local-dir ." >&2
  exit 1
fi

# Warn rather than fail: a deliberate swap (a fine-tune, a different quant) is
# legitimate, but an unnoticed one is not.
if command -v sha256sum > /dev/null 2>&1; then
  echo "Verifying model checksum..."
  ACTUAL_SHA="$(sha256sum "$MODEL_FILE" | cut -d' ' -f1)"
  if [ "$ACTUAL_SHA" != "$MODEL_SHA256" ]; then
    echo "WARNING: $MODEL_FILE does not match the pinned revision." >&2
    echo "  expected $MODEL_SHA256" >&2
    echo "  actual   $ACTUAL_SHA" >&2
    echo "  pinned revision: $MODEL_REVISION" >&2
    echo "Continuing. If this was not intentional, re-download at that revision." >&2
  else
    echo "Model matches pinned revision $MODEL_REVISION"
  fi
fi

# The network is created by another stack (nginx-proxy-manager's compose file),
# so it is a precondition here, not something to create silently -- creating it
# with the wrong subnet would leave the proxy unable to reach this container.
if ! docker network inspect "$NETWORK" > /dev/null 2>&1; then
  echo "ERROR: docker network '$NETWORK' does not exist." >&2
  echo "It normally comes from the nginx-proxy-manager stack; start that first." >&2
  echo "To create it manually:" >&2
  echo "  docker network create --subnet 172.22.0.0/16 $NETWORK" >&2
  exit 1
fi

# --ip only works on a network with an explicit IPAM subnet. Without one docker
# fails with "user specified IP address is supported only when connecting to
# networks with user configured subnets", which does not say what to do.
SUBNETS="$(docker network inspect -f \
  '{{range .IPAM.Config}}{{.Subnet}} {{end}}' "$NETWORK" 2>/dev/null | tr -s ' ')"
if [ -z "${SUBNETS// /}" ]; then
  echo "ERROR: network '$NETWORK' has no configured subnet, so the static IP" >&2
  echo "$STATIC_IP cannot be assigned. Recreate it with an explicit --subnet," >&2
  echo "or unset STATIC_IP to let docker assign an address." >&2
  exit 1
fi
echo "Network $NETWORK found (subnet: ${SUBNETS% })"

echo "Building BitNet 2B Docker image..."
docker build -t "$IMAGE" .

docker rm -f "$CONTAINER" 2>/dev/null || true

RUN_ARGS=(
  --name "$CONTAINER"
  --network "$NETWORK"
  --ip "$STATIC_IP"
  --restart unless-stopped
)

# Authentication is off unless a key is supplied. Without one every /v1 route
# is open to anything that can reach the container.
if [ -n "${BITNET_API_KEY:-}" ]; then
  echo "API key authentication: ENABLED"
else
  echo "API key authentication: DISABLED (set BITNET_API_KEY to require one)"
fi

# Forward configuration into the container by CONVENTION, not by a list: every
# BITNET_* variable is a container setting, and the deploy-only knobs
# (IMAGE, CONTAINER, NETWORK, STATIC_IP, HOST_PORT) deliberately carry no such
# prefix because they govern `docker run` itself and mean nothing inside.
#
# This was a hand-maintained list, and it drifted exactly as you would expect:
# BITNET_TEMPERATURE, BITNET_SYSTEM_PROMPT, BITNET_LOOP_GUARD, the DRY knobs and
# BITNET_PROMPT_FORMAT were all added to app.py, documented as configurable, and
# never added here -- so setting any of them in .env did nothing at all. Reading
# the environment instead means a new setting works the moment it exists.
#
# compgen -v yields variable NAMES only, so values containing newlines (a
# multi-line BITNET_SYSTEM_PROMPT) survive intact. Unset variables are skipped
# so the image's own ENV defaults apply rather than being overridden with "".
for var in $(compgen -v | grep '^BITNET_' | sort) MODEL_ID LLAMA_SERVER_PORT; do
  if [ -n "${!var:-}" ]; then
    RUN_ARGS+=(-e "$var=${!var}")
  fi
done

# /download is served from a mount, so download.zip is optional and never a
# build dependency. The route returns 404 when nothing is mounted.
if [ -f "download.zip" ]; then
  RUN_ARGS+=(-v "$(pwd)/download.zip:/app/downloads/download.zip:ro")
  echo "Download file: mounted"
fi

if [ -n "$HOST_PORT" ]; then
  RUN_ARGS+=(-p "${HOST_PORT}:${API_PORT}")
fi

echo "Starting BitNet 2B container..."
docker run -d "${RUN_ARGS[@]}" "$IMAGE"

# Confirm the address was actually taken. `docker run --ip` fails loudly on a
# conflict, but a container that exits immediately also leaves no address, and
# that is worth catching here rather than when the proxy 502s.
ACTUAL_IP="$(docker inspect -f \
  "{{with index .NetworkSettings.Networks \"$NETWORK\"}}{{.IPAddress}}{{end}}" \
  "$CONTAINER" 2>/dev/null || true)"
if [ "$ACTUAL_IP" != "$STATIC_IP" ]; then
  echo "ERROR: expected the container at $STATIC_IP on $NETWORK," >&2
  echo "but it reports '${ACTUAL_IP:-no address}'. Container logs:" >&2
  docker logs --tail 20 "$CONTAINER" >&2 2>/dev/null || true
  exit 1
fi

cat <<EOF

BitNet 2B API running on network $NETWORK
Internal: http://$CONTAINER:$API_PORT
Static IP: http://$STATIC_IP:$API_PORT  (confirmed assigned)

Point the nginx-proxy-manager host at $STATIC_IP port $API_PORT.

EOF

if [ -n "$HOST_PORT" ]; then
  cat <<EOF
Published on the host at port $HOST_PORT:

  curl http://localhost:${HOST_PORT}/health
  curl http://localhost:${HOST_PORT}/v1/models
  curl -s http://localhost:${HOST_PORT}/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{"messages":[{"role":"user","content":"Hello"}]}'
EOF
else
  cat <<EOF
No host port published, so localhost will not reach it. Test from inside the
container, or re-run with HOST_PORT=${API_PORT} ./start.sh

  docker exec $CONTAINER curl -s http://127.0.0.1:${API_PORT}/health
  docker exec $CONTAINER curl -s http://127.0.0.1:${API_PORT}/v1/models
EOF
fi

cat <<EOF

The model loads before the API answers; give it a minute or two on first start.
Follow it with: docker logs -f $CONTAINER
EOF
