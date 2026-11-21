#!/usr/bin/env bash
set -Eeuo pipefail

PORT="${1:-8501}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="$SCRIPT_DIR/logs/pids"

for service in ngrok streamlit; do
    PID_FILE="$PID_DIR/${service}_${PORT}.pid"
    if [[ ! -f "$PID_FILE" ]]; then
        printf '%s tidak tercatat berjalan untuk port %s.\n' "$service" "$PORT"
        continue
    fi
    PID="$(<"$PID_FILE")"
    if [[ "$PID" =~ ^[0-9]+$ ]] && kill -0 "$PID" 2>/dev/null; then
        kill "$PID" 2>/dev/null || true
        printf '%s dihentikan (PID %s).\n' "$service" "$PID"
    else
        printf 'PID %s untuk %s sudah tidak aktif.\n' "$PID" "$service"
    fi
    rm -f "$PID_FILE"
done
