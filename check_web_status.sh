#!/usr/bin/env bash
set -u

PORT="${1:-8501}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="$SCRIPT_DIR/logs/pids"

for service in streamlit ngrok; do
    PID_FILE="$PID_DIR/${service}_${PORT}.pid"
    if [[ -f "$PID_FILE" ]]; then
        PID="$(<"$PID_FILE")"
        if kill -0 "$PID" 2>/dev/null; then
            printf '%s: berjalan (PID %s)\n' "$service" "$PID"
        else
            printf '%s: PID lama, proses sudah berhenti\n' "$service"
        fi
    else
        printf '%s: tidak tercatat berjalan\n' "$service"
    fi
done

if command -v curl >/dev/null 2>&1 && curl -fsS "http://127.0.0.1:$PORT/_stcore/health" >/dev/null 2>&1; then
    printf 'Aplikasi: http://127.0.0.1:%s merespons\n' "$PORT"
else
    printf 'Aplikasi: tidak merespons pada port %s\n' "$PORT"
fi

NGROK_LOG="$SCRIPT_DIR/logs/ngrok_${PORT}.log"
URL="$(grep -oE 'url=https://[^[:space:]\"]+' "$NGROK_LOG" 2>/dev/null | sed 's/^url=//' | grep -Fx 'https://backtrack-emcee-numbness.ngrok-free.dev' | tail -n 1 || true)"
if [[ -n "$URL" ]]; then
    printf 'URL ngrok: %s\n' "$URL"
else
    printf 'URL ngrok: belum tersedia\n'
fi
