#!/usr/bin/env bash
set -Eeuo pipefail

PORT="${1:-8501}"
NGROK_URL="https://backtrack-emcee-numbness.ngrok-free.dev"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
PID_DIR="$LOG_DIR/pids"
STREAMLIT_PID_FILE="$PID_DIR/streamlit_${PORT}.pid"
NGROK_PID_FILE="$PID_DIR/ngrok_${PORT}.pid"
STREAMLIT_LOG="$LOG_DIR/streamlit_${PORT}.log"
NGROK_LOG="$LOG_DIR/ngrok_${PORT}.log"

if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
    printf 'Port tidak valid: %s\n' "$PORT" >&2
    exit 2
fi

if ! command -v ngrok >/dev/null 2>&1; then
    printf 'ngrok belum terpasang atau tidak ada di PATH. Instal ngrok dan konfigurasi authtoken terlebih dahulu.\n' >&2
    exit 1
fi

if [[ -x "$SCRIPT_DIR/venv/bin/python" ]]; then
    PYTHON="$SCRIPT_DIR/venv/bin/python"
else
    PYTHON="$(command -v python3 || true)"
fi

if [[ -z "$PYTHON" ]]; then
    printf 'Python 3 tidak ditemukan.\n' >&2
    exit 1
fi

if ! "$PYTHON" -c 'import streamlit' >/dev/null 2>&1; then
    printf 'Streamlit tidak tersedia pada %s. Install dependencies proyek terlebih dahulu.\n' "$PYTHON" >&2
    exit 1
fi

mkdir -p "$PID_DIR"
cd "$SCRIPT_DIR"

if [[ -f "$STREAMLIT_PID_FILE" ]] && kill -0 "$(<"$STREAMLIT_PID_FILE")" 2>/dev/null; then
    printf 'Streamlit tampaknya sudah berjalan (PID %s). Hentikan dulu dengan ./stop_web_background.sh %s\n' "$(<"$STREAMLIT_PID_FILE")" "$PORT" >&2
    exit 1
fi
if [[ -f "$NGROK_PID_FILE" ]] && kill -0 "$(<"$NGROK_PID_FILE")" 2>/dev/null; then
    printf 'Ngrok tampaknya sudah berjalan (PID %s). Hentikan dulu dengan ./stop_web_background.sh %s\n' "$(<"$NGROK_PID_FILE")" "$PORT" >&2
    exit 1
fi
if command -v curl >/dev/null 2>&1 && curl -fsS "http://127.0.0.1:$PORT/_stcore/health" >/dev/null 2>&1; then
    STREAMLIT_PID="$("$PYTHON" - "$PORT" "$SCRIPT_DIR" <<'PY'
import os
import subprocess
import sys

port, project_dir = sys.argv[1], os.path.realpath(sys.argv[2])
for row in subprocess.check_output(["ps", "-eo", "pid=,args="], text=True).splitlines():
    pid, _, args = row.strip().partition(" ")
    if "streamlit run app.py" not in args or f"--server.port {port}" not in args:
        continue
    if os.path.realpath(f"/proc/{pid}/cwd") == project_dir:
        print(pid)
        break
PY
    )"
    if [[ -z "$STREAMLIT_PID" ]]; then
        printf 'Port %s merespons sebagai Streamlit, tetapi bukan app.py dari folder proyek ini. Ngrok tidak dijalankan.\n' "$PORT" >&2
        exit 1
    fi
    printf '%s\n' "$STREAMLIT_PID" >"$STREAMLIT_PID_FILE"
    printf 'Memakai Streamlit ColonoMind yang sudah aktif (PID %s).\n' "$STREAMLIT_PID"
else
    if "$PYTHON" -c 'import socket,sys
try:
    socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=1).close()
except OSError:
    raise SystemExit(1)
raise SystemExit(0)' "$PORT"; then
        printf 'Port %s sudah dipakai layanan lain. Ngrok tidak dijalankan.\n' "$PORT" >&2
        exit 1
    fi

    nohup "$PYTHON" -m streamlit run "$SCRIPT_DIR/app.py" \
        --server.port "$PORT" \
        --server.address 127.0.0.1 \
        --server.headless true >"$STREAMLIT_LOG" 2>&1 </dev/null &
    STREAMLIT_PID=$!
    printf '%s\n' "$STREAMLIT_PID" >"$STREAMLIT_PID_FILE"

    for _ in {1..30}; do
        if ! kill -0 "$STREAMLIT_PID" 2>/dev/null; then
            printf 'Streamlit gagal mulai. Lihat %s\n' "$STREAMLIT_LOG" >&2
            rm -f "$STREAMLIT_PID_FILE"
            exit 1
        fi
        if command -v curl >/dev/null 2>&1 && curl -fsS "http://127.0.0.1:$PORT/_stcore/health" >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done

    if ! command -v curl >/dev/null 2>&1 || ! curl -fsS "http://127.0.0.1:$PORT/_stcore/health" >/dev/null 2>&1; then
        printf 'Streamlit belum merespons pada port %s. Lihat %s\n' "$PORT" "$STREAMLIT_LOG" >&2
        kill "$STREAMLIT_PID" 2>/dev/null || true
        rm -f "$STREAMLIT_PID_FILE"
        exit 1
    fi
fi

nohup ngrok http "http://127.0.0.1:$PORT" --log=stdout --log-format=logfmt >"$NGROK_LOG" 2>&1 </dev/null &
NGROK_PID=$!
printf '%s\n' "$NGROK_PID" >"$NGROK_PID_FILE"

for _ in {1..20}; do
    if ! kill -0 "$NGROK_PID" 2>/dev/null; then
        break
    fi
    ACTIVE_NGROK_URL="$(grep -oE 'url=https://[^[:space:]\"]+' "$NGROK_LOG" 2>/dev/null | sed 's/^url=//' | grep -Fx "$NGROK_URL" | tail -n 1 || true)"
    [[ -n "$ACTIVE_NGROK_URL" ]] && break
    sleep 1
done

if [[ -z "$ACTIVE_NGROK_URL" ]]; then
    printf 'Ngrok belum menghasilkan URL publik. Log: %s\n' "$NGROK_LOG" >&2
    printf 'Streamlit masih berjalan di http://127.0.0.1:%s (PID %s). Jalankan ./stop_web_background.sh %s untuk menghentikan.\n' "$PORT" "$STREAMLIT_PID" "$PORT" >&2
    exit 1
fi

printf 'Streamlit berjalan di http://127.0.0.1:%s (PID %s)\n' "$PORT" "$STREAMLIT_PID"
printf 'URL ngrok: %s\n' "$NGROK_URL"
printf 'Log: %s dan %s\n' "$STREAMLIT_LOG" "$NGROK_LOG"
