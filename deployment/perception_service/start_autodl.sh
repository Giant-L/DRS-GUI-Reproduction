#!/usr/bin/env bash
set -euo pipefail

PERSIST_ROOT="${AUTODL_PERSIST_ROOT:-/root/autodl-tmp}"
PROJECT_ROOT="${PROJECT_ROOT:-${PERSIST_ROOT}/DRS-GUI-Reproduction}"
OMNIPARSER_ROOT="${OMNIPARSER_ROOT:-${PERSIST_ROOT}/OmniParser}"
PID_FILE="${PERCEPTION_PID_FILE:-${PERSIST_ROOT}/perception.pid}"
LOG_FILE="${PERCEPTION_LOG_FILE:-${PERSIST_ROOT}/perception.log}"

if [[ -f "${PID_FILE}" ]]; then
    existing_pid="$(<"${PID_FILE}")"
    if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && (( existing_pid > 1 )) \
        && kill -0 "${existing_pid}" 2>/dev/null; then
        echo "perception service already running (pid=${existing_pid})"
        exit 0
    fi
fi

cd "${PROJECT_ROOT}"
export HF_HOME="${HF_HOME:-${PERSIST_ROOT}/models/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export OMNIPARSER_ROOT
export OMNIPARSER_DETECTOR_PATH="${OMNIPARSER_DETECTOR_PATH:-${OMNIPARSER_ROOT}/weights/icon_detect/model.pt}"
export OMNIPARSER_CAPTION_PATH="${OMNIPARSER_CAPTION_PATH:-${OMNIPARSER_ROOT}/weights/icon_caption_florence}"
export OMNIPARSER_PROCESSOR_PATH="${OMNIPARSER_PROCESSOR_PATH:-${PERSIST_ROOT}/models/florence-2-base}"
export INSTRUCTOR_MODEL_PATH="${INSTRUCTOR_MODEL_PATH:-${PERSIST_ROOT}/models/instructor-large}"
export INSTRUCTOR_DEVICE="${INSTRUCTOR_DEVICE:-cuda}"
export EASYOCR_MODULE_PATH="${EASYOCR_MODULE_PATH:-${PERSIST_ROOT}/models/easyocr}"

nohup python -m uvicorn deployment.perception_service.app:app \
    --host 127.0.0.1 --port "${PERCEPTION_PORT:-8010}" \
    >"${LOG_FILE}" 2>&1 &
service_pid=$!
echo "${service_pid}" >"${PID_FILE}"
sleep 2
kill -0 "${service_pid}"
echo "perception service started (pid=${service_pid}, log=${LOG_FILE})"
