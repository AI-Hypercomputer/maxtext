#!/bin/bash
# GPT-3 175B Model.
# Train GPT-3 175B on v5p-12288 slice, with a custom topology of 8x16x48.

# Example to invoke this script:
# bash src/MaxText/configs/v5p/gpt3_175b/v5p_12288.sh YOUR_RUN gs://YOUR_BUCKET"

set -euox pipefail

# Read arguments or use defaults from environment variables
RUNNAME=${1:-${RUNNAME:-some-run}}
BASE_OUTPUT_DIRECTORY=${2:-${BASE_OUTPUT_DIRECTORY:-gs://some-bucket}}

chmod +x "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}"/v5p/gpt3_175b/gpt3_175b_base.sh
./MaxText/configs/v5p/gpt3_175b/gpt3_175b_base.sh 1 "minimal" 16 48 8 "${RUNNAME}" "${BASE_OUTPUT_DIRECTORY}" 