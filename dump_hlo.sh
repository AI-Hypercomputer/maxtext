DUMP_NAME=${1}

if [[ ${SLICE_ID} -eq 0 && ${WORKER_ID} -eq 0 ]]; then
    echo "Dumping HLO"
    cd /tmp/hlo-dumps/${DUMP_NAME}
    tar -czf  /tmp/hlo_dump_${DUMP_NAME}_zip.tar.gz ./
    gsutil cp /tmp/hlo_dump_${DUMP_NAME}_zip.tar.gz gs://mattdavidow-hlo-dumps
fi
