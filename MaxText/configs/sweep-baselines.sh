
prng_key_array=(0 1 3)

echo "starting loops"
for prng_key in ${prng_key_array[@]}; do
    RUN_NAME=mattdavidow-sweep-baselines-a2_PRNGKey_${prng_key}
    echo ${RUN_NAME}
    echo ${RUN_NAME} >> mattdavidow-sweep-baselines-a2-run-names.txt
    python3 multihost_job.py --BUCKET_NAME="mattdavidow-maxtext-br" --RUN_NAME=${RUN_NAME} --TPU_TYPE=v5litepod-256 --NUM_SLICES=1 --VERSION=v2-alpha-tpuv5-lite --COMMAND="bash setup.sh && bash MaxText/configs/sweep-baseline-individual.sh ${steps} ${prng_key} ${RUN_NAME}" --ZONE=us-east5-b      
done
echo "finished loops"