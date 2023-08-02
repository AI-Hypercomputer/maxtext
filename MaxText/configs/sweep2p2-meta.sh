
fwd_int8_array=("false" "true")
bwd_int8_array=("false" "true")
step_array=(200 400 1000 2000 3000)
prng_key_array=(0 1)

echo "starting loops"
for fwd_int8 in ${fwd_int8_array[@]}; do
    for bwd_int8 in ${bwd_int8_array[@]}; do
        for step in ${step_array[@]}; do
            for PRNG_KEY in ${PRNG_KEY_ARRAY[@]}; do
                run_name=mattdavidow-sweep2p2-a1_fwd_${fwd_int8}_bwd_${bwd_int8}_step_${step}_PRNGKey_${prng_key}
                echo ${run_name}
                echo ${run_name} >> mattdavidow-sweep2p2-run-names-a1.txt
                python3 multihost_job.py --BUCKET_NAME="mattdavidow-maxtext-br" --RUN_NAME=${run_name} --TPU_TYPE=v5litepod-256 --NUM_SLICES=1 --VERSION=v2-alpha-tpuv5-lite --COMMAND="bash setup.sh && bash MaxText/configs/sweep2p2-individual.sh ${fwd_int8} ${bwd_int8} ${step} ${prng_key} ${run_name}" --ZONE=us-east5-b      
            done
        done
    done
done
echo "finished loops"