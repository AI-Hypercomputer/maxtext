
fwd_int8_array=("false")
bwd_int8_array=("false")
step_array=(400 2000)
prng_key_array=(0 1)

echo "starting loops"
for fwd_int8 in ${fwd_int8_array[@]}; do
    for bwd_int8 in ${bwd_int8_array[@]}; do
        multijob_run_name=mattdavidow-sweep2p2-mj-a2_fwd_${fwd_int8}_bwd_${bwd_int8}_multijob
        cmd=""
        for step in ${step_array[@]}; do
            for prng_key in ${prng_key_array[@]}; do
                run_name=mattdavidow-sweep2p2-mj-a2_fwd_${fwd_int8}_bwd_${bwd_int8}_step_${step}_PRNGKey_${prng_key}
                #echo ${run_name}
                echo ${run_name} >> mattdavidow-sweep2p2-run-names-a1.txt
            cmd=$cmd" && echo 'starting $run_name' && bash MaxText/configs/sweep2p2-individual.sh ${fwd_int8} ${bwd_int8} ${step} ${prng_key} ${run_name} && sleep 60"
                
            done
        done
        #echo $cmd

        python3 multihost_job.py --BUCKET_NAME="mattdavidow-maxtext-br" --RUN_NAME=${multijob_run_name} --TPU_TYPE=v5litepod-256 --NUM_SLICES=1 --VERSION=v2-alpha-tpuv5-lite --COMMAND="bash setup.sh $cmd" --ZONE=us-east5-b      
    done
done
echo "finished loops"