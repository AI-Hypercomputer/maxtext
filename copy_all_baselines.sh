step_array=(200 400 1000 2000 3000)
prng_key_array=(0 1 3)

# step_array=(200)
# prng_key_array=(3)

base_directory=gs://maxtext-experiments-multipod/mattdavidow-sweep-baselines-a2_PRNGKey_

for step in ${step_array[@]}; do
    for prng_key in ${prng_key_array[@]}; do
        output_directory=gs://maxtext-experiments-multipod/mattdavidow-int8-baseline_step_${step}_PRNGKey_${prng_key}
        directory=${base_directory}${prng_key}
        gsutil -m cp -r $directory/tensorboard $output_directory/tensorboard
        gsutil -m cp -r $directory/checkpoints/$step $output_directory/checkpoints/$step
    done
done
