set -e
echo "Copying HLO Runner"
gcloud storage cp gs://mohitkhatwani-logs/host_offloading_2024-11-01-19-38/cloud_hlo_runner/cloud_hlo_runner .
chmod +x cloud_hlo_runner

echo "Copying HLO modules"

gcloud storage cp gs://mohitkhatwani-logs/host_offloading_2024-11-01-19-38/gke-tpu-20495c71-085p/module_0012.jit__unnamed_wrapped_function_.before_optimizations.txt .
gcloud storage cp gs://mohitkhatwani-logs/host_offloading_2024-11-01-19-38/gke-tpu-20495c71-085p/module_0012.jit__unnamed_wrapped_function_.execution_options.txt .
gcloud storage cp gs://mohitkhatwani-logs/host_offloading_2024-11-01-19-38/gke-tpu-20495c71-085p/module_0012.jit__unnamed_wrapped_function_.tpu_comp_env.txt .

pip install --pre --upgrade libtpu -f https://storage.googleapis.com/libtpu-wheels/index.html

export TPU_LIBRARY_PATH=$(python3 -c "import libtpu;import os;libtpu_path=os.path.join(os.path.dirname(libtpu.__file__), 'libtpu.so');print(libtpu_path)")

unset LD_PRELOAD

./cloud_hlo_runner module_0012.jit__unnamed_wrapped_function_.before_optimizations.txt --execution_options_path module_0012.jit__unnamed_wrapped_function_.execution_options.txt --tpu_compilation_env_file_path module_0012.jit__unnamed_wrapped_function_.tpu_comp_env.txt --tpu_profile_dump_path $HOME/profile.xspace