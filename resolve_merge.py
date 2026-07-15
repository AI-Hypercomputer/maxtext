with open('.github/workflows/ci_pipeline.yml', 'r') as f:
    c = f.read()

# 1. Resolve tpu-tests conflict
tpu_old = '''<<<<<<< HEAD
    needs: [analyze_code_changes, build_and_upload_maxtext_package, gate_test_run]
    if: |
      always() &&
      needs.analyze_code_changes.outputs.run_tests == 'true' &&
      needs.build_and_upload_maxtext_package.result == 'success'

=======
    needs: [gate_test_run]
    if: |
      always() &&
      needs.gate_test_run.result == 'success'
>>>>>>> main'''

tpu_new = '''    needs: [gate_test_run]
    if: |
      always() &&
      needs.gate_test_run.result == 'success''''

if tpu_old in c:
    c = c.replace(tpu_old, tpu_new, 1)

# 2. Resolve tpu7x-tests conflict
tpu7x_old = '''<<<<<<< HEAD
    needs: [analyze_code_changes, build_and_upload_maxtext_package, gate_test_run]
    if: |
      always() &&
      needs.build_and_upload_maxtext_package.result == 'success' &&
      (github.event_name != 'pull_request' || needs.analyze_code_changes.outputs.has_new_tests == 'true') &&
      (github.event_name == 'pull_request' || github.event_name == 'schedule' || github.event_name == 'workflow_dispatch')
=======
    needs: [gate_test_run]
    if: |
      always() &&
      needs.gate_test_run.result == 'success' &&
      github.ref == 'refs/heads/main' && (github.event_name == 'schedule' || github.event_name == 'workflow_dispatch')
>>>>>>> main'''

tpu7x_new = '''    needs: [analyze_code_changes, gate_test_run]
    if: |
      always() &&
      needs.gate_test_run.result == 'success' &&
      (github.event_name != 'pull_request' || needs.analyze_code_changes.outputs.has_new_tests == 'true') &&
      (github.event_name == 'pull_request' || github.event_name == 'schedule' || github.event_name == 'workflow_dispatch')'''

if tpu7x_old in c:
    c = c.replace(tpu7x_old, tpu7x_new, 1)

# 3. Resolve gpu-tests conflict
gpu_old = '''<<<<<<< HEAD
    needs: [analyze_code_changes, build_and_upload_maxtext_package, gate_test_run]
    if: |
      always() &&
      needs.analyze_code_changes.outputs.run_tests == 'true' &&
      needs.build_and_upload_maxtext_package.result == 'success'

=======
    needs: [gate_test_run]
    if: |
      always() &&
      needs.gate_test_run.result == 'success'
>>>>>>> main'''

gpu_new = '''    needs: [gate_test_run]
    if: |
      always() &&
      needs.gate_test_run.result == 'success''''

if gpu_old in c:
    c = c.replace(gpu_old, gpu_new, 1)

# 4. Resolve cpu-tests conflict
cpu_old = '''<<<<<<< HEAD
    needs: [analyze_code_changes, build_and_upload_maxtext_package, gate_test_run]
    if: |
      always() &&
      needs.analyze_code_changes.outputs.run_tests == 'true' &&
      needs.build_and_upload_maxtext_package.result == 'success'

=======
    needs: [gate_test_run]
    if: |
      always() &&
      needs.gate_test_run.result == 'success'
>>>>>>> main'''

cpu_new = '''    needs: [gate_test_run]
    if: |
      always() &&
      needs.gate_test_run.result == 'success''''

if cpu_old in c:
    c = c.replace(cpu_old, cpu_new, 1)

with open('.github/workflows/ci_pipeline.yml', 'w') as f:
    f.write(c)
print('Resolved ci_pipeline.yml conflicts cleanly!')
