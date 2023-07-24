# Compile legacy JSonnet templates into JSON

rm -rf configs/jsonnet
jsonnet -J ./ml-testing-accelerators --create-output-dirs --multi configs/jsonnet ml-testing-accelerators/tests/all_tests.jsonnet
