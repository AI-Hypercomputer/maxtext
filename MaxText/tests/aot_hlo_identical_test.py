"""
Copyright 2024 Google LLC
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
     https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import unittest
import pytest
import os
import shutil
import hashlib
import sys
import subprocess


class AotHloIdenticalTest(unittest.TestCase):
    def run_compile_and_real(self, script_path, aot_dump_dir, real_dump_dir, extra_config_args):
        """Executes a Bash script and returns the completed process object."""
        if extra_config_args is not None:
            cmd = ["bash", script_path, aot_dump_dir, real_dump_dir, extra_config_args]
        else:
            cmd = ["bash", script_path, aot_dump_dir, real_dump_dir]
        try:
            result = subprocess.run(
                cmd,                # Command to run the script
                check=True,         # Raise an exception if the script fails
                stdout=sys.stdout,  # Stream to stdout
                stderr=sys.stdout,  # Stream to stdout
                text=True           # Decode output and error as text
            )
            return result
        except subprocess.CalledProcessError as e:
            print(f"Error running script: {e.returncode}")
            print(f"Output: {e.stdout}")
            print(f"Error: {e.stderr}")

    def find_file_by_substring(self, directory, substring):
        for filename in os.listdir(directory):
            if substring in filename:
                return os.path.join(directory,filename)
        raise FileNotFoundError(f"Could not find a file in directory {directory} with substring {substring}")

    def delete_dir(self, dir):
        if os.path.exists(dir):
            shutil.rmtree(dir)

    def check_large_files_equal(self, file_path1, file_path2):
        """Asserts that two potentially large text files have identical content."""

        hasher1 = hashlib.sha256()
        hasher2 = hashlib.sha256()

        with open(file_path1, "rb") as f1, open(file_path2, "rb") as f2:
            # Read files in chunks for memory efficiency
            while True:
                chunk1 = f1.read(8192)  # 8 KB chunks
                chunk2 = f2.read(8192)

                if not chunk1 and not chunk2:  # Reached the end of both files
                    break
                hasher1.update(chunk1)
                hasher2.update(chunk2)

        # Handle potential empty files
        if not hasher1.digest() or not hasher2.digest():
            # One or both files are empty
            return False

        if hasher1.hexdigest() != hasher2.hexdigest():
            # Files have different contents
            return False
        return True

    def assert_compile_and_real_match_hlo(self, test_name, extra_config_args):
        hlo_filename_substring="jit_train_step.after_optimizations_after_buffer_assignment.txt"
        compile_dump_dir=f"/tmp/compile_test_xla_dump/{test_name}/aot/"
        train_dump_dir=f"/tmp/compile_test_xla_dump/{test_name}/real/"
        self.delete_dir(compile_dump_dir) # Ensure directories empty before use
        self.delete_dir(train_dump_dir)

        self.run_compile_and_real("tests/aot_hlo_identical_script.sh", compile_dump_dir, train_dump_dir, extra_config_args)

        compile_hlo_file = self.find_file_by_substring(compile_dump_dir, hlo_filename_substring)
        train_hlo_file = self.find_file_by_substring(train_dump_dir, hlo_filename_substring)
        print(f"AOT compiled HLO file for test {test_name}: {compile_hlo_file}", flush=True)
        print(f"Real runs HLO file for test {test_name}: {train_hlo_file}", flush=True)

        files_equal = self.check_large_files_equal(compile_hlo_file, train_hlo_file)
        self.delete_dir(compile_dump_dir) # Cleanup directories after use
        self.delete_dir(train_dump_dir)
        assert files_equal, f"AOT Compiled and real HLO files are not identical for test {test_name}!"
        print("AOT Compiled and train HLO files are identical for test {test_name}!")

    @pytest.mark.tpu
    def test_default_hlo_match(self):
        self.assert_compile_and_real_match_hlo("default_run", None)

    @pytest.mark.tpu
    def test_int8_hlo_match(self):
        self.assert_compile_and_real_match_hlo("int8", "quantization=int8")
