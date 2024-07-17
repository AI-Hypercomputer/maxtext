# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages

def parse_requirements(filename):
  """load requirements from a pip requirements file."""
  with open(filename) as f:
    lineiter = [line.strip() for line in f]
    return lineiter

setup(
    name='maxtext',
    version='0.1.0',
    author='Google',
    description='MaxText is a high performance, highly scalable, open-source LLM written in pure Python/Jax and targeting Google Cloud TPUs and GPUs for training and inference',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/google/maxtext',
    packages=find_packages(where='MaxText', exclude=['configs','scratch_code','tests','test_assets']),
    package_dir={'': 'MaxText'},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
    install_requires=parse_requirements("requirements.txt"),
)