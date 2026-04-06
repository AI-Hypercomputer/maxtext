#!/bin/bash

# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script generates feature.proto and example.proto
# Install protoc with a version matching your protobuf version.
# For example, to install protoc 29.5 that matches protobuf 5.29.5:
# PROTOC_VERSION=29.5
# PROTOC_ARCH=$(uname -m)
# if [ "$PROTOC_ARCH" = "x86_64" ]; then ARCH="x86_64"; elif [ "$PROTOC_ARCH" = "aarch64" ]; then ARCH="aarch_64"; fi
# URL="https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/protoc-${PROTOC_VERSION}-linux-${ARCH}.zip"
# curl -L $URL -o protoc.zip && unzip protoc.zip -d protoc_temp
# cp protoc_temp/bin/protoc ~/.local/bin/protoc && cp -r protoc_temp/include/* ~/.local/include/
# rm -rf protoc.zip protoc_temp

# Under the maxtext/src
cd ./src

# Compile
protoc --proto_path=. --python_out=. \
maxtext/input_pipeline/protos/feature.proto \
maxtext/input_pipeline/protos/example.proto
