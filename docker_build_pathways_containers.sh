#!/bin/bash

# Copyright 2023 Google LLC
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

# apt-get install docker
# apt-get install docker-compose-plugin
# docker compose version
# gcloud auth configure-docker us-docker.pkg.dev --quiet

cd MaxText/utils_pathways
docker compose down
sleep 10
docker compose up &
sleep 20
docker compose ps
cd ..
echo "Docker compose installation complete!"