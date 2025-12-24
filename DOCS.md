<!--
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
-->
Documentation… documentation!
=============================

## Dependencies
First install the dependencies:
```sh
$ python3 -m pip install -r requirements_docs.txt
```
(or `uv pip install` …)

## Build
```sh
$ sphinx-build -M html docs out
```

## Serve
You can use any static file HTTP server, e.g.:
```sh
$ python3 -m http.server -d 'out/html'
```

## Build & server (watch for changes)
```sh
$ python3 -m pip install sphinx-autobuild
$ sphinx-autobuild docs out
```

## Release to readthedocs

See GitHub Action
