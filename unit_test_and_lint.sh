#!/bin/bash

# SPDX-License-Identifier: Apache-2.0

python3 -m pylint $(git ls-files '*.py')

python3 -m pytest --pyargs MaxText.tests
