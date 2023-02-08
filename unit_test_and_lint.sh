#!/bin/bash
pylint MaxText/

cd MaxText
python3 -m pytest
