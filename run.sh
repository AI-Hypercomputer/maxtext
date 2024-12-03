#!/bin/bash

docker run --rm -it -v $(pwd):/deps -v $(pwd)/../mnt:/checkpoint dev
