#!/bin/bash
pids=(1 2 3)
if [ ! ${#pids[@]} -eq 0 ] 
then
    echo "go me"
else
 echo -e "No existing processes found, so your TPU is ready to use!"
fi
