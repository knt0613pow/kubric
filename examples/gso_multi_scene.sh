#!/bin/bash


for ((i=0; i<=100; i++))
do
    echo "Running gso2.py with --scene_idx=$i"
    python examples/gso2.py --scene_idx $i
done