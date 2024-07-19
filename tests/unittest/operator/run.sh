#!/bin/bash
set -e

files=$(find . -name "test_*.py")
for f in ${files[@]};
do
    start_time=$(date +%s)
    pytest $f
    end_time=$(date +%s)
    cost=$[ $end_time-$start_time ]
    echo "$f cost: $cost"
done
