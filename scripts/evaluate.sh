#!/bin/bash

orig=$1
trg=$2
hyp=$3
ref_m2=${orig%src}m2
hyp_m2=$hyp.m2

python3 ~/errant/errant/commands/parallel_to_m2.py -orig $orig -cor $trg -out $ref_m2 -tok
python3 ~/errant/errant/commands/parallel_to_m2.py -orig $orig -cor $hyp -out $hyp_m2 -tok
python3 ~/errant/errant/commands/compare_m2.py -ref $ref_m2 -hyp $hyp_m2 -v
