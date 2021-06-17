#!/usr/bin/env bash

if [ "$#" -eq 1 ]; then
    source $1/bin/activate
else
    source ../venv3/bin/activate
fi

gSTART=0
N_DIGITS=10
N_TEST=20

for ((START = $gSTART; START < $N_TEST; START += $N_DIGITS))
do
  python simple_cnn_mnist.py $START $N_DIGITS $N_TEST
  sleep 1
done
