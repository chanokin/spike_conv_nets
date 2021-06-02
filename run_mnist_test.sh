#!/usr/bin/env bash

source ../venv3/bin/activate
gSTART=1000
N_DIGITS=1
N_TEST=10000

for ((START = $gSTART; START < $N_TEST; START += $N_DIGITS))
do
  python simple_cnn_mnist.py $START $N_DIGITS $N_TEST
  sleep 1
done
