#!/bin/bash
OMP_NUM_THREADS=1 python main.py --num-processes 4 --max-episode-length 100 --eta 0.00 --size 10
