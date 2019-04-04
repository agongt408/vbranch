#!/bin/sh

# Run experiment with various configs
# https://stackoverflow.com/questions/24207916/catching-the-exception-thrown-by-python-script-in-shell-script

python experiment-vb.py --num_branches 4 --model_id 1 | tee train.log
python experiment-vb.py --num_branches 4 --model_id 2 | tee -a train.log
python experiment-vb.py --num_branches 4 --model_id 3 | tee -a train.log
python experiment-vb.py --num_branches 4 --model_id 4 | tee -a train.log

python experiment.py --model_id 1 | tee -a train.log
python experiment.py --model_id 2 | tee -a train.log
python experiment.py --model_id 3 | tee -a train.log
python experiment.py --model_id 4 | tee -a train.log
python experiment.py --model_id 5 | tee -a train.log
python experiment.py --model_id 6 | tee -a train.log
python experiment.py --model_id 7 | tee -a train.log
python experiment.py --model_id 8 | tee -a train.log
