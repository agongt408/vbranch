# !/bin/sh

# Run experiment with various configs
# https://stackoverflow.com/questions/24207916/catching-the-exception-thrown-by-python-script-in-shell-script

# python experiments/classification/baseline.py --architecture fcn --epochs 30 --trials 8 | tee -a logs/train.log
#
# python experiments/classification/vbranch.py --architecture fcn --epochs 30 --trials 4 --shared_frac 0 --num_branches 4 | tee -a logs/train.log
# python experiments/classification/vbranch.py --architecture fcn --epochs 30 --trials 4 --shared_frac 0.25 --num_branches 4 | tee -a logs/train.log
# python experiments/classification/vbranch.py --architecture fcn --epochs 30 --trials 4 --shared_frac 0.5 --num_branches 4 | tee -a logs/train.log
# python experiments/classification/vbranch.py --architecture fcn --epochs 30 --trials 4 --shared_frac 0.75 --num_branches 4 | tee -a logs/train.log
# python experiments/classification/vbranch.py --architecture fcn --epochs 30 --trials 4 --shared_frac 1 --num_branches 4 | tee -a logs/train.log

# python experiments/classification/baseline.py --architecture cnn --epochs 30 --trials 8 | tee -a logs/train.log
#
# python experiments/classification/vbranch.py --architecture cnn --epochs 30 --trials 4 --shared_frac 0 --num_branches 4 | tee -a logs/train.log
# python experiments/classification/vbranch.py --architecture cnn --epochs 30 --trials 4 --shared_frac 0.25 --num_branches 4 | tee -a logs/train.log
# python experiments/classification/vbranch.py --architecture cnn --epochs 30 --trials 4 --shared_frac 0.5 --num_branches 4 | tee -a logs/train.log
# python experiments/classification/vbranch.py --architecture cnn --epochs 30 --trials 4 --shared_frac 0.75 --num_branches 4 | tee -a logs/train.log
# python experiments/classification/vbranch.py --architecture cnn --epochs 30 --trials 4 --shared_frac 1 --num_branches 4 | tee -a logs/train.log

# python experiments/classification/baseline.py --architecture cnn --trials 8 --model_id 1 2 3 4 --test