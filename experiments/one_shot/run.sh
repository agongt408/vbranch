# !/bin/sh

# python experiments/one_shot/baseline.py --architecture simple --epochs 90 --trials 8
# python experiments/one_shot/baseline.py --architecture simple --trials 8 --model_id 1 2 3 4 5 --test

# python experiments/one_shot/baseline.py --architecture simple --model_id 1 --test
# python experiments/one_shot/baseline.py --architecture simple --model_id 2 --test
# python experiments/one_shot/baseline.py --architecture simple --model_id 3 --test
# python experiments/one_shot/baseline.py --architecture simple --model_id 4 --test
# python experiments/one_shot/baseline.py --architecture simple --model_id 5 --test
# python experiments/one_shot/baseline.py --architecture simple --model_id 6 --test
# python experiments/one_shot/baseline.py --architecture simple --model_id 7 --test
# python experiments/one_shot/baseline.py --architecture simple --model_id 8 --test

# python experiments/one_shot/vbranch.py --shared_frac 0 --num_branches 2 | tee -a logs/train-omniglot.log
# python experiments/one_shot/vbranch.py --shared_frac 0.25 --num_branches 2 | tee -a logs/train-omniglot.log
# python experiments/one_shot/vbranch.py --shared_frac 0.5 --num_branches 2 | tee -a logs/train-omniglot.log
# python experiments/one_shot/vbranch.py --shared_frac 0.75 --num_branches 2 | tee -a logs/train-omniglot.log
# python experiments/one_shot/vbranch.py --shared_frac 1 --num_branches 2 | tee -a logs/train-omniglot.log
#
# python experiments/one_shot/vbranch.py --shared_frac 0 --num_branches 3 | tee -a logs/train-omniglot.log
# python experiments/one_shot/vbranch.py --shared_frac 0.25 --num_branches 3 | tee -a logs/train-omniglot.log
# python experiments/one_shot/vbranch.py --shared_frac 0.5 --num_branches 3 | tee -a logs/train-omniglot.log
# python experiments/one_shot/vbranch.py --shared_frac 0.75 --num_branches 3 | tee -a logs/train-omniglot.log
# python experiments/one_shot/vbranch.py --shared_frac 1 --num_branches 3 | tee -a logs/train-omniglot.log
#
# python experiments/one_shot/vbranch.py --shared_frac 0 --num_branches 4 | tee -a logs/train-omniglot.log
# python experiments/one_shot/vbranch.py --shared_frac 0.25 --num_branches 4 | tee -a logs/train-omniglot.log
# python experiments/one_shot/vbranch.py --shared_frac 0.5 --num_branches 4 | tee -a logs/train-omniglot.log
# python experiments/one_shot/vbranch.py --shared_frac 0.75 --num_branches 4 | tee -a logs/train-omniglot.log
# python experiments/one_shot/vbranch.py --shared_frac 1 --num_branches 4 | tee -a logs/train-omniglot.log

python experiments/one_shot/vbranch.py --shared_frac 0 --num_branches 2 --test
python experiments/one_shot/vbranch.py --shared_frac 0.25 --num_branches 2 --test
python experiments/one_shot/vbranch.py --shared_frac 0.5 --num_branches 2 --test
python experiments/one_shot/vbranch.py --shared_frac 0.75 --num_branches 2 --test
python experiments/one_shot/vbranch.py --shared_frac 1 --num_branches 2 --test

python experiments/one_shot/vbranch.py --shared_frac 0 --num_branches 3 --test
python experiments/one_shot/vbranch.py --shared_frac 0.25 --num_branches 3 --test
python experiments/one_shot/vbranch.py --shared_frac 0.5 --num_branches 3 --test
python experiments/one_shot/vbranch.py --shared_frac 0.75 --num_branches 3 --test
python experiments/one_shot/vbranch.py --shared_frac 1 --num_branches 3 --test

python experiments/one_shot/vbranch.py --shared_frac 0 --num_branches 4 --test
python experiments/one_shot/vbranch.py --shared_frac 0.25 --num_branches 4 --test
python experiments/one_shot/vbranch.py --shared_frac 0.5 --num_branches 4 --test
python experiments/one_shot/vbranch.py --shared_frac 0.75 --num_branches 4 --test
python experiments/one_shot/vbranch.py --shared_frac 1 --num_branches 4 --test
