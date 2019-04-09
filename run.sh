# !/bin/sh

# Run experiment with various configs
# https://stackoverflow.com/questions/24207916/catching-the-exception-thrown-by-python-script-in-shell-script

# python experiment-vb.py --num_branches 4 --model_id 1 | tee logs/train.log
# python experiment-vb.py --num_branches 4 --model_id 2 | tee -a logs/train.log
# python experiment-vb.py --num_branches 4 --model_id 3 | tee -a logs/train.log
# python experiment-vb.py --num_branches 4 --model_id 4 | tee -a logs/train.log

# python experiment.py --model_id 1 | tee -a logs/train.log
# python experiment.py --model_id 2 | tee -a logs/train.log
# python experiment.py --model_id 3 | tee -a logs/train.log
# python experiment.py --model_id 4 | tee -a logs/train.log
# python experiment.py --model_id 5 | tee -a logs/train.log
# python experiment.py --model_id 6 | tee -a logs/train.log
# python experiment.py --model_id 7 | tee -a logs/train.log
# python experiment.py --model_id 8 | tee -a logs/train.log

# python experiment.py --test 1 2 3 7 | tee logs/test.log
# python experiment.py --test 1 3 7 8 | tee -a logs/test.log
# python experiment.py --test 1 4 7 8 | tee -a logs/test.log
# python experiment.py --test 1 2 4 8 | tee -a logs/test.log

# python experiment-vb.py --num_branches 4 --model_id 1 --architecture cnn | tee logs/train-cnn.log
# python experiment-vb.py --num_branches 4 --model_id 2 --architecture cnn | tee -a logs/train-cnn.log
# python experiment-vb.py --num_branches 4 --model_id 3 --architecture cnn | tee -a logs/train-cnn.log
# python experiment-vb.py --num_branches 4 --model_id 4 --architecture cnn | tee -a logs/train-cnn.log

# python experiment.py --model_id 1 --architecture cnn | tee -a logs/train-cnn.log
# python experiment.py --model_id 2 --architecture cnn | tee -a logs/train-cnn.log
# python experiment.py --model_id 3 --architecture cnn | tee -a logs/train-cnn.log
# python experiment.py --model_id 4 --architecture cnn | tee -a logs/train-cnn.log
# python experiment.py --model_id 5 --architecture cnn | tee -a logs/train-cnn.log
# python experiment.py --model_id 6 --architecture cnn | tee -a logs/train-cnn.log
# python experiment.py --model_id 7 --architecture cnn | tee -a logs/train-cnn.log
# python experiment.py --model_id 8 --architecture cnn | tee -a logs/train-cnn.log

# python experiment-vb.py --num_branches 4 --model_id 1 --architecture cnn --epochs 30 | tee logs/train-cnn.log
# python experiment-vb.py --num_branches 4 --model_id 2 --architecture cnn --epochs 30 | tee -a logs/train-cnn.log
# python experiment-vb.py --num_branches 4 --model_id 3 --architecture cnn --epochs 30 | tee -a logs/train-cnn.log
# python experiment-vb.py --num_branches 4 --model_id 4 --architecture cnn --epochs 30 | tee -a logs/train-cnn.log

python experiment.py --model_id 1 --architecture cnn --epochs 30 | tee -a logs/train-cnn.log
python experiment.py --model_id 2 --architecture cnn --epochs 30 | tee -a logs/train-cnn.log
python experiment.py --model_id 3 --architecture cnn --epochs 30 | tee -a logs/train-cnn.log
python experiment.py --model_id 4 --architecture cnn --epochs 30 | tee -a logs/train-cnn.log
python experiment.py --model_id 5 --architecture cnn --epochs 30 | tee -a logs/train-cnn.log
python experiment.py --model_id 6 --architecture cnn --epochs 30 | tee -a logs/train-cnn.log
python experiment.py --model_id 7 --architecture cnn --epochs 30 | tee -a logs/train-cnn.log
python experiment.py --model_id 8 --architecture cnn --epochs 30 | tee -a logs/train-cnn.log

# python experiment-vb.py --num_branches 4 --model_id 1 --architecture cnn --epochs 30 | tee logs/train-cnn.log
# python experiment-vb.py --num_branches 4 --model_id 2 --architecture cnn --epochs 30 | tee -a logs/train-cnn.log
# python experiment-vb.py --num_branches 4 --model_id 3 --architecture cnn --epochs 30 | tee -a logs/train-cnn.log
# python experiment-vb.py --num_branches 4 --model_id 4 --architecture cnn --epochs 30 | tee -a logs/train-cnn.log

# python experiment-vb.py --num_branches 4 --model_id 1 --architecture cnn --epochs 30 --shared_frac 0.25 | tee -a logs/train-cnn.log
# python experiment-vb.py --num_branches 4 --model_id 2 --architecture cnn --epochs 30 --shared_frac 0.25 | tee -a logs/train-cnn.log
# python experiment-vb.py --num_branches 4 --model_id 3 --architecture cnn --epochs 30 --shared_frac 0.25 | tee -a logs/train-cnn.log
# python experiment-vb.py --num_branches 4 --model_id 4 --architecture cnn --epochs 30 --shared_frac 0.25 | tee -a logs/train-cnn.log

# python experiment-vb.py --num_branches 4 --model_id 1 --architecture cnn --epochs 30 --shared_frac 0.5 | tee -a logs/train-cnn.log
# python experiment-vb.py --num_branches 4 --model_id 2 --architecture cnn --epochs 30 --shared_frac 0.5 | tee -a logs/train-cnn.log
# python experiment-vb.py --num_branches 4 --model_id 3 --architecture cnn --epochs 30 --shared_frac 0.5 | tee -a logs/train-cnn.log
# python experiment-vb.py --num_branches 4 --model_id 4 --architecture cnn --epochs 30 --shared_frac 0.5 | tee -a logs/train-cnn.log

# python experiment-vb.py --num_branches 4 --model_id 1 --architecture cnn --epochs 30 --shared_frac 0.75 | tee -a logs/train-cnn.log
# python experiment-vb.py --num_branches 4 --model_id 2 --architecture cnn --epochs 30 --shared_frac 0.75 | tee -a logs/train-cnn.log
# python experiment-vb.py --num_branches 4 --model_id 3 --architecture cnn --epochs 30 --shared_frac 0.75 | tee -a logs/train-cnn.log
# python experiment-vb.py --num_branches 4 --model_id 4 --architecture cnn --epochs 30 --shared_frac 0.75 | tee -a logs/train-cnn.log
