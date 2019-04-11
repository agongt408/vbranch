# !/bin/sh

# Run experiment with various configs
# https://stackoverflow.com/questions/24207916/catching-the-exception-thrown-by-python-script-in-shell-script

python experiment.py --architecture cnn --epochs 30 --trials 8 | tee -a logs/train.log

python experiment-vb.py --architecture cnn --epochs 30 --trials 4 --shared_frac 0 | tee -a logs/train.log
python experiment-vb.py --architecture cnn --epochs 30 --trials 4 --shared_frac 0.25 | tee -a logs/train.log
python experiment-vb.py --architecture cnn --epochs 30 --trials 4 --shared_frac 0.5 | tee -a logs/train.log
python experiment-vb.py --architecture cnn --epochs 30 --trials 4 --shared_frac 0.75 | tee -a logs/train.log
