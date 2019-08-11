import os

# Omniglot

for num_branches in range(3, 5):
    cmd = f'python experiments/one_shot/baseline.py ' \
        f'--architecture simple --dataset omniglot --epochs 60 --trials 1 --A {num_branches} ' \
        f'--path omniglot_zero/omniglot-simple/B{num_branches} --model_id 5 6 7 8'
    os.system(cmd)

    # model_id_list = ''
    # for i in range(num_branches):
    #     model_id_list += '%d ' % (i+1)
    #
    # cmd = 'python experiments/classification/baseline.py ' + \
    #     '--architecture fcn --dataset mnist ' + \
    #     '--test --trials 8 --model_id ' + model_id_list
    # os.system(cmd)

    for shared_frac in [0., 0.25, 0.5, 0.75, 1.]:
        cmd = f'python experiments/one_shot/vbranch.py ' \
            f'--architecture simple --dataset omniglot --epochs 60 ' \
            f'--num_branches {num_branches} --shared_frac {shared_frac} ' \
            f'--trials 1 --A 1 --path omniglot_zero/vb-omniglot-simple --model_id 5 6 7 8'
        os.system(cmd)
