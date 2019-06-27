import os

# Omniglot

cmd = 'python experiments/one_shot/baseline.py ' + \
    '--architecture simple --dataset omniglot --epochs 60'
os.system(cmd)

for num_branches in range(2, 5):
    # model_id_list = ''
    # for i in range(num_branches):
    #     model_id_list += '%d ' % (i+1)
    #
    # cmd = 'python experiments/classification/baseline.py ' + \
    #     '--architecture fcn --dataset mnist ' + \
    #     '--test --trials 8 --model_id ' + model_id_list
    # os.system(cmd)

    for shared_frac in [0, 0.25, 0.5, 0.75, 1]:
        cmd = 'python experiments/one_shot/vbranch.py ' + \
            '--architecture simple --dataset omniglot --epochs 60 ' + \
            '--num_branches ' + str(num_branches) + ' ' + \
            '--shared_frac ' + str(shared_frac)
        os.system(cmd)
