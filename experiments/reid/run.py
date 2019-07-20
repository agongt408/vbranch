import os

# cmd = 'python experiments/reid/baseline.py --architecture densenet --epochs 10'
# os.system(cmd)

for num_branches in range(2, 3):
#     # model_id_list = ''
#     # for i in range(num_branches):
#     #     model_id_list += '%d ' % (i+1)
#     #
#     # cmd = 'python experiments/classification/baseline.py ' + \
#     #     '--architecture fcn --dataset mnist ' + \
#     #     '--test --trials 8 --model_id ' + model_id_list
#     # os.system(cmd)
#
    for shared_frac in [0]: # , 0.25, 0.5, 0.75, 1]:
        cmd = 'python experiments/reid/vbranch.py ' + \
            '--num_branches ' + str(num_branches) + ' ' + \
            '--shared_frac ' + str(shared_frac) + ' ' + \
            '--epochs 10 --architecture densenet'
        os.system(cmd)
