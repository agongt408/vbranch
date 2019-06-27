import os

# # Toy dataset
#
# for num_classes in [100, 200, 400]:
#     for samples_per_class in [100, 200, 400]:
#         cmd = 'python experiments/classification/baseline.py ' + \
#             '--architecture fcn ' + \
#             '--dataset toy --epochs 15 --trials 8 ' + \
#             '--num_classes ' + str(num_classes) + ' ' + \
#             '--samples_per_class ' + str(samples_per_class)
#         os.system(cmd)
#
#         for num_branches in range(2, 7):
#             model_id_list = ''
#             for i in range(num_branches):
#                 model_id_list += '%d ' % (i+1)
#
#             cmd = 'python experiments/classification/baseline.py ' + \
#                 '--architecture fcn ' + \
#                 '--dataset toy --trials 4 ' + \
#                 '--num_classes ' + str(num_classes) + ' ' + \
#                 '--samples_per_class ' + str(samples_per_class) + ' ' + \
#                 '--test --trials 8 --model_id ' + model_id_list
#             os.system(cmd)
#
#             for shared_frac in [0, 0.25, 0.5, 0.75, 1]:
#                 cmd = 'python experiments/classification/vbranch.py ' + \
#                     '--architecture fcn ' + \
#                     '--dataset toy --epochs 15 --trials 4 ' + \
#                     '--num_classes ' + str(num_classes) + ' ' + \
#                     '--samples_per_class ' + str(samples_per_class) + ' ' + \
#                     '--num_branches ' + str(num_branches) + ' ' + \
#                     '--shared_frac ' + str(shared_frac)
#                 os.system(cmd)

# # MNIST
#
# cmd = 'python experiments/classification/baseline.py ' + \
#     '--architecture fcn --dataset mnist --epochs 15 --trials 8'
# os.system(cmd)
#
for num_branches in range(2, 7):
    # model_id_list = ''
    # for i in range(num_branches):
    #     model_id_list += '%d ' % (i+1)
    #
    # cmd = 'python experiments/classification/baseline.py ' + \
    #     '--architecture fcn --dataset mnist ' + \
    #     '--test --trials 8 --model_id ' + model_id_list
    # os.system(cmd)

    for shared_frac in [0, 0.25, 0.5, 0.75, 1]:
        cmd = 'python experiments/classification/vbranch.py ' + \
            '--architecture fcn2 ' + \
            '--dataset mnist --epochs 15 --trials 8 ' + \
            '--num_branches ' + str(num_branches) + ' ' + \
            '--shared_frac ' + str(shared_frac)
        os.system(cmd)
