import os

# dirpath = os.getcwd()

for num_classes in [10, 20, 40, 100, 200, 400]:
    for samples_per_class in [100, 200, 400, 1000]:
        cmd = 'python experiments/classification/baseline.py ' + \
            '--dataset toy --epochs 15 --trials 8 ' + \
            '--num_classes ' + str(num_classes) + ' ' + \
            '--samples_per_class ' + str(samples_per_class)
        os.system(cmd)

        for num_branches in range(2, 7):
            for shared_frac in [0, 0.25, 0.5, 0.75, 1]:
                cmd = 'python experiments/classification/baseline.py ' + \
                    '--dataset toy --epochs 15 --trials 4 ' + \
                    '--num_classes ' + str(num_classes) + ' ' + \
                    '--samples_per_class ' + str(samples_per_class) + ' ' + \
                    '--num_branches ' + str(num_branches) + ' ' + \
                    '--shared_frac ' + str(shared_frac)
                os.system(cmd)
