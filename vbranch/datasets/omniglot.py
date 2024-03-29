import os
import cv2
import numpy as np
import zipfile

class Omniglot(object):
    def __init__(self, set, A, P, K, flatten=True, preprocess=True):
        self.dir_path = _extract_omniglot_images(set)

        self.files = {}
        for alpha in os.listdir(self.dir_path):
            self.files[alpha] = {}
            characters = os.listdir(os.path.join(self.dir_path, alpha))
            for char in characters:
                im_dir = os.path.join(self.dir_path, alpha, char)
                self.files[alpha][char] = os.listdir(im_dir)

        self.A = A
        self.P = P
        self.K = K
        self.flatten = flatten
        self.preprocess = preprocess

    def __next__(self):
        return self.sample(self.A, self.P, self.K, self.flatten, self.preprocess)

    def sample(self, A, P, K, flatten=True, preprocess=True):
        """
        Args:
            - A: number of alphabets per batch
            - P: number of characters per alphabet
            - K: numer of samples per character
            - flatten: if true, return batch with shape [None, ...],
            else return triplet with shape [A, P, K, ...]
        Returns:
            numpy array"""
        # print('sample')
        def sample(arr, n, replace=False):
            return np.random.choice(arr, n, replace=replace)

        batch = []

        a_choice = sample(list(self.files.keys()), A, False)
        for alpha in a_choice:
            p_choice = sample(list(self.files[alpha].keys()), P, False)
            for p in p_choice:
                k_choice = sample(list(self.files[alpha][p]), K, False)
                for k in k_choice:
                    im_path = os.path.join(self.dir_path, alpha, p, k)
                    # Only use first channel since in grayscale
                    batch.append(cv2.imread(im_path)[..., 0])

        batch = np.stack(batch)[..., np.newaxis]

        if preprocess:
            batch = batch / 127.5 - 1

        if not flatten:
            batch = np.reshape(batch, (A, P, K) + batch.shape[1:])

        return batch

    def get_flattened_files(self):
        index_list = []
        files_list = []

        counter = 0
        for alpha in self.files.keys():
            for char in self.files[alpha].keys():
                for name in self.files[alpha][char]:
                    index_list.append([counter, alpha, char, name])
                    path = os.path.join(self.dir_path,alpha,char,name)
                    files_list.append(path)
                    counter += 1

        return index_list, files_list

def load_generator(set, A, P, K, sync=False, n_branches=1):
    if sync:
        sync_gen = SyncGenerator(A, P, K, n_branches)
        branch_gen = [Slicer(sync_gen, i) for i in range(n_branches)]
        return branch_gen

    return Omniglot(set, A, P, K)

def _extract_omniglot_images(set):
    # Clone Omniglot repo from GitHub if not in dir
    if not os.path.isdir('omniglot'):
        omniglot_url = 'git@github.com:brendenlake/omniglot.git'
        print('Cloning Omniglot repo: {}'.format(omniglot_url))
        cmd = 'git submodule add --force {}'.format(omniglot_url)
        os.system(cmd)

    if set == 'train':
        dir_path = 'omniglot/python/images_background'
        if not os.path.isdir('omniglot/python/images_background'):
            with zipfile.ZipFile(dir_path + '.zip', 'r') as zip_ref:
                zip_ref.extractall('omniglot/python')
            print('Images extracted from {}'.format(dir_path + '.zip'))
    elif set == 'test':
        dir_path = 'omniglot/python/images_evaluation'
        if not os.path.isdir('omniglot/python/images_evaluation'):
            with zipfile.ZipFile(dir_path + '.zip', 'r') as zip_ref:
                zip_ref.extractall('omniglot/python')
            print('Images extracted from {}'.format(dir_path + '.zip'))
    else:
        raise ValueError('invalid data set')

    return dir_path

# Enable non-replacement A sampling for multiple branches
class SyncGenerator(object):
    def __init__(self, A, P, K, n_branches):
        self.A = A
        self.P = P
        self.K = K
        self.n_branches = n_branches
        self.gen = Omniglot('train', A*n_branches, P, K)
        self.batch = None
        self.requests = 0

    def get(self, i):
        if self.batch is None:
            self.batch = next(self.gen)
            self.requests = self.n_branches

        start = i*self.A*self.P*self.K
        end = (i+1)*self.A*self.P*self.K
        branch_batch = self.batch[start: end]
        self.requests -= 1

        if self.requests == 0:
            self.batch = None

        return branch_batch

class Slicer(object):
    def __init__(self, parent, branch):
        self.parent = parent
        self.branch = branch

    def __next__(self):
        return self.parent.get(self.branch)
