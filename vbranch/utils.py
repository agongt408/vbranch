import tensorflow as tf
import numpy as np

class Summary(object):
    """Helper class used to print model summaries"""

    def __init__(self, *labels):
        self.labels = labels
        self.rows = [] # Stores contents of each row
        self.show_line = [] # Stores whether to show line for each row

    def add(self, *items, show_line=True):
        assert len(self.labels) == len(items)
        self.rows.append(items)
        self.show_line.append(show_line)

    def show(self):
        # Include labels
        print_rows = [self.labels,] + self.rows
        print_show_line = [True,] + self.show_line

        widths = []
        for i in range(len(self.labels)):
            w = np.max([len(str(row[i])) for row in print_rows]) + 2
            widths.append(w)

        total_width = np.sum(widths)

        for r, row in enumerate(print_rows):
            str_f = ''
            for i in range(len(row)):
                str_f += ('{:<'+str(widths[i])+'}').format(str(row[i]))

            print(str_f)

            if print_show_line[r]:
                print('-' * total_width)

def get_shape_as_str(tensor):
    shape = tensor.get_shape().as_list()
    return shape_to_str(shape)

def shape_to_str(shape):
    return str(shape).replace(' ', '')

def get_num_params(tensor):
    return np.prod(tensor.get_shape().as_list())
