import tensorflow as tf

class PrintLine(object):
    """
    Helper class used to print each line of model summaries"""

    def __init__(self, *widths):
        self.widths = widths

    def __call__(self, *items, show_line=True):
        str_f = ''
        for i in range(len(items)):
            str_f += ('{:<' + str(self.widths[i]) + '}').format(str(items[i]))
        print(str_f)

        if show_line:
            print('-' * len(str_f))
