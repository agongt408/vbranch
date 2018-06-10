import argparse

def get_parser(description='Train TriNet (https://arxiv.org/pdf/1703.07737.pdf)', 
                default_file_root='trinet'):
                
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument('--test', action='store', default=None, nargs='+', 
                        help='model iters to test')
                        
    parser.add_argument('--P', dest='P_param', action='store',
                        default=5, const=5, nargs='?', type=int,
                        help='P parameter when constructing training batch')
                        
    parser.add_argument('--K', dest='K_param', action='store',
                        default=4, const=4, nargs='?', type=int,
                        help='K parameter when constructing training batch')
                        
    parser.add_argument('--dataset', action='store', default='market', 
                        const='market', nargs='?', choices=['market', 'cuhk03', 'duke'],
                        help="training dataset ('market', 'cuhk03', or 'duke')")
    
    parser.add_argument('--era', action='store', default=15, const=15, nargs='?', 
                        type=int, help='# of training era, default 10 epochs per era')
    
    parser.add_argument('--t1', action='store', default=5, const=5, nargs='?', 
                        type=int, help='# of era before learning rate decay begins')
    
    parser.add_argument('--no_weights', action='store_true',
                        help='if true, weights not loaded')
    
    parser.add_argument('--weights_file', action='store', type=file, nargs='?', 
                        help="path to model weights")
    
    parser.add_argument('--eval', dest='eval_era', action='store',
                        type=int, default=0, const=1, nargs='?',
                        help='evaluate (test) model after this many training era')
    
    parser.add_argument('--file_root', action='store', default=default_file_root,
                        const=default_file_root, nargs='?',
                        help='filename root of saved weights, ' + \
                            'P and K parameters automatically added to end of root')
    
    parser.add_argument('--diagnostic', action='store_true',
                        help='if true, batch normalization diagnostic turned on')
    
    parser.add_argument('--memory_fraction', action='store', default=1.0, 
                        const=1.0, type=float, nargs='?',
                        help='fraction of total gpu memory used')
    
    parser.add_argument('--continue', dest='continue_from_root', action='store',
                        default=None, const=None, nargs='?',
                        help='if true, continue training given model file root')
    
    parser.add_argument('--blocks', action='store', default=4, const=4, nargs='?',
                        type=int, help='number of blocks')

    return parser
    
    
def line_break(n=64):
    print "#" * n
