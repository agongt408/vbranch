import argparse
import json
import os
import pprint as pp
import numpy as np
import pickle

KEY_PATH = '/home/albert/aws/ec2-ubuntu.pem'

# https://docs.python.org/2/library/argparse.html

parser = argparse.ArgumentParser(
            description='Train TriNet (https://arxiv.org/pdf/1703.07737.pdf)')
parser.add_argument('files', action='store', nargs='+',
                    help='paths to local files to sync')
parser.add_argument('--dns', dest='dns', nargs='?', action='store', 
                    default=None, const=None, help='public DNS')
parser.add_argument('--aws_src', dest='aws_src', action='store', nargs='?',
                    default=None, const='albert/src/', 
                    help='path to src directory on AWS: ~/albert/src/')
parser.add_argument('--aws_tf', dest='aws_tf', action='store', nargs='?',
                    default=None,
                    const='anaconda3/envs/tensorflow_p27/',
                    help='path to tensorflow directory on AWS: ' + \
                        '~/anaconda3/envs/tensorflow_p27/')
parser.add_argument('--aws_root', dest='aws_root', action='store', nargs='?',
                    default=None, const=None, help='path to append before files')

args = parser.parse_args()

if args.dns is None:
    if os.path.exists('./dns.p'):
        dns = pickle.load(open('dns.p', 'rb'))
        
        if not dns:
            raise ValueError, 'file dns.p is empty'
    else:
        raise ValueError, 'file dns.p does not exist'
else:
    dns = args.dns
    pickle.dump(dns, open( "dns.p", "wb" ) )

n_none = 0
for r in range(3):
    if [args.aws_src, args.aws_tf, args.aws_root][r] is not None:
        n_none += 1
        idx = r
  
if n_none == 0:
    root = ''
elif n_none == 1:
    root = [args.aws_src, args.aws_tf, args.aws_root][idx]
else:
    raise ValueError, \
        'at most one of --aws_src, --aws_tf, --aws_root can be not None'

print 'dns:' , dns
print 'root:' , root

for f in args.files:
    cmd = 'scp -i %s %s ubuntu@%s:%s' % \
        (KEY_PATH, f, dns, os.path.join(root, f))
    # print cmd
    os.system(cmd)
