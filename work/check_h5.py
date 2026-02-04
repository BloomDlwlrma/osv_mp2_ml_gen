import h5py
import numpy as np
import argparse 
from os import path


msg = "Command line script to check hdf5 data file"
parser = argparse.ArgumentParser(description=msg)

parser.add_argument("i", help="input file")
parser.add_argument("-L", "--level", nargs=1, action="store", help="Determine the tree level printed out, starting from 0")
parser.add_argument("-k", "--key", action="append", help="Reading data attached to the provided key")
parser.add_argument("-n", "--norm", action="store_true", help="Return norm of specified data")
parser.add_argument("-p", "--print", action="store_true", help="print out specified data")

args = parser.parse_args()

def h5_tree(val, pre='', depth=0, level=None):
# https://stackoverflow.com/questions/61133916/is-there-in-python-a-single-function-that-shows-the-full-structure-of-a-hdf5-fi
    """
    Modified recursive h5py tree, recursion returns when 'depth' is greater than 'level' specified.
    """
    if type(level) is int:
        if depth > level:
            return 
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            # the last item
            if type(val) == h5py._hl.group.Group:
                print(pre + '└── ' + key)
                h5_tree(val, pre+'    ', depth+1, level)
            else:
                try:
                    print(pre + '└── ' + key + ' (%d)' % len(val))
                except TypeError:
                    print(pre + '└── ' + key + ' (scalar)')
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + '├── ' + key)
                h5_tree(val, pre+'│   ', depth+1, level)
            else:
                try:
                    print(pre + '├── ' + key + ' (%d)' % len(val))
                except TypeError:
                    print(pre + '├── ' + key + ' (scalar)')


def print_key(key, f, _norm, _print):
    print(key)
    print('└── ' + 'shape ' + str(f[key].shape))
    print('└── ' + 'dtype ' + '%s'%f[key].dtype)
    if _norm:
        print(f"└── norm {np.linalg.norm(f[key][:]):.6f}") 
    if _print:
        print(f"└── data \n{f[key][:]}") 



fname = args.i
if not path.isfile(fname):
    print("Provided file does not exist")
    exit()
else:
    f = h5py.File(fname, 'r')

if args.level is not None:
    level = int(args.level[0])
else:
    level = args.level

if args.key is not None:
    for key in args.key:
        if isinstance(f[key], h5py._hl.group.Group):
            h5_tree(f[key], level=level)
        else:
            print_key(key, f, args.norm, args.print)
else: 
    h5_tree(f, level=level)

f.close()

