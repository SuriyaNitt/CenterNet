import os.path as osp
import sys
import os

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = os.path.join(os.environ["CenterNet_ROOT"], "src")

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, 'lib')
add_path(lib_path)
