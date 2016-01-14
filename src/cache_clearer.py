import os
import glob

def clear_cache(objects):
    [os.remove(f) for f in glob.glob(os.path.join('data',objects,'cache','.*'))]

if __name__ == '__main__':
    clear_cache('buildings')
    clear_cache('faces')