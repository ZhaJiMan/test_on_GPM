import shutil

import numpy as np

def recreate_dir(dirpath):
    '''重新创建目录.要求dirpath是Path对象.'''
    if dirpath.exists():
        shutil.rmtree(str(dirpath))
    dirpath.mkdir()

def decompose_int(x):
    '''将整数x分解为m*n的形式,m*n大于等于x,同时形状接近于正方形.'''
    mid = np.sqrt(x)
    m = int(np.floor(mid))
    n = int(np.ceil(mid))
    if m * n < x:
        m += 1

    return m, n

