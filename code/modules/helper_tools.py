import shutil
from pathlib import Path

def new_dir(dirpath, parents=False):
    '''
    若目录不存在, 那么新建该目录.

    Parameters
    ----------
    dirpath : Path
        目录对象.

    parents : bool
        若为True, 路径链上不存在的目录也会被顺便创建.
    '''
    if not dirpath.exists():
        dirpath.mkdir(parents=parents)

def renew_dir(dirpath, parents=False):
    '''
    重新创建目录.

    Parameters
    ----------
    dirpath : Path
        目录对象.

    parents : bool
        若为True, 路径链上不存在的目录也会被顺便创建.
    '''
    if dirpath.exists():
        shutil.rmtree(str(dirpath))
    dirpath.mkdir(parents=parents)
