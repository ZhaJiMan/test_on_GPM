#----------------------------------------------------------------------------
# 2021/07/08
# 一些辅助用的函数.
#----------------------------------------------------------------------------
import shutil

import numpy as np

def recreate_dir(dirpath):
    '''重新创建目录.要求dirpath是Path对象.'''
    if dirpath.exists():
        shutil.rmtree(str(dirpath))
    dirpath.mkdir()

def letter_subplots(axes, position, fontsize):
    '''
    给形状相同的组图里的每个子图标注字母.

    position为Axes坐标下字母的位置.
    (0, 0)表示左下角,(1, 1)表示右上角.
    '''
    for i, ax in enumerate(axes.flat):
        letter = chr(ord('`') + i + 1)
        ax.text(
            *position, f'({letter})', fontsize=fontsize,
            ha='center', va='center', transform=ax.transAxes
        )
