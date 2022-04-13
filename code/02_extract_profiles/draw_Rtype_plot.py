'''
2022-04-12
画出两组个例四种雨型(层云, 对流, 浅云和其它)比例的柱状图.
'''
import json
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt

# 读取配置文件, 将config作为全局变量.
with open('config.json') as f:
    config = json.load(f)

if __name__ == '__main__':
    # 读取两组数据集.
    dirpath_result = Path(config['dirpath_result'])
    dirpath_merged = Path(config['dirpath_data'], 'DPR_case', 'merged')
    ds_dusty = xr.load_dataset(str(dirpath_merged / 'data_dusty.nc'))
    ds_clean = xr.load_dataset(str(dirpath_merged / 'data_clean.nc'))

    # 用npoints存储四种雨型廓线的数量.
    npoints = np.zeros((4, 2))
    for j, ds in enumerate([ds_dusty, ds_clean]):
        rainType = ds.rainType.values
        for i in range(4):
            npoints[i, j] = np.count_nonzero(rainType == i + 1)
    # 计算每种雨型占雨型总数的百分比.
    ratios = npoints / npoints.sum(axis=0) * 100

    # 画出柱状图.
    fig, ax = plt.subplots()
    x = np.arange(4)
    width = 0.35
    ax.bar(x - width / 2, ratios[:, 0], width, color='C1', label='Dusty')
    ax.bar(x + width / 2, ratios[:, 1], width, color='C0', label='Dusty')
    ax.legend(loc='upper right')
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(['stra.', 'conv.', 'shallow', 'other'])
    ax.set_xlabel('Rain Type', fontsize='medium')
    ax.set_ylabel('Percentage', fontsize='medium')
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(20))
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter())
    ax.set_title('Percentages of Rain Types', fontsize='medium')

    # 保存图片.
    filepath_output = dirpath_result / 'Rtypes.png'
    fig.savefig(str(filepath_output), dpi=300, bbox_inches='tight')
    plt.close(fig)
