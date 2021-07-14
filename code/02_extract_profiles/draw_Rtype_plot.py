#----------------------------------------------------------------------------
# 2021/05/08
# 画出两组数据的四种雨型比例的柱状图.
#----------------------------------------------------------------------------
import json
from pathlib import Path

import numpy as np
import xarray as xr

import matplotlib as mpl
import matplotlib.pyplot as plt

# 读取配置文件,作为全局变量.
with open('config.json', 'r') as f:
    config = json.load(f)

def draw_Rtype_plot(dusty_ds, clean_ds, output_filepath):
    '''画出四种雨型比例的柱状图.'''
    ds_list = [dusty_ds, clean_ds]
    # 用data存储雨型的计数.
    # 第一维表示四种雨型,第二维表示污染与清洁分组.
    data = np.zeros((4, 2))
    for j, ds in enumerate(ds_list):
        rainType = ds.rainType.data
        for i in range(4):
            data[i, j] = np.count_nonzero(rainType == i + 1)

    # 将雨型计数转换为百分比.
    data = data / data.sum(axis=0) * 100

    # 画出柱状图.
    fig, ax = plt.subplots()
    x = np.arange(1, 5)
    width = 0.35
    ax.bar(
        x - width / 2, data[:, 0], width, color='C1', label=f'Dusty'
    )
    ax.bar(
        x + width / 2, data[:, 1], width, color='C0', label=f'Clean'
    )
    ax.legend(loc='upper left')
    ax.set_xlim(0.5, 4.5)
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(['stra.', 'conv.', 'warm', 'other'])
    ax.set_xlabel('Rain Type', fontsize='medium')
    ax.set_ylabel('Ratio', fontsize='medium')
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(20))
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter())
    ax.set_title('Ratio of Rain Types', fontsize='medium')

    # 保存图片.
    fig.savefig(str(output_filepath), dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    # 读取dusty cases和clean cases.
    result_dirpath = Path(config['result_dirpath'])
    with open(str(result_dirpath / 'found_cases.json'), 'r') as f:
        records = json.load(f)
    dusty_ds = xr.load_dataset(records['dusty']['profile_filepath'])
    clean_ds = xr.load_dataset(records['clean']['profile_filepath'])

    draw_Rtype_plot(dusty_ds, clean_ds, result_dirpath / 'Rtypes.png')
