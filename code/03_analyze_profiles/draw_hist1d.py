'''
2022-04-12
两组数据集分雨型画出多种降水参量的一维PDF.
组图形状为(2, nvar), 行表示雨型, 列表示变量, 一张子图里画出污染组和清洁组的
PDF分布. PDF分布通过一维histogram计算平滑得到.
'''
import json
from pathlib import Path
import sys
sys.path.append('../modules')

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import helper_tools
import profile_tools
import plot_tools

# 读取配置文件, 将config作为全局变量.
with open('config.json') as f:
    config = json.load(f)

if __name__ == '__main__':
    # 读取两组数据集.
    dirpath_input = Path(config['dirpath_input'])
    ds_dusty = xr.load_dataset(str(dirpath_input / 'data_dusty.nc'))
    ds_clean = xr.load_dataset(str(dirpath_input / 'data_clean.nc'))

    # 创建输出目录.
    dirpath_output = Path(config['dirpath_result'], 'statistics')
    helper_tools.new_dir(dirpath_output)

    varnames = [
        'precipRateNearSurface',
        'heightStormTop',
        'tempStormTop',
        'pct89'
    ]
    # 为层云降水和对流降水准备不同分辨率的bins.
    bins_arr = np.array([
        [np.logspace(-2, 2, 31), np.logspace(-2, 2, 21)],
        [np.linspace(0, 12, 31), np.linspace(0, 12, 21)],
        [np.linspace(-60, 20, 31), np.linspace(-60, 20, 21)],
        [np.linspace(150, 300, 31), np.linspace(150, 300, 21)]
    ], dtype=object).T
    # 用hs存储计算出的histogram.
    # 第一维表示两种雨型, 第二维表示四种变量, 第三维表示污染分组.
    hs = np.zeros((2, 4, 2), dtype=object)
    avgs = np.zeros((2, 4, 2))
    for k, ds in enumerate([ds_dusty, ds_clean]):
        for i in range(2):
            ds_temp = ds.isel(npoint=(ds.rainType == i + 1))
            for j in range(4):
                var = ds_temp[varnames[j]].values
                avgs[i, j, k] = np.nanmean(var)
                hs[i, j, k] = profile_tools.hist1d(
                    var, bins_arr[i, j], sigma=1, norm='sum'
                ) * 100

    # 画图用的参数.
    Rtypes = ['Stratiform', 'Convective']
    groups = ['Dusty', 'Clean']
    colors = ['C1', 'C0']
    xlabels = [
        'Surface Rain Rate (mm/h)',
        'Height of Storm Top (km)',
        'Temperature of Storm Top (℃)',
        'PCT89 (K)'
    ]

    # 组图形状为(2, 4).
    # 行表示两种雨型, 列表示4种变量, 每张子图中画有两组histogram.
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    for (i, j), ax in np.ndenumerate(axes):
        bins = bins_arr[i, j]
        x = (bins[1:] + bins[:-1]) / 2
        for k in range(2):
            h = hs[i, j, k]
            avg = avgs[i, j, k]
            ax.plot(
                x, h, color=colors[k], lw=1.5,
                label=f'{groups[k]} (mean={avg:.2f})'
            )
            ax.fill_between(x, h, color=colors[k], alpha=0.4)
        ax.legend(fontsize='xx-small', loc='upper right')
        # 标出雨型.
        ax.set_title(Rtypes[i], fontsize='small')

    # 设置第一列的横坐标.
    for ax in axes[:, 0]:
        ax.set_xscale('log')
        ax.set_xlim(1E-2, 1E2)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    # 设置第二列的横坐标.
    for ax in axes[:, 1]:
        ax.set_xlim(0, 12)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    # 设置第三列的横坐标.
    for ax in axes[:, 2]:
        ax.set_xlim(20, -60)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
    # 设置第四列的横坐标.
    for ax in axes[:, 3]:
        ax.set_xlim(150, 300)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(30))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(15))

    # 同时设置纵坐标和labelsize.
    for ax in axes.flat:
        ax.set_ylim(0, 1.2 * ax.get_ylim()[1])
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.tick_params(labelsize='x-small')

    # 在组图边缘设置标签.
    for xlabel, ax in zip(xlabels, axes[1, :]):
        ax.set_xlabel(xlabel, fontsize='small')
    for ax in axes[:, 0]:
        ax.set_ylabel('PDF (%)', fontsize='small')

    # 为子图标出字母标识.
    plot_tools.letter_axes(axes, 0.06, 0.95, fontsize='small')

    # 保存图片.
    filepath_output = dirpath_output / 'hist1d.png'
    fig.savefig(str(filepath_output), dpi=300, bbox_inches='tight')
    plt.close(fig)
