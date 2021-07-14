#----------------------------------------------------------------------------
# 2021/05/08
# 画出之前找出的每一个污染个例与清洁个例的CFAD图.
#
# 生成四张图,每张图上画出污染(清洁)组所有个例的层云(对流)降水CFAD图.
#----------------------------------------------------------------------------
import json
from pathlib import Path
import sys
sys.path.append('../modules')
from profile_funcs import calc_cfad
from helper_funcs import recreate_dir, decompose_int

import numpy as np
import xarray as xr

import matplotlib as mpl
import matplotlib.pyplot as plt
import cmaps

# 读取配置文件,作为全局变量使用.
with open('config.json', 'r') as f:
    config = json.load(f)

def draw_one_plot(cases, Rtype, fig_title, output_filepath):
    '''画出一组个例的某个雨型的CFAD组图.'''
    # 设置用来画CFAD的bins.
    dx = 1
    dy = 0.25
    xmin, xmax = 10, 60
    ymin, ymax = 1.5, 16
    xbins = np.linspace(xmin, xmax, int((xmax - xmin) / dx) + 1)
    ybins = np.linspace(ymin, ymax, int((ymax - ymin) / dy) + 1)
    # 读取用来做图的数据.
    ncase = len(cases)
    cfads = np.zeros((ncase, len(ybins) - 1, len(xbins) - 1))
    npoints = np.zeros(ncase, dtype=int)
    times = []
    for i, case in enumerate(cases):
        times.append(case['rain_time'])
        with xr.open_dataset(case['profile_filepath']) as ds:
            hgt = ds.height.data
            # DataArray.data可以含有NaN.
            rainType = ds.rainType.data
            z = ds.zFactorCorrected.data

        # 根据雨型截取数据.
        if Rtype == 'stra':
            flag = rainType == 1
        elif Rtype == 'conv':
            flag = rainType == 2
        # 若廓线数目不为0,那么计算cfad.
        npoints[i] = np.count_nonzero(flag)
        if npoints[i] > 0:
            z = z[flag, :]
            cfads[i, :, :] = calc_cfad(z, hgt, xbins, ybins, norm='sum')
        else:
            continue

    # 自动划分组图的形状.
    m, n = decompose_int(ncase)
    figsize = (n * 3, m * 4)
    # 画出m*n大小的组图.
    fig, axes = plt.subplots(nrows=m, ncols=n, figsize=figsize)
    cmap = cmaps.WhBlGrYeRe
    for i in range(ncase):
        ax = axes.flatten()[i]
        cfad = cfads[i]
        time = times[i]
        npoint = npoints[i]

        # 画出CFAD图.
        im = ax.pcolormesh(xbins, ybins, cfad, cmap=cmap, vmin=0)
        cbar = fig.colorbar(im, ax=ax, aspect=30, extend='both')
        cbar.ax.tick_params(labelsize='xx-small')
        cbar.ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(6))
        cbar.ax.yaxis.set_major_formatter(
            mpl.ticker.PercentFormatter(decimals=2)
        )

        # 用额外的信息作为子图标题.
        ax.set_title(time, fontsize='x-small', loc='left')
        ax.set_title(f'N={npoint}', fontsize='x-small', loc='right')

    # 设置第一列和最后一排的label.
    for ax in axes[:, 0]:
        ax.set_ylabel('Height (km)', fontsize='x-small')
    for ax in axes[-1, :]:
        ax.set_xlabel('Reflectivity (dBZ)', fontsize='x-small')

    # 这些子图的xy ticks设置一致,所以放在一起设置.
    for ax in axes.flat:
        ax.set_xlim(10, 60)
        ax.set_ylim(0, 12)
        ax.tick_params(labelsize='x-small')
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))

    # 将多余的子图隐藏.
    if m * n > ncase:
        for ax in axes.flatten()[ncase:]:
            ax.set_visible(False)

    # 设置整张图的标题.
    fig.suptitle(fig_title, y=0.92, fontsize='large')

    # 保存图片.
    fig.savefig(str(output_filepath), dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    # 读取两组个例.
    result_dirpath = Path(config['result_dirpath'])
    with open(str(result_dirpath / 'found_cases.json'), 'r') as f:
        records = json.load(f)
    dusty_cases = records['dusty']['cases']
    clean_cases = records['clean']['cases']

    # 重新创建图片目录.
    output_dirpath = result_dirpath / 'CFADs'
    recreate_dir(output_dirpath)
    dusty_dirpath = output_dirpath / 'dusty_cases'
    clean_dirpath = output_dirpath / 'clean_cases'
    dusty_dirpath.mkdir()
    clean_dirpath.mkdir()

    draw_one_plot(
        dusty_cases, 'stra', 'Stratiform Rains of Dusty Cases',
        dusty_dirpath / 'stra.png'
    )
    draw_one_plot(
        dusty_cases, 'conv', 'Convective Rains of Dusty Cases',
        dusty_dirpath / 'conv.png'
    )
    draw_one_plot(
        clean_cases, 'stra', 'Stratiform Rains of Clean Cases',
        clean_dirpath / 'stra.png'
    )
    draw_one_plot(
        clean_cases, 'conv', 'Convective Rains of Clean Cases',
        clean_dirpath / 'conv.png'
    )
