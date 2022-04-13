'''
2022-04-12
画出每一个污染个例和清洁个例的CFAD图.
生成四张组图, 对应于两种雨型和两种个例的组合, 每张组图上画出所有个例的CFAD.
'''
import json
from pathlib import Path
import sys
sys.path.append('../modules')

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cmaps

import helper_tools
import profile_tools

# 读取配置文件, 将config作为全局变量.
with open('config.json') as f:
    config = json.load(f)

def decompose_int(x):
    '''将整数x分解为m*n的形式, m*n大于等于x, 同时形状接近于正方形.'''
    mid = np.sqrt(x)
    m = int(np.floor(mid))
    n = int(np.ceil(mid))
    if m * n < x:
        m += 1

    return m, n

def draw_one_plot(cases, Rtype, title, filepath_output):
    '''给定雨型, 在一张组图上画出所有个例的CFAD图.'''
    # 设置画CFAD的bins.
    dx = 1
    dy = 0.25
    xmin, xmax = 10, 60
    ymin, ymax = 1.5, 12
    nx = int((xmax - xmin) / dx)
    ny = int((ymax - ymin) / dy)
    xbins = np.linspace(xmin, xmax, nx + 1)
    ybins = np.linspace(ymin, ymax, ny + 1)

    # 读取并计算用来做图的数据.
    ncase = len(cases)
    cfads = np.zeros((ncase, ny, nx))
    npoints = np.zeros(ncase, dtype=int)
    times = []
    for i, case in enumerate(cases):
        times.append(case['rain_time'])
        with xr.open_dataset(case['filepath_profile']) as ds:
            height = ds.height.values
            rainType = ds.rainType.values
            zFactor = ds.zFactorCorrected.values

        # 根据雨型截取数据.
        if Rtype == 'stra':
            mask = rainType == 1
        elif Rtype == 'conv':
            mask = rainType == 2

        # 若廓线数目不为0,那么计算cfad.
        npoints[i] = np.count_nonzero(mask)
        if npoints[i] > 0:
            zFactor = zFactor[mask, :]
            cfads[i, :, :] = profile_tools.cfad(
                zFactor, height, xbins, ybins, norm='sum'
            ) * 100

    # 组图形状为(m, n), 自动从ncase中分解得到.
    m, n = decompose_int(ncase)
    figsize = (n * 3, m * 4)
    fig, axes = plt.subplots(m, n, figsize=figsize)

    # 画出CFAD.
    for i, ax in enumerate(axes.flat[:ncase]):
        vmax = None if npoints[i] > 0 else 1
        im = ax.pcolormesh(
            xbins, ybins, cfads[i, :, :],
            cmap=cmaps.WhBlGrYeRe, vmin=0, vmax=vmax,
            shading='flat'
        )
        cbar = fig.colorbar(
            im, ax=ax, aspect=30, extend='both',
            ticks=mticker.LinearLocator(5),
            format=mticker.PercentFormatter(decimals=2)
        )
        cbar.ax.tick_params(labelsize='xx-small')

        # 用额外的信息作为子图标题.
        ax.set_title(times[i], fontsize='x-small', loc='left')
        ax.set_title(f'N={npoints[i]}', fontsize='x-small', loc='right')

    # 在组图边缘设置标签.
    for ax in axes[:, 0]:
        ax.set_ylabel('Height (km)', fontsize='x-small')
    for ax in axes[-1, :]:
        ax.set_xlabel('Reflectivity (dBZ)', fontsize='x-small')

    # 同时设置所有子图的刻度.
    for ax in axes.flat:
        ax.set_xlim(10, 60)
        ax.set_ylim(0, 12)
        ax.tick_params(labelsize='x-small')
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(2))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(1))

    # 隐藏多余的子图.
    if m * n > ncase:
        for ax in axes.flat[ncase:]:
            ax.set_visible(False)

    # 设置整张图的标题.
    fig.suptitle(title, y=0.9, fontsize='large')

    # 保存图片.
    fig.savefig(str(filepath_output), dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    # 读取两组个例.
    dirpath_input = Path(config['dirpath_input'])
    with open(str(dirpath_input / 'cases_dusty.json')) as f:
        cases_dusty = json.load(f)
    with open(str(dirpath_input / 'cases_clean.json')) as f:
        cases_clean = json.load(f)

    # 创建输出目录.
    dirpath_output = Path(config['dirpath_result'], 'CFADs')
    helper_tools.renew_dir(dirpath_output)

    # 画出两组个例两种雨型的所有CFAD图.
    draw_one_plot(
        cases_dusty, Rtype='stra',
        title='Stratiform Rains of Dusty Cases',
        filepath_output=(dirpath_output / 'stra_dusty.png')
    )
    draw_one_plot(
        cases_dusty, Rtype='conv',
        title='Convective Rains of Dusty Cases',
        filepath_output=(dirpath_output / 'conv_dusty.png')
    )
    draw_one_plot(
        cases_clean, Rtype='stra',
        title='Stratiform Rains of Clean Cases',
        filepath_output=(dirpath_output / 'stra_clean.png')
    )
    draw_one_plot(
        cases_clean, Rtype='conv',
        title='Convective Rains of Clean Cases',
        filepath_output=(dirpath_output / 'conv_clean.png')
    )
