'''
2022-04-12
画出两组数据集两种雨型在两种坐标下的CFAD图.
组图形状为(2, 3), 行表示两种雨型, 前两列表示污染组和清洁组, 第三列是污染组
减去清洁组. 根据两种坐标画出两张组图.
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
import plot_tools

# 读取配置文件, 将config作为全局变量.
with open('config.json') as f:
    config = json.load(f)

def draw_one_plot(ds_dusty, ds_clean, coordname, filepath_output):
    '''画出两组数据两种雨型的CFAD图. coordname用于指定垂直坐标.'''
    # 设置画CFAD图的bins.
    xbins = np.linspace(10, 50, 41)
    if coordname == 'height':
        ybins = np.linspace(1.5, 12, 43)
    elif coordname == 'temp':
        ybins = np.linspace(-60, 10, 71)
    x = (xbins[1:] + xbins[:-1]) / 2
    y = (ybins[1:] + ybins[:-1]) / 2

    # 根据垂直坐标选取所需的变量.
    varname = 'zFactorCorrected'
    if coordname == 'temp':
        varname += '_t'

    # 用cfads存储计算的CFAD.
    # 第一维是雨型, 第二维是污染分组和差值.
    coord = ds_dusty[coordname].values
    cfads = np.zeros((2, 3, len(y), len(x)))
    npoints = np.zeros((2, 2), dtype=int)
    for j, ds in enumerate([ds_dusty, ds_clean]):
        for i in range(2):
            var = ds[varname].isel(npoint=(ds.rainType == i + 1)).values
            npoints[i, j] = var.shape[0]
            cfads[i, j, :, :] = profile_tools.cfad(
                var, coord, xbins, ybins,
                sigma=0.5, norm='sum'
            ) * 100
    # 计算两组的差值.
    cfads[:, 2, :, :] = cfads[:, 0, :, :] - cfads[:, 1, :, :]

    # 根据垂直坐标决定纵坐标的设置.
    if coordname == 'height':
        ylabel = 'Height (km)'
        ylims = (0, 12)
        ybase = 2
    elif coordname == 'temp':
        ylabel = 'Temperature (℃)'
        ylims = (20, -60)
        ybase = 20
    # 画图用的参数.
    Rtypes = ['Stratiform', 'Convective']
    groups = ['Dusty', 'Clean', 'Dusty - Clean']

    # 组图形状为(2, 3).
    # 行表示两种雨型, 前两列表示两个分组, 第三列是两组之差.
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for (i, j), ax in np.ndenumerate(axes):
        # 前两列画普通图.
        cfad = cfads[i, j, :, :]
        if j <= 1:
            im = ax.contourf(
                x, y, cfad, levels=40,
                cmap=cmaps.WhBlGrYeRe, extend='both'
            )
            rstr = f'{groups[j]} (N={npoints[i, j]})'
        # 第三列画差值.
        else:
            vmax = 0.95 * np.abs(cfad).max()
            levels = np.linspace(-vmax, vmax, 40)
            im = ax.contourf(
                x, y, cfad, levels=levels,
                cmap=cmaps.NCV_blu_red, extend='both'
            )
            rstr = groups[j]
        cbar = fig.colorbar(
            im, ax=ax, aspect=30,
            ticks=mticker.LinearLocator(5),
            format=mticker.PercentFormatter(decimals=2)
        )
        cbar.ax.tick_params(labelsize='x-small')
        # 设置左右小标题.
        ax.set_title(Rtypes[i], loc='left', fontsize='small')
        ax.set_title(rstr, loc='right', fontsize='small')

    # 设置坐标轴.
    for ax in axes.flat:
        ax.set_xlim(10, 50)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
        ax.set_ylim(*ylims)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(ybase))
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.tick_params(labelsize='x-small')

    # 在组图边缘设置标签.
    for ax in axes[1, :]:
        ax.set_xlabel('Reflectivity (dBZ)', fontsize='small')
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, fontsize='small')

    # 为子图标出字母标识.
    plot_tools.letter_axes(axes, 0.06, 0.96, fontsize='small')

    # 保存图片.
    fig.savefig(str(filepath_output), dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    # 读取两组数据集.
    dirpath_input = Path(config['dirpath_input'])
    ds_dusty = xr.load_dataset(str(dirpath_input / 'data_dusty.nc'))
    ds_clean = xr.load_dataset(str(dirpath_input / 'data_clean.nc'))

    # 创建输出目录.
    dirpath_output = Path(config['dirpath_result'], 'CFADs')
    helper_tools.renew_dir(dirpath_output)

    # 画出高度坐标与温度坐标下的CFAD图.
    draw_one_plot(
        ds_dusty, ds_clean,
        coordname='height',
        filepath_output=(dirpath_output / 'CFADs.png')
    )
    draw_one_plot(
        ds_dusty, ds_clean,
        coordname='temp',
        filepath_output=(dirpath_output / 'CFADs_t.png')
    )
