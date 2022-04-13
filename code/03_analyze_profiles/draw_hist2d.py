'''
2022-04-12
两组数据集分雨型画出地表降水率vs雨顶温度, 地表降水率vsPCT89的二维PDF分布.
组图形状为(2, 3), 行表示雨型, 前两列表示污染组和清洁组, 第三列是污染组的分布
减去清洁组的分布. 根据所选变量画出两张组图.
PDF分布通过二维histogram计算平滑得到.
'''
import json
from pathlib import Path
import sys
sys.path.append('../modules')

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cmaps

import helper_tools
import profile_tools
import plot_tools

# 读取配置文件, 将config作为全局变量.
with open('config.json') as f:
    config = json.load(f)

def draw_surfRr_vs_st(ds_dusty, ds_clean, filepath_output):
    '''画出地表降水率与雨顶温度之间的二维histogram图.'''
    # 为层云降水和对流降水准备不同分辨率的bins.
    xbins_list = [
        np.logspace(-1, 2, 41),
        np.logspace(-1, 2, 21)
    ]
    ybins_list = [
        np.linspace(-60, 20, 41),
        np.linspace(-60, 20, 21)
    ]

    # 用Hs存储计算的二维histogram.
    # 第一维是雨型, 第二维是污染分组和差值.
    Hs = np.empty((2, 3), dtype=object)
    for i in range(2):
        for j, ds in enumerate([ds_dusty, ds_clean]):
            ds = ds.isel(npoint=(ds.rainType == i + 1))
            xvar = ds.precipRateNearSurface.values
            yvar = ds.tempStormTop.values
            Hs[i, j] = profile_tools.hist2d(
                xvar, yvar,
                xbins=xbins_list[i],
                ybins=ybins_list[i],
                sigma=1, norm='sum'
            ) * 100
    # 计算两组的差值.
    Hs[:, 2] = Hs[:, 0] - Hs[:, 1]

    # 画图用的参数.
    Rtypes = ['Stratiform', 'Convective']
    groups = ['Dusty', 'clean', 'Dusty - Clean']
    xlist = [(bins[1:] + bins[:-1]) / 2 for bins in xbins_list]
    ylist = [(bins[1:] + bins[:-1]) / 2 for bins in ybins_list]

    # 组图形状为(2, 3).
    # 行表示两种雨型, 前两列表示污染分组, 第三列是两组之差.
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    for (i, j), ax in np.ndenumerate(axes):
        # 前两列画普通图.
        H = Hs[i, j]
        x, y = xlist[i], ylist[i]
        if j <= 1:
            im = ax.contourf(
                x, y, H, levels=40,
                cmap=cmaps.WhBlGrYeRe, extend='both'
            )
        # 第三列画差值.
        else:
            vmax = 0.95 * np.abs(H).max()
            levels = np.linspace(-vmax, vmax, 40)
            im = ax.contourf(
                x, y, H, levels=levels,
                cmap=cmaps.NCV_blu_red, extend='both'
            )
        cbar = fig.colorbar(
            im, ax=ax, aspect=20,
            ticks=mticker.LinearLocator(5),
            format=mticker.PercentFormatter(decimals=2)
        )
        cbar.ax.tick_params(labelsize='x-small')
        # 设置左右小标题.
        ax.set_title(Rtypes[i], loc='left', fontsize='small')
        ax.set_title(groups[j], loc='right', fontsize='small')

    # 为每个子图设置相同的刻度.
    for ax in axes.flat:
        ax.set_xscale('log')
        ax.set_xlim(0.1, 10)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_ylim(20, -60)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(10))
        ax.tick_params(labelsize='x-small')

    # 在组图边缘设置标签.
    for ax in axes[-1, :]:
        ax.set_xlabel('Surface Rain Rate (mm/h)', fontsize='small')
    for ax in axes[:, 0]:
        ax.set_ylabel('Temperature of Storm Top (℃)', fontsize='small')

    # 为子图标出字母标识.
    plot_tools.letter_axes(axes, 0.06, 0.96, fontsize='small')

    # 保存图片.
    fig.savefig(str(filepath_output), dpi=300, bbox_inches='tight')
    plt.close(fig)

def draw_surfRr_vs_pct(dusty_ds, clean_ds, filepath_output):
    '''画出地表降水率与PCT89之间的二维histogram图.'''
    # 为层云降水和对流降水准备不同分辨率的bins.
    xbins_list = [
        np.logspace(-2, 2, 41),
        np.logspace(-2, 2, 21)
    ]
    ybins_list = [
        np.linspace(200, 300, 41),
        np.linspace(200, 300, 21)
    ]

    # 用Hs存储计算的二维histogram.
    # 第一维是雨型, 第二维是污染分组和差值.
    Hs = np.empty((2, 3), dtype=object)
    for i in range(2):
        for j, ds in enumerate([ds_dusty, ds_clean]):
            ds = ds.isel(npoint=(ds.rainType == i + 1))
            xvar = ds.precipRateNearSurface.values
            yvar = ds.pct89.values
            Hs[i, j] = profile_tools.hist2d(
                xvar, yvar,
                xbins=xbins_list[i],
                ybins=ybins_list[i],
                sigma=1, norm='sum'
            ) * 100
    # 计算两组的差值.
    Hs[:, 2] = Hs[:, 0] - Hs[:, 1]

    # 画图用的参数.
    Rtypes = ['Stratiform', 'Convective']
    groups = ['Dusty', 'clean', 'Dusty - Clean']
    xlist = [(bins[1:] + bins[:-1]) / 2 for bins in xbins_list]
    ylist = [(bins[1:] + bins[:-1]) / 2 for bins in ybins_list]

    # 组图形状为(2, 3).
    # 行表示两种雨型, 前两列表示污染分组, 第三列是两组之差.
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    for (i, j), ax in np.ndenumerate(axes):
        # 前两列画普通图.
        H = Hs[i, j]
        x, y = xlist[i], ylist[i]
        if j <= 1:
            im = ax.contourf(
                x, y, H, levels=40,
                cmap=cmaps.WhBlGrYeRe, extend='both'
            )
        # 第三列画差值.
        else:
            vmax = 0.95 * np.abs(H).max()
            levels = np.linspace(-vmax, vmax, 40)
            im = ax.contourf(
                x, y, H, levels=levels,
                cmap=cmaps.NCV_blu_red, extend='both'
            )
        cbar = fig.colorbar(
            im, ax=ax, aspect=20,
            ticks=mticker.LinearLocator(5),
            format=mticker.PercentFormatter(decimals=2)
        )
        cbar.ax.tick_params(labelsize='x-small')

        # 设置左右小标题.
        ax.set_title(Rtypes[i], loc='left', fontsize='small')
        ax.set_title(groups[j], loc='right', fontsize='small')

    # 为每个子图设置相同的刻度.
    for ax in axes.flat:
        ax.set_xscale('log')
        ax.set_xlim(0.1, 10)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_ylim(200, 300)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(10))
        ax.tick_params(labelsize='x-small')

    # 在组图边缘设置标签.
    for ax in axes[-1, :]:
        ax.set_xlabel('Surface Rain Rate (mm/h)', fontsize='small')
    for ax in axes[:, 0]:
        ax.set_ylabel('PCT89 (K)', fontsize='small')

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
    dirpath_output = Path(config['dirpath_result'], 'statistics')
    helper_tools.new_dir(dirpath_output)

    draw_surfRr_vs_st(
        ds_dusty, ds_clean,
        dirpath_output / 'surfRr_vs_st.png'
    )
    draw_surfRr_vs_pct(
        ds_dusty, ds_clean,
        dirpath_output / 'surfRr_vs_pct.png'
    )
