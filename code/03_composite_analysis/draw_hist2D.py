#----------------------------------------------------------------------------
# 2021/08/21
# 画出两组个例的两种雨型的两个变量之间构成的二维histogram(PDF)分布图.
#
# 画出的变量有:
# - 地表降水率 vs. 雨顶温度
# - 地表降水率 vs. PCT89
# 其中地表降水率采用对数坐标表示.
#
# 组图形状为(2, 3),行表示雨型,列表示dusty,clean和dusty-clean.
# 画出两张图,分别代表不同的变量组合.
#----------------------------------------------------------------------------
import json
from pathlib import Path
import sys
sys.path.append('../modules')

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmaps

from helper_funcs import letter_subplots

# 读取配置文件,作为全局变量使用.
with open('config.json', 'r') as f:
    config = json.load(f)

# 全局的平滑参数.
SMOOTH = True
SIGMA = 1

def hist2D(x, y, bins):
    '''
    统计x与y构成的二维histogram.

    bins参数请参考np.histogram2d.
    返回以百分比为单位的histogram.
    '''
    H = np.histogram2d(x, y, bins)[0].T
    # 进行平滑.
    if SMOOTH:
        H = gaussian_filter(H, sigma=SIGMA)
    # 归一化为出现频率.
    s = H.sum()
    if s > 0:
        H = H / s * 100

    return H

def draw_surfRr_vs_st(dusty_ds, clean_ds, output_filepath):
    '''
    画出地表降水率与雨顶温度之间的二维histogram图,以展现
    二者的概率密度分布.
    '''
    # 为层云降水和对流降水分别准备bins.
    bins_list = [
        [np.logspace(-2, 2, 50), np.linspace(-60, 20, 50)],
        [np.logspace(-2, 2, 20), np.linspace(-60, 20, 20)],
    ]
    # 用data存储画图用的数据.
    # 行表示两种雨型,列表示dusty,clean和dusty-clean.
    data = np.empty((2, 3), dtype=object)
    for i in range(2):
        bins = bins_list[i]
        xbins, ybins = bins
        # 计算画contourf时的xy中点.
        xc = (xbins[1:] + xbins[:-1]) / 2
        yc = (ybins[1:] + ybins[:-1]) / 2
        for j, ds in enumerate([dusty_ds, clean_ds]):
            ds = ds.isel(npoint=(ds.rainType == i + 1))
            xvar = ds.precipRateNearSurface.data
            yvar = ds.tempStormTop.data

            # 计算二维histogram并平滑.
            H = hist2D(xvar, yvar, bins)
            data[i, j] = (xc, yc, H)

    # 计算同一雨型的dusty和clean组的差值,并存入data中.
    for i in range(2):
        xc, yc, H1 = data[i, 0]
        xc, yc, H2 = data[i, 1]
        data[i, 2] = (xc, yc, H1 - H2)

    # 画图用的参数.
    Rstrs = ['Stratiform', 'Convective']
    groups = ['Dusty', 'clean', 'Dusty - Clean']
    # 组图形状为(2, 3).
    # 行表示雨型,前两列表示两个分组,第三列是两组之差.
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

    # 在每张子图上画出contourf图.
    for i in range(2):
        for j in range(3):
            ax = axes[i, j]
            xc, yc, H = data[i, j]
            # 前两列画普通图.
            if j <= 1:
                im = ax.contourf(
                    xc, yc, H, levels=100,
                    cmap=cmaps.WhBlGrYeRe, extend='both'
                )
            # 第三列画前两列的差值.
            else:
                Hmax = 0.95 * np.abs(H).max()
                levels = np.linspace(-Hmax, Hmax, 100)
                im = ax.contourf(
                    xc, yc, H, levels=levels,
                    cmap=cmaps.NCV_blu_red, extend='both'
                )
            cbar = fig.colorbar(
                im, ax=ax, aspect=20,
                ticks=mpl.ticker.MaxNLocator(6),
                format=mpl.ticker.PercentFormatter(decimals=2)
            )
            cbar.ax.tick_params(labelsize='x-small')

    # 设置子图的左右小标题.
    for i, Rstr in enumerate(Rstrs):
        for j, group in enumerate(groups):
            ax = axes[i, j]
            ax.set_title(group, loc='left', fontsize='small')
            ax.set_title(Rstr, loc='right', fontsize='small')

    # 为每个子图设置相同的ticks.
    for ax in axes.flat:
        ax.set_xscale('log')
        ax.set_xlim(1E-2, 1E2)
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.set_ylim(20, -60)
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(20))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(10))
        ax.tick_params(labelsize='x-small')
    # 在第一列和最下面一行设置label.
    for ax in axes[-1, :]:
        ax.set_xlabel('Rain Rate (mm/h)', fontsize='small')
    for ax in axes[:, 0]:
        ax.set_ylabel('Temperature of Storm Top (℃)', fontsize='small')

    # 为子图标出字母标识.
    letter_subplots(axes, (0.06, 0.96), 'small')

    # 保存图片.
    fig.savefig(str(output_filepath), dpi=300, bbox_inches='tight')
    plt.close(fig)

def draw_surfRr_vs_pct(dusty_ds, clean_ds, output_filepath):
    '''
    画出地表降水率与PCT89之间的二维histogram图,以展现
    二者的概率密度分布.
    '''
    # 为层云降水和对流降水分别准备bins.
    bins_list = [
        [np.logspace(-2, 2, 50), np.linspace(200, 300, 50)],
        [np.logspace(-2, 2, 20), np.linspace(200, 300, 20)],
    ]
    # 用data存储画图用的数据.
    # 行表示两种雨型,列表示dusty,clean和dusty-clean.
    data = np.empty((2, 3), dtype=object)
    for i in range(2):
        bins = bins_list[i]
        xbins, ybins = bins
        # 计算画contourf时的xy中点.
        xc = (xbins[1:] + xbins[:-1]) / 2
        yc = (ybins[1:] + ybins[:-1]) / 2
        for j, ds in enumerate([dusty_ds, clean_ds]):
            ds = ds.isel(npoint=(ds.rainType == i + 1))
            xvar = ds.precipRateNearSurface.data
            yvar = ds.PCT89.data

            # 计算二维histogram并平滑.
            H = hist2D(xvar, yvar, bins)
            data[i, j] = (xc, yc, H)

    # 计算同一雨型的dusty和clean组的差值,并存入data中.
    for i in range(2):
        xc, yc, H1 = data[i, 0]
        xc, yc, H2 = data[i, 1]
        data[i, 2] = (xc, yc, H1 - H2)

    # 画图用的参数.
    Rstrs = ['Stratiform', 'Convective']
    groups = ['Dusty', 'clean', 'Dusty - Clean']
    # 组图形状为(2, 3).
    # 行表示雨型,前两列表示两个分组,第三列是两组之差.
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

    # 在每张子图上画出contourf图.
    for i in range(2):
        for j in range(3):
            ax = axes[i, j]
            xc, yc, H = data[i, j]
            # 前两列画普通图.
            if j <= 1:
                im = ax.contourf(
                    xc, yc, H, levels=100,
                    cmap=cmaps.WhBlGrYeRe, extend='both'
                )
            # 第三列画前两列的差值.
            else:
                Hmax = 0.95 * np.abs(H).max()
                levels = np.linspace(-Hmax, Hmax, 100)
                im = ax.contourf(
                    xc, yc, H, levels=levels,
                    cmap=cmaps.NCV_blu_red, extend='both'
                )
            cbar = fig.colorbar(
                im, ax=ax, aspect=20,
                ticks=mpl.ticker.MaxNLocator(6),
                format=mpl.ticker.PercentFormatter(decimals=2)
            )
            cbar.ax.tick_params(labelsize='x-small')

    # 设置子图的左右小标题.
    for i, Rstr in enumerate(Rstrs):
        for j, group in enumerate(groups):
            ax = axes[i, j]
            ax.set_title(group, loc='left', fontsize='small')
            ax.set_title(Rstr, loc='right', fontsize='small')

    # 为每个子图设置相同的ticks.
    for ax in axes.flat:
        ax.set_xscale('log')
        ax.set_xlim(1E-2, 1E2)
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.set_ylim(200, 300)
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(20))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(10))
        ax.tick_params(labelsize='x-small')
    # 在第一列和最下面一行设置label.
    for ax in axes[-1, :]:
        ax.set_xlabel('Rain Rate (mm/h)', fontsize='small')
    for ax in axes[:, 0]:
        ax.set_ylabel('PCT89 (K)', fontsize='small')

    # 为子图标出字母标识.
    letter_subplots(axes, (0.06, 0.96), 'small')

    # 保存图片.
    fig.savefig(str(output_filepath), dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    # 读取dusty cases和clean cases.
    input_dirpath = Path(config['input_dirpath'])
    with open(str(input_dirpath / 'found_cases.json'), 'r') as f:
        records = json.load(f)
    dusty_ds = xr.load_dataset(records['dusty']['profile_filepath'])
    clean_ds = xr.load_dataset(records['clean']['profile_filepath'])

    # 若输出目录不存在,那么新建.
    output_dirpath = Path(config['result_dirpath']) / 'statistics'
    if not output_dirpath.exists():
        output_dirpath.mkdir()

    draw_surfRr_vs_st(
        dusty_ds, clean_ds,
        output_dirpath / 'surfRr_vs_st.png'
    )
    draw_surfRr_vs_pct(
        dusty_ds, clean_ds,
        output_dirpath / 'surfRr_vs_pct.png'
    )
