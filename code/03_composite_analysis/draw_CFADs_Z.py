#----------------------------------------------------------------------------
# 2021/07/08
# 画出高度坐标与温度坐标下雷达反射率的CFAD图.
#
# 画两张组图,分别代表高度坐标与温度坐标.
# 组图形状为(2, 3),行表示雨型,列表示污染组,清洁组,污染-清洁之差.
#----------------------------------------------------------------------------
import json
from pathlib import Path

import sys
sys.path.append('../modules')
from profile_funcs import calc_cfad

import numpy as np
import xarray as xr

import cmaps
import matplotlib as mpl
import matplotlib.pyplot as plt

# 读取配置文件,作为全局变量使用.
with open('config.json', 'r') as f:
    config = json.load(f)

def draw_CFAD_hgt(dusty_ds, clean_ds, output_filepath):
    '''
    画出高度坐标下两组个例的不同雨型的CFAD图.

    组图形状为(2, 3),行表示雨型,前两列表示污染分组,第三列为两组之差.
    '''
    # 设置画CFAD图的bins.
    nx = 50
    ny = 84     # ny不可以随便设置,需要考虑到rangeBinSize.
    xbins = np.linspace(10, 60, nx + 1)
    ybins = np.linspace(1.5, 12, ny + 1)

    # 计算CFAD.
    # 第一维是雨型,第二维是污染分组.
    hgt = dusty_ds.height.data
    cfads = np.zeros((2, 2, ny, nx))
    diffs = np.zeros((2, ny, nx))
    for j, ds in enumerate([dusty_ds, clean_ds]):
        for i in range(2):
            z = ds.zFactorCorrected.isel(npoint=(ds.rainType == i + 1)).data
            # CFAD单位设为百分比.
            cfads[i, j, :, :] = calc_cfad(
                z, hgt, xbins, ybins, norm='sum'
            ) * 100

    # 计算同一雨型的差值.
    for i in range(2):
        diffs[i, :, :] = cfads[i, 0, :, :] - cfads[i, 1, :, :]

    # 画图用的参数.
    Rstrs = ['Stratiform', 'Convective']
    groups = ['Dusty', 'Clean', 'Dusty - Clean']
    # 组图形状为(2, 3).
    # 行表示两种雨型,前两列表示两个分组,第三列是两组之差.
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

    # 在每张子图上画出CFAD图.
    for i in range(2):
        for j in range(3):
            ax = axes[i, j]
            # 前两列画普通CFAD图.
            if j <= 1:
                cfad = cfads[i, j, :, :]
                im = ax.pcolormesh(
                    xbins, ybins, cfad, cmap=cmaps.WhBlGrYeRe
                )
            # 第三列画两组CFAD之差.
            else:
                diff = diffs[i]
                zmax = 0.95 * np.abs(diff).max()
                im = ax.pcolormesh(
                    xbins, ybins, diff, cmap=cmaps.BlWhRe,
                    vmin=-zmax, vmax=zmax
                )
            cbar = fig.colorbar(
                im, ax=ax, aspect=30, extend='both',
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
        ax.set_xlim(10, 60)
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
        ax.set_ylim(0, 12)
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
        ax.tick_params(labelsize='x-small')

    # 在最下面标上xlabel.
    for ax in axes[1, :]:
        ax.set_xlabel('Reflectivity (dBZ)', fontsize='small')
    # 在最左边标上ylabel.
    for ax in axes[:, 0]:
        ax.set_ylabel('Height (km)', fontsize='small')

    # 保存图片.
    fig.savefig(str(output_filepath), dpi=300, bbox_inches='tight')
    plt.close(fig)

def draw_CFAD_temp(dusty_ds, clean_ds, output_filepath):
    '''基本同draw_CFAD_hgt函数,不过垂直坐标换为温度.'''
    # 设置画CFAD图的bins.
    nx = 50
    ny = 70     # ny不可以随便设置,需要考虑到rangeBinSize.
    xbins = np.linspace(10, 60, nx + 1)
    ybins = np.linspace(-60, 10, ny + 1)

    # 计算CFAD.
    # 第一维是雨型,第二维是污染分组.
    hgt = dusty_ds.temp.data
    cfads = np.zeros((2, 2, ny, nx))
    diffs = np.zeros((2, ny, nx))
    for j, ds in enumerate([dusty_ds, clean_ds]):
        # 因为xarray会自动把含缺测的整型数组变为浮点型.
        rainType = ds.rainType.fillna(-9999).astype(int)
        for i in range(2):
            z = ds.zFactorCorrected_t.isel(npoint=(rainType == i + 1)).data
            # CFAD单位设为百分比.
            cfads[i, j, :, :] = calc_cfad(
                z, hgt, xbins, ybins, norm='sum'
            ) * 100

    # 计算同一雨型的差值.
    for i in range(2):
        diffs[i, :, :] = cfads[i, 0, :, :] - cfads[i, 1, :, :]

    # 画图用的参数.
    Rstrs = ['Stratiform', 'Convective']
    groups = ['Dusty', 'Clean', 'Dusty - Clean']
    # 组图形状为(2, 3).
    # 行表示两种雨型,前两列表示两个分组,第三列是两组之差.
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

    # 在每张子图上画出CFAD图.
    for i in range(2):
        for j in range(3):
            ax = axes[i, j]
            # 前两列画普通CFAD图.
            if j <= 1:
                cfad = cfads[i, j, :, :]
                im = ax.pcolormesh(
                    xbins, ybins, cfad, cmap=cmaps.WhBlGrYeRe
                )
            # 第三列画两组CFAD之差.
            else:
                diff = diffs[i]
                zmax = 0.95 * np.abs(diff).max()
                im = ax.pcolormesh(
                    xbins, ybins, diff, cmap=cmaps.BlWhRe,
                    vmin=-zmax, vmax=zmax
                )
            cbar = fig.colorbar(
                im, ax=ax, aspect=30, extend='both',
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
        ax.set_xlim(10, 60)
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
        ax.set_ylim(20, -60)
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(20))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(10))
        ax.tick_params(labelsize='x-small')

    # 在最下面标上xlabel.
    for ax in axes[1, :]:
        ax.set_xlabel('Reflectivity (dBZ)', fontsize='small')
    # 在最左边标上ylabel.
    for ax in axes[:, 0]:
        ax.set_ylabel('Temperature (℃)', fontsize='small')

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
    output_dirpath = Path(config['result_dirpath']) / 'CFADs'
    if not output_dirpath.exists():
        output_dirpath.mkdir()

    draw_CFAD_hgt(dusty_ds, clean_ds, output_dirpath / 'CFADs_hgt.png')
    draw_CFAD_temp(dusty_ds, clean_ds, output_dirpath / 'CFADs_temp.png')
