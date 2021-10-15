#----------------------------------------------------------------------------
# 2021/05/08
# 分别画出高度坐标与温度坐标下的所有降水廓线,以显示坐标转换的效果.
#
# 两张图代表两种坐标,每张图由四个子图组成,行表示雨型,列表示污染分组.
#----------------------------------------------------------------------------
import json
from pathlib import Path
import sys
sys.path.append('../modules')

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt

from helper_funcs import letter_subplots

# 读取配置文件,作为全局变量使用.
with open('config.json', 'r') as f:
    config = json.load(f)

def draw_profiles_hgt(dusty_ds, clean_ds, output_filepath):
    '''
    以廓线数*高度的二维填色图画出dusty cases与clean cases所有廓线,
    以明确廓线的范围和缺测区域.

    组图形状为2*2,行代表雨型分组,列代表污染分组.
    '''
    # data第一维表示两种雨型,第二维表示两组个例.
    data = np.empty((2, 2), dtype=object)
    for j, ds in enumerate([dusty_ds, clean_ds]):
        rainType = ds.rainType.data
        precipRate = ds.precipRate.to_masked_array()
        y = ds.height.data
        # 选出层云和对流降水.
        for i in range(2):
            flag = rainType == (i + 1)
            Rr = precipRate[flag, :]
            x = np.arange(Rr.shape[0])
            data[i, j] = (x, y, Rr)

    bounds = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0]
    nbin = len(bounds) - 1
    norm = mpl.colors.BoundaryNorm(bounds, nbin)
    cmap = mpl.cm.get_cmap('jet', nbin)
    cmap.set_under('white')
    # 将缺测值设为红色.
    cmap.set_bad('black')

    # 组图的行表示雨型,列表示污染分组.
    groups = ['Dusty', 'Clean']
    Rtypes = ['Stratiform', 'Convective']
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    fig.subplots_adjust(wspace=0.15, hspace=0.3)
    for i in range(2):
        Rtype = Rtypes[i]
        for j in range(2):
            ax = axes[i, j]
            group = groups[j]
            x, y, Rr = data[i, j]

            # 画出所有降水廓线.
            im = ax.pcolormesh(x, y, Rr[:-1, :-1].T, cmap=cmap, norm=norm)

            ax.set_title(Rtype, fontsize='x-small', loc='left')
            ax.set_title(group, fontsize='x-small', loc='right')
            ax.set_ylim(0, 16)
            ax.tick_params(labelsize='x-small')
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(7))
            ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
            ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(4))
            ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(2))

    cbar = fig.colorbar(
        im, ax=axes.ravel().tolist(), extend='both',
        orientation='horizontal', shrink=0.8, aspect=30, pad=0.1
    )
    cbar.ax.tick_params(labelsize='x-small')
    cbar.set_label('Rain Rate [mm/h]', fontsize='x-small')

    # 在组图边缘设置label.
    for ax in axes[-1, :]:
        ax.set_xlabel('Number', fontsize='x-small')
    for ax in axes[:, 0]:
        ax.set_ylabel('Height [km]', fontsize='x-small')

    # 为子图标出字母标识.
    letter_subplots(axes, (0.04, 0.94), 'x-small')

    fig.savefig(str(output_filepath), dpi=300, bbox_inches='tight')
    plt.close(fig)

def draw_profiles_temp(dusty_ds, clean_ds, output_filepath):
    '''基本上等同于draw_profiles_hgt函数,但是将纵坐标换为温度.'''
    fill_value = -9999.9
    # data第一维表示两种雨型,第二维表示两组个例.
    data = np.empty((2, 2), dtype=object)
    for j, ds in enumerate([dusty_ds, clean_ds]):
        rainType = ds.rainType.data
        precipRate = ds.precipRate_t.to_masked_array()
        y = ds.temp.data
        # 选出层云和对流降水.
        for i in range(2):
            flag = rainType == (i + 1)
            Rr = precipRate[flag, :]
            x = np.arange(Rr.shape[0])
            data[i, j] = (x, y, Rr)

    bounds = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0]
    nbin = len(bounds) - 1
    norm = mpl.colors.BoundaryNorm(bounds, nbin)
    cmap = mpl.cm.get_cmap('jet', nbin)
    cmap.set_under('white')
    # 将缺测值设为红色.
    cmap.set_bad('black')

    groups = ['Dusty', 'Clean']
    Rtypes = ['Stratiform', 'Convective']
    # 组图的行表示雨型,列表示污染分组.
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    fig.subplots_adjust(wspace=0.15, hspace=0.3)
    for i in range(2):
        Rtype = Rtypes[i]
        for j in range(2):
            ax = axes[i, j]
            group = groups[j]
            x, y, Rr = data[i, j]

            # 画出所有降水廓线.
            im = ax.pcolormesh(x, y, Rr[:-1, :-1].T, cmap=cmap, norm=norm)

            ax.set_title(Rtype, fontsize='x-small', loc='left')
            ax.set_title(group, fontsize='x-small', loc='right')
            ax.set_ylim(20, -60)
            ax.tick_params(labelsize='x-small')
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(7))
            ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
            ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(20))
            ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(10))

    cbar = fig.colorbar(
        im, ax=axes.ravel().tolist(), extend='both',
        orientation='horizontal', shrink=0.8, aspect=30, pad=0.1
    )
    cbar.ax.tick_params(labelsize='x-small')
    cbar.set_label('Rain Rate (mm/h)', fontsize='x-small')

    # 在组图边缘设置label.
    for ax in axes[-1, :]:
        ax.set_xlabel('Number', fontsize='x-small')
    for ax in axes[:, 0]:
        ax.set_ylabel('Temperature (℃)', fontsize='x-small')

    # 为子图标出字母标识.
    letter_subplots(axes, (0.04, 0.94), 'x-small')

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
    output_dirpath = Path(config['result_dirpath']) / 'Rr_profiles'
    if not output_dirpath.exists():
        output_dirpath.mkdir()

    # 画出两种坐标下所有的降水廓线.
    draw_profiles_hgt(
        dusty_ds, clean_ds,
        output_dirpath / 'all_profiles_hgt.png'
    )
    draw_profiles_temp(
        dusty_ds, clean_ds,
        output_dirpath / 'all_profiles_temp.png'
    )
