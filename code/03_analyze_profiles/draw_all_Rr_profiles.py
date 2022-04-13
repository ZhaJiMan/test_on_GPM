'''
2022-04-12
分别画出高度坐标与温度坐标下的所有降水率廓线, 以显示坐标转换的效果.
两张组图分别代表两种坐标. 组图形状为(2, 2), 行表示两种雨型, 列表示污染组和
清洁组, 子图中画出排列在一起的所有降水率廓线.
'''
import json
from pathlib import Path
import sys
sys.path.append('../modules')

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

import helper_tools
import plot_tools

# 读取配置文件, 将config作为全局变量.
with open('config.json') as f:
    config = json.load(f)

def draw_all_profiles(ds_dusty, ds_clean, coordname, filepath_output):
    '''画出两种数据两种雨型的所有降水廓线. coordname用于指定垂直坐标.'''
    # 决定垂直坐标.
    coord = ds_dusty[coordname].values
    varname = 'precipRate'
    if coordname == 'temp':
        varname += '_t'

    # 用profiles保存降水廓线数据.
    # 第一维表示两种雨型, 第二维表示污染分组.
    profiles = np.empty((2, 2), dtype=object)
    for j, ds in enumerate([ds_dusty, ds_clean]):
        rainType = ds.rainType.values
        var = ds[varname].values
        for i in range(2):
            mask = rainType == i + 1
            profiles[i, j] = var[mask, :]

    # 根据垂直坐标选取设置.
    if coordname == 'height':
        ylabel = 'Height (km)'
        ylims = (0, 12)
        ybase = 2
    elif coordname == 'temp':
        ylabel = 'Temperature (℃)'
        ylims = (20, -60)
        ybase = 20
    # 画图的参数.
    Rtypes = ['Stratiform', 'Convective']
    groups = ['Dusty', 'Clean']

    # 设置cmap和norm.
    cmap, norm = plot_tools.get_rain_cmap()
    cmap.set_under('white')
    cmap.set_bad('dimgray')

    # 组图形状为(2, 2).
    # 行表示雨型, 列表示污染分组.
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    fig.subplots_adjust(wspace=0.15, hspace=0.3)
    for (i, j), ax in np.ndenumerate(axes):
        profile = profiles[i, j]
        x = np.arange(profile.shape[0])
        im = ax.pcolormesh(x, coord, profile.T, cmap=cmap, norm=norm)
        ax.set_title(Rtypes[i], fontsize='x-small', loc='left')
        ax.set_title(groups[j], fontsize='x-small', loc='right')
    cbar = fig.colorbar(
        im, ax=axes, extend='both', orientation='horizontal',
        shrink=0.8, aspect=30, pad=0.1
    )
    cbar.set_label('Rain Rate (mm/hr)', fontsize='x-small')
    cbar.ax.tick_params(labelsize='x-small')

    # 设置坐标轴.
    for ax in axes.flat:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(6))
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.set_ylim(*ylims)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(ybase))
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.tick_params(labelsize='x-small')

    # 在组图边缘设置标签.
    for ax in axes[-1, :]:
        ax.set_xlabel('Number', fontsize='x-small')
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, fontsize='x-small')

    # 为子图标出字母标识.
    plot_tools.letter_axes(axes, 0.04, 0.94, fontsize='x-small')

    # 保存图片.
    fig.savefig(str(filepath_output), dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    # 读取两组数据集.
    dirpath_input = Path(config['dirpath_input'])
    ds_dusty = xr.load_dataset(str(dirpath_input / 'data_dusty.nc'))
    ds_clean = xr.load_dataset(str(dirpath_input / 'data_clean.nc'))

    # 创建输出目录.
    dirpath_output = Path(config['dirpath_result'], 'Rr_profiles', 'all')
    helper_tools.new_dir(dirpath_output, parents=True)

    draw_all_profiles(
        ds_dusty, ds_clean,
        coordname='height',
        filepath_output=(dirpath_output / 'all_Rr_profiles.png')
    )
    draw_all_profiles(
        ds_dusty, ds_clean,
        coordname='temp',
        filepath_output=(dirpath_output / 'all_Rr_profiles_t.png')
    )

