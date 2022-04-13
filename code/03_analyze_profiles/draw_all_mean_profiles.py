'''
2022-04-12
对于两组数据集和两种雨型, 画出三种降水参量或三种潜热的平均廓线.
组图形状为(2, 3), 行表示两种雨型, 列表示三种参量, 子图中画出污染组和清洁组的
平均廓线. 会画出高度坐标和温度坐标两个版本, 同时再结合降水参量和潜热参量,
最后共画出4张图.
'''
import json
from pathlib import Path
import itertools
import sys
sys.path.append('../modules')

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import helper_tools
import profile_tools
import plot_tools

# 读取配置文件, 将config作为全局变量.
with open('config.json') as f:
    config = json.load(f)

def set_LH_xlim(ax):
    '''设置潜热的横坐标范围.'''
    xmax = np.abs(ax.get_xlim()).max()
    ax.set_xlim(-0.8 * xmax, 1.2 * xmax)

def draw_all_mean_profiles(
    ds_dusty, ds_clean,
    coordname, vartype,
    dirpath_output
):
    # 决定垂直坐标.
    coord = ds_dusty[coordname].values
    # 决定绘制的变量.
    if vartype == 'Rr':
        varnames = ['precipRate', 'Nw', 'Dm']
    elif vartype == 'LH':
        varnames = ['csh', 'slh', 'vph']
    if coordname == 'temp':
        for i in range(3):
            varnames[i] += '_t'

    # 用means存储平均廓线, sems存储标准误差廓线.
    # 第一维表示两种雨型, 第二维表示三种变量, 第三维表示污染分组.
    nh = len(coord)
    means = np.full((2, 3, 2, nh), np.nan)
    sems = np.full((2, 3, 2, nh), np.nan)
    npoints = np.zeros((2, 2), dtype=int)
    for k, ds in enumerate([ds_dusty, ds_clean]):
        for i in range(2):
            ds_temp = ds.isel(npoint=(ds.rainType == i + 1))
            npoints[i, k] = ds_temp.npoint.size
            for j in range(3):
                var = ds_temp[varnames[j]].values
                means[i, j, k, :] = np.nanmean(var, axis=0)
                sems[i, j, k, :] = profile_tools.nansem(var, axis=0)
    # 进行平滑.
    means = profile_tools.smooth_profiles(means, sigma=1)
    sems = profile_tools.smooth_profiles(sems, sigma=1)
    # 仅选取部分高度的数据.
    if coordname == 'height':
        mask = coord >= 1.5
    elif coordname == 'temp':
        mask = coord <= 10
    means = means[..., mask]
    sems = sems[..., mask]
    coord = coord[mask]

    # 根据垂直坐标决定纵坐标的设置.
    if coordname == 'height':
        ylabel = 'Height (km)'
        ylims = (0, 12)
        ybase = 2
    elif coordname == 'temp':
        ylabel = 'Temperature (℃)'
        ylims = (20, -60)
        ybase = 20

    # 根据廓线变量决定横坐标的设置.
    if vartype == 'Rr':
        xlabels = [
            'Rain Rate (mm/hr)',
            'Nw (dBNw)',
            'Dm (mm)'
        ]
        xlim_funcs = [
            lambda ax: ax.set_xlim(0, None),
            lambda ax: ax.set_xlim(20, 40),
            lambda ax: ax.set_xlim(None, None)
        ]
    elif vartype == 'LH':
        xlabels = [
            'CSH LH (K/hr)',
            'SLH LH (K/hr)',
            'VPH LH (K/hr)'
        ]
        xlim_funcs = [set_LH_xlim] * 3

    # 画图参数.
    Rtypes = ['Stratiform', 'Convective']
    groups = ['Dusty', 'Clean']
    colors = ['C1', 'C0']

    # 组图形状为(2, 3).
    # 行表示两种雨型, 列表示三种变量, 每个子图中画出两组廓线.
    fig, axes = plt.subplots(2, 3, figsize=(8, 6))
    fig.subplots_adjust(wspace=0.3, hspace=0.25)
    for (i, j), ax in np.ndenumerate(axes):
        for k in range(2):
            mean = means[i, j, k, :]
            sem = sems[i, j, k, :]
            ax.plot(
                mean, coord, lw=1, c=colors[k],
                label=f'{groups[k]} (N={npoints[i, k]})'
            )
            ax.fill_betweenx(
                coord,
                mean - 1.96 * sem,
                mean + 1.96 * sem,
                color=colors[k], alpha=0.4
            )
        ax.legend(fontsize='xx-small', loc='upper right', handlelength=1)
        ax.set_title(Rtypes[i], fontsize='x-small')

    # 设置坐标轴.
    for (i, j), ax in np.ndenumerate(axes):
        xlim_funcs[j](ax)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.set_ylim(*ylims)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(ybase))
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.tick_params(labelsize='x-small')

    # 在组图边缘设置标签.
    for xlabel, ax in zip(xlabels, axes[1, :]):
        ax.set_xlabel(xlabel, fontsize='x-small')
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, fontsize='x-small')

    # 潜热变量加上x=0的辅助线.
    if vartype == 'LH':
        for ax in axes.flat:
            ax.axvline(0, color='k', ls='--', lw=0.6)

    # 为子图标出字母标识.
    plot_tools.letter_axes(axes, 0.06, 0.95, fontsize='x-small')

    # 保存图片.
    stem = f'all_mean_{vartype}_profiles'
    if coordname == 'temp':
        stem += '_t'
    filepath_output = dirpath_output / (stem + '.png')
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

    coordnames = ['height', 'temp']
    vartypes = ['Rr', 'LH']
    for coordname, vartype in itertools.product(coordnames, vartypes):
        draw_all_mean_profiles(
            ds_dusty, ds_clean,
            coordname, vartype,
            dirpath_output
        )
