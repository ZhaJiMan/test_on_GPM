'''
2022-04-12
类似于draw_binned_differences, 但是廓线只取降水率, 列表示三种气象变量.
'''
import json
from pathlib import Path
import itertools
import sys
sys.path.append('../modules')

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

import helper_tools
import profile_tools
import plot_tools
from draw_binned_profiles import get_bin_labels

# 读取配置文件, 将config作为全局变量.
with open('config.json') as f:
    config = json.load(f)

def draw_binned_condition(
    ds_dusty, ds_clean,
    coordname, filepath_output
):
    # 决定垂直坐标.
    coord = ds_dusty[coordname].values
    # 决定绘制的变量.
    varname = 'precipRate'
    if coordname == 'temp':
        varname += '_t'

    # 简便起见, 两种雨型共用一组bins.
    binnames = ['w', 'tcwv', 'cape']
    bins_arr = np.array([
        [-3, -0.5, 0, 2],
        [0, 15, 20, 50],
        [0, 5, 50, 500]
    ], dtype=float)
    nbin = len(binnames)

    # 用data存储每个分组中的廓线数组, 用means存储平均廓线.
    # 第一维表示污染分组, 第二维表示两种雨型, 第三维表示三种bin变量,
    # 第四维表示bin的分组.
    nh = len(coord)
    data = np.empty((2, 2, 3, nbin), dtype=object)
    means = np.zeros((2, 2, 3, nbin, nh))
    npoints = np.zeros((2, 2, 3, nbin), dtype=int)
    for i, ds in enumerate([ds_dusty, ds_clean]):
        for j in range(2):
            ds_temp = ds.isel(npoint=(ds.rainType == j + 1))
            yvar = ds_temp[varname].values
            for k, binname in enumerate(binnames):
                xvar = ds_temp[binname].values
                binner = profile_tools.ProfileBinner(
                    xvar, yvar, bins_arr[k]
                )
                data[i, j, k, :] = binner.groups
                means[i, j, k, :, :] = binner.mean()
                npoints[i, j, k, :] = binner.counts

    # 计算污染组与清洁组的平均廓线之差, 并利用data进行t检验.
    diffs = means[0, ...] - means[1, ...]
    masks = np.zeros(diffs.shape, dtype=bool)
    data_dusty = data[0, ...]
    data_clean = data[1, ...]
    for index in np.ndindex(data_dusty.shape):
        yvar_dusty = data_dusty[index]
        yvar_clean = data_clean[index]
        if yvar_dusty is not None and yvar_clean is not None:
            masks[index] = profile_tools.ttest_profiles(
                yvar_dusty, yvar_clean, alpha=0.05
            )

    # 进行平滑并将未通过检验的部分设为缺测.
    diffs = profile_tools.smooth_profiles(diffs, sigma=1)
    tested = np.where(masks, diffs, np.nan)
    # 仅选取部分高度的数据.
    if coordname == 'height':
        mask = coord >= 1.5
    elif coordname == 'temp':
        mask = coord <= 10
    diffs = diffs[..., mask]
    tested = tested[..., mask]
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

    # 根据bin变量决定柱状图的横坐标.
    xlabel = 'Rain Rate (mm/h)'
    longnames = []
    for binname in binnames:
        if binname == 'w':
            longname = 'ω (Pa/s)'
        elif binname == 'tcwv':
            longname = 'TCWV (kg/m²)'
        elif binname == 'mfd':
            longname = 'MFD (kg/m²/s)'
        elif binname == 'cape':
            longname = 'CAPE (J/kg)'
        longnames.append(longname)

    # 画图参数.
    Rtypes = ['Stratiform', 'Convective']
    colors = ['C0', 'C2', 'C1', 'C3']
    bin_labels_list = [get_bin_labels(bins) for bins in bins_arr]

    # 组图形状为(2, 3).
    # 行表示两种雨型, 列表示三种bin变量, 每个子图中画出nbin组廓线,
    # 并分裂出一个子图用柱状图画出廓线数量.
    fig, axes = plt.subplots(2, 3, figsize=(10, 10))
    fig.subplots_adjust(wspace=0.3, hspace=0.25, right=0.8)
    for (i, j), ax in np.ndenumerate(axes):
        for k in range(nbin):
            # 用加粗的半透明线表示通过检验的部分.
            ax.plot(diffs[i, j, k, :], coord, lw=1, c=colors[k])
            ax.plot(tested[i, j, k, :], coord, lw=3, c=colors[k], alpha=0.5)
        # 添加x=0处的辅助线.
        ax.axvline(0, color='k', ls='--', lw=0.6)
        ax.set_title(Rtypes[i], fontsize='x-small')

    # 设置坐标轴.
    for (i, j), ax in np.ndenumerate(axes):
        xmax = np.abs(ax.get_xlim()).max()
        ax.set_xlim(-xmax, xmax)
        ax.set_xlabel(xlabel, fontsize='x-small')
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.set_ylim(*ylims)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(ybase))
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.tick_params(labelsize='x-small')

    # 在组图边缘设置纵坐标标签.
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, fontsize='x-small')

    # 分出子图画柱状图.
    width = 0.25
    pad = 0.32
    x = np.arange(nbin)
    patch_dusty = mpatches.Patch(color='w', ec='k', lw=1, label='Dusty')
    patch_clean = mpatches.Patch(
        color='w', ec='k', lw=1, ls='--', label='Clean'
    )
    for (i, j), ax in np.ndenumerate(axes):
        divider = make_axes_locatable(ax)
        axb = divider.append_axes('bottom', size=0.8, pad=0.4)

        # 分别画出污染组和清洁组的廓线数.
        axb.bar(
            x - pad / 2, npoints[0, i, j, :], width,
            color=colors, ec='k', lw=1
        )
        axb.bar(
            x + pad / 2, npoints[1, i, j, :], width,
            color=colors, ec='k', lw=1, ls='--'
        )
        axb.legend(
            handles=[patch_dusty, patch_clean],
            loc='upper right', fontsize='xx-small'
        )

        # 设置坐标轴.
        axb.set_xticks(x)
        axb.set_xticklabels(bin_labels_list[j])
        axb.set_xlabel(longnames[j], fontsize='x-small')
        axb.set_ylim(0, 1.2 * axb.get_ylim()[1])
        if j == 0:
            axb.set_ylabel('Number', fontsize='x-small')
        axb.tick_params(labelsize='x-small')

    # 为子图标出字母标识.
    plot_tools.letter_axes(axes, 0.06, 0.95, fontsize='x-small')

    # 保存图片.
    fig.savefig(str(filepath_output), dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    # 读取两组数据集.
    dirpath_input = Path(config['dirpath_input'])
    ds_dusty = xr.load_dataset(str(dirpath_input / 'data_dusty.nc'))
    ds_clean = xr.load_dataset(str(dirpath_input / 'data_clean.nc'))

    # 创建输出目录.
    dirpath_output = Path(
        config['dirpath_result'], 'Rr_profiles', 'binned_condition'
    )
    helper_tools.new_dir(dirpath_output, parents=True)

    # 画出高度坐标与温度坐标下的廓线差异图.
    draw_binned_condition(
        ds_dusty, ds_clean, coordname='height',
        filepath_output=(dirpath_output / 'Rr.png')
    )
    draw_binned_condition(
        ds_dusty, ds_clean, coordname='temp',
        filepath_output=(dirpath_output / 'Rr_t.png')
    )
