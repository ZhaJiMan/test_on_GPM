'''
2022-04-12
画出两组数据集两种雨型分bin平均后的廓线的差值.
组图形状为(2, 3), 行表示两种雨型, 列表示三种廓线变量, 每张子图中画出nbin组
平均廓线之差.
当组图设定为降水图时, 三种廓线变量取降水率和DSD.
当组图设定为潜热图时, 三种廓线变量取三种潜热.
组图可以设定分bin变量和垂直坐标.
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

def draw_binned_differences(
    ds_dusty, ds_clean,
    coordname, vartype, binname,
    bins_stra, bins_conv,
    dirpath_output
):
    '''
    画出两组数据集按两种雨型分类, 进行分bin平均的廓线的差值.

    coordname指定垂直坐标, 可选高度和温度.
    vartype指定画降水参量还是潜热参量. 'Rr'表示画降水率和DSD,
    'LH'表示画三种潜热.
    binname指定分bin的变量, 可选地表降水率, 雨顶温度和PCT89.
    bins_stra和bins_conv分别给出两种雨型分bin的方案, 要求二者等长.
    '''
    # 检查bins长度.
    if len(bins_stra) != len(bins_conv):
        raise ValueError('要求bins_stra和bins_conv长度相同')
    nbin = len(bins_stra) - 1

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

    # 用data存储每个分组中的廓线数组, 用means存储平均廓线.
    # 第一维表示污染分组, 第二维表示两种雨型, 第三维表示三种变量,
    # 第四维表示bin的分组.
    nh = len(coord)
    data = np.empty((2, 2, 3, nbin), dtype=object)
    means = np.zeros((2, 2, 3, nbin, nh))
    npoints = np.zeros((2, 2, nbin), dtype=int)
    for i, ds in enumerate([ds_dusty, ds_clean]):
        for j, bins in enumerate([bins_stra, bins_conv]):
            ds_temp = ds.isel(npoint=(ds.rainType == j + 1))
            xvar = ds_temp[binname].values
            yvars = [ds_temp[varname].values for varname in varnames]
            binner = profile_tools.ProfileBinner(xvar, yvars, bins)
            data[i, j, :, :] = binner.groups
            means[i, j, :, :, :] = binner.mean()
            npoints[i, j, :] = binner.counts

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

    # 根据廓线变量决定横坐标的设置.
    if vartype == 'Rr':
        xlabels = [
            'Rain Rate (mm/hr)',
            'Nw (dBNw)',
            'Dm (mm)'
        ]
    elif vartype == 'LH':
        xlabels = [
            'CSH LH (K/hr)',
            'SLH LH (K/hr)',
            'VPH LH (K/hr)'
        ]

    # 根据bin变量决定标题.
    if binname == 'precipRateNearSurface':
        longname = 'Surface Rain Rate (mm/hr)'
    elif binname == 'heightStormTop':
        longname == 'Height of Storm Top (km)'
    elif binname == 'tempStormTop':
        longname = 'Temperature of Storm Top (℃)'
    elif binname == 'pct89':
        longname = 'PCT89 (K)'

    # 画图参数.
    Rtypes = ['Stratiform', 'Convective']
    colors = ['C0', 'C2', 'C1']
    bin_labels_list = [
        get_bin_labels(bins_stra),
        get_bin_labels(bins_conv)
    ]

    # 组图形状为(2, 3).
    # 行表示两种雨型, 列表示三种变量, 每个子图中画出nbin组廓线,
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
        ax.set_xlabel(xlabels[j], fontsize='x-small')
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
            x - pad / 2, npoints[0, i, :], width,
            color=colors, ec='k', lw=1
        )
        axb.bar(
            x + pad / 2, npoints[1, i, :], width,
            color=colors, ec='k', lw=1, ls='--'
        )
        axb.legend(
            handles=[patch_dusty, patch_clean],
            loc='upper right', fontsize='xx-small'
        )

        # 设置坐标轴.
        axb.set_xticks(x)
        axb.set_xticklabels(bin_labels_list[i])
        axb.set_xlabel(longname, fontsize='x-small')
        axb.set_ylim(0, 1.2 * axb.get_ylim()[1])
        if j == 0:
            axb.set_ylabel('Number', fontsize='x-small')
        axb.tick_params(labelsize='x-small')

    # 为子图标出字母标识.
    plot_tools.letter_axes(axes, 0.06, 0.95, fontsize='x-small')
    if binname == 'tempStormTop':
        print(npoints)

    # 保存图片.
    stem = f'{vartype}_binned_by_{binname}'
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
    dirpath_output = Path(
        config['dirpath_result'], 'Rr_profiles', 'binned_diff'
    )
    helper_tools.new_dir(dirpath_output, parents=True)

    coordnames = ['height', 'temp']
    vartypes = ['Rr', 'LH']
    for coordname, vartype in itertools.product(coordnames, vartypes):
        # 用地表降水率分组.
        draw_binned_differences(
            ds_dusty, ds_clean,
            coordname=coordname,
            vartype=vartype,
            binname='precipRateNearSurface',
            bins_stra=[0.2, 0.5, 1, 5],
            bins_conv=[1, 2, 5, 10],
            dirpath_output=dirpath_output
        )

        # 用雨顶温度分组.
        draw_binned_differences(
            ds_dusty, ds_clean,
            coordname=coordname,
            vartype=vartype,
            binname='tempStormTop',
            bins_stra=[0, -10, -20, -40],
            bins_conv=[0, -20, -30, -40],
            dirpath_output=dirpath_output
        )

        # 用PCT89分组.
        draw_binned_differences(
            ds_dusty, ds_clean,
            coordname=coordname,
            vartype=vartype,
            binname='pct89',
            bins_stra=[280, 270, 260, 240],
            bins_conv=[280, 260, 240, 220],
            dirpath_output=dirpath_output
        )
