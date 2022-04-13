'''
2022-04-12
画出两组数据集两种雨型分bin平均的廓线.
组图形状为(2, nbin), 行表示两种雨型, 列表示nbin个分bin, 每张子图里画出
污染组和清洁组的平均廓线.

支持的廓线变量:
- precipRate, Nw, Dm
- CSH, SLH, VPH

支持的分bin变量:
- precipRateNearSurface
- tempStormTop
- PCT89

支持的坐标:
- 高度坐标
- 温度坐标
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
from draw_all_mean_profiles import set_LH_xlim

# 读取配置文件, 将config作为全局变量.
with open('config.json') as f:
    config = json.load(f)

def get_bin_labels(bins):
    '''生成每个bin的范围对应的标签.'''
    nbin = len(bins) - 1
    labels = []
    for i in range(nbin):
        if bins[i] < bins[i + 1]:
            label = f'({bins[i]}, {bins[i + 1]}]'
        else:
            label = f'({bins[i + 1]}, {bins[i]}]'
        labels.append(label)

    return labels

def draw_binned_profiles(
    ds_dusty, ds_clean,
    coordname, varname, binname,
    bins_stra, bins_conv,
    dirpath_output
):
    '''
    画出两组数据集按两种雨型分类, 再进行分bin平均的廓线.

    coordname指定垂直坐标, 可选高度和温度.
    varname指定廓线变量, 可选降水率, DSD和三种潜热.
    binname指定分bin的变量, 可选地表降水率, 雨顶温度和PCT89.
    bins_stra和bins_conv分别给出两种雨型分bin的方案, 要求二者等长.
    '''
    # 检查bins长度.
    if len(bins_stra) != len(bins_conv):
        raise ValueError('要求bins_stra和bins_conv长度相同')
    nbin = len(bins_stra) - 1

    # 决定垂直坐标.
    coord = ds_dusty[coordname].values
    if coordname == 'temp':
        varname += '_t'

    # 用means存储平均廓线, 用sems存储标准误差廓线.
    # 第一维表示两种雨型, 第二维表示bin的分组, 第三位表示污染分组.
    nh = len(coord)
    means = np.zeros((2, nbin, 2, nh))
    sems = np.zeros((2, nbin, 2, nh))
    npoints = np.zeros((2, nbin, 2), dtype=int)
    for k, ds in enumerate([ds_dusty, ds_clean]):
        for i, bins in enumerate([bins_stra, bins_conv]):
            ds_temp = ds.isel(npoint=(ds.rainType == i + 1))
            yvar = ds_temp[varname].values
            xvar = ds_temp[binname].values
            binner = profile_tools.ProfileBinner(xvar, yvar, bins)
            npoints[i, :, k] = binner.counts
            means[i, :, k, :] = binner.mean()
            sems[i, :, k, :] = binner.sem()
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
    varname = varname.strip('_t')
    if varname == 'precipRate':
        xlabel = 'Rain Rate (mm/hr)'
        set_xlim = lambda ax: ax.set_xlim(0, None)
    elif varname == 'Nw':
        xlabel = 'Nw (dBNw)'
        set_xlim = lambda ax: ax.set_xlim(20, 40)
    elif varname == 'Dm':
        xlabel = 'Dm (mm)'
        set_xlim = lambda ax: ax.set_xlim(None, None)
    elif varname == 'csh':
        xlabel = 'CSH LH (K/hr)'
        set_xlim = set_LH_xlim
    elif varname == 'slh':
        xlabel = 'SLH LH (K/hr)'
        set_xlim = set_LH_xlim
    elif varname == 'vph':
        xlabel = 'VPH LH (K/hr)'
        set_xlim = set_LH_xlim

    # 根据bin变量决定标题.
    if binname == 'precipRateNearSurface':
        longname = 'Surface Rain Rate'
        units = 'mm/hr'
    elif binname == 'heightStormTop':
        longname == 'Height of Storm Top'
        units = 'km'
    elif binname == 'tempStormTop':
        longname = 'Temperature of Storm Top'
        units = '℃'
    elif binname == 'pct89':
        longname = 'PCT89'
        units = 'K'
    title = f'Binned by {longname}'

    # 画图用的参数.
    Rtypes = ['Stra.', 'Conv.']
    groups = ['Dusty', 'Clean']
    colors = ['C1', 'C0']
    bin_labels_list = [
        get_bin_labels(bins_stra),
        get_bin_labels(bins_conv)
    ]

    # 组图形状为(2, nbin).
    # 行表示两种雨型, 列表示分bin, 每个子图中画出两组廓线.
    fig, axes = plt.subplots(2, nbin, figsize=(3 * nbin, 6))
    fig.subplots_adjust(wspace=0.3, hspace=0.25)
    for (i, j), ax in np.ndenumerate(axes):
        for k in range(2):
            mean = means[i, j, k, :]
            sem = sems[i, j, k, :]
            ax.plot(
                mean, coord, lw=1, color=colors[k],
                label=f'{groups[k]} (N={npoints[i, j, k]})'
            )
            ax.fill_betweenx(
                coord,
                mean - 1.96 * sem,
                mean + 1.96 * sem,
                color=colors[k], alpha=0.4
            )
        ax.legend(fontsize='xx-small', loc='upper right', handlelength=1.0)
        # 标出小标题.
        bin_labels = bin_labels_list[i]
        ax.set_title(Rtypes[i], loc='left', fontsize='x-small')
        ax.set_title(
            f'{bin_labels[j]} ({units})',
            loc='right', fontsize='x-small'
        )

    # 设置坐标轴.
    for ax in axes.flat:
        set_xlim(ax)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.set_ylim(*ylims)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(ybase))
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.tick_params(labelsize='x-small')

    # 在组图边缘设置标签.
    for ax in axes[1, :]:
        ax.set_xlabel(xlabel, fontsize='x-small')
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, fontsize='x-small')

    # 潜热变量加上x=0的辅助线.
    if varname in ['csh', 'slh', 'vph']:
        for ax in axes.flat:
            ax.axvline(0, color='k', ls='--', lw=0.6)

    # 给组图添加标题.
    fig.suptitle(title, y=0.95, fontsize='medium')
    # 为子图标出字母标识.
    plot_tools.letter_axes(axes, 0.06, 0.95, fontsize='x-small')

    # 保存图片.
    stem = f'{varname}_binned_by_{binname}'
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
        config['dirpath_result'], 'Rr_profiles', 'binned_mean'
    )
    helper_tools.new_dir(dirpath_output, parents=True)

    coordnames = ['height', 'temp']
    varanames = ['precipRate', 'Nw', 'Dm', 'csh', 'slh', 'vph']

    for coordname, varname in itertools.product(coordnames, varanames):
        # 用地表降水率分组.
        draw_binned_profiles(
            ds_dusty, ds_clean,
            coordname=coordname,
            varname=varname,
            binname='precipRateNearSurface',
            bins_stra=[0.2, 0.5, 1, 5],
            bins_conv=[1, 2, 5, 10],
            dirpath_output=dirpath_output
        )

        # 用雨顶温度分组.
        draw_binned_profiles(
            ds_dusty, ds_clean,
            coordname=coordname,
            varname=varname,
            binname='tempStormTop',
            bins_stra=[0, -10, -20, -40],
            bins_conv=[0, -20, -30, -40],
            dirpath_output=dirpath_output
        )

        # 用PCT89分组.
        draw_binned_profiles(
            ds_dusty, ds_clean,
            coordname=coordname,
            varname=varname,
            binname='pct89',
            bins_stra=[280, 270, 260, 240],
            bins_conv=[280, 260, 240, 220],
            dirpath_output=dirpath_output
        )
