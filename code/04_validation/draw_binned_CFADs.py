#----------------------------------------------------------------------------
# 2021/10/15
# 画出2014-2020年春季所有降水样本的CFAD图,并分别按月份和纬度分组.
#
# 画出两张图,每张的形状是(2, nbin).行代表雨型,列代表分组.
# 每一张代表不同的分组变量.
#----------------------------------------------------------------------------
import json
from pathlib import Path
import sys
sys.path.append('../modules')

import numpy as np
import xarray as xr
import cmaps
import matplotlib as mpl
import matplotlib.pyplot as plt

from profile_funcs import calc_cfad, Binner
from helper_funcs import letter_subplots

# 读取配置文件,作为全局变量使用.
with open('config.json', 'r') as f:
    config = json.load(f)

def draw_CFADs(
    ds, profile_varname,
    bin_varname, bins, bin_labels,
    coordname, output_filepath
):
    '''
    画出廓线数据分bin后的CFAD图.

    ds时数据集.
    profile_varname是廓线的变量名.这里仅为雷达反射率.
    bin_varname是用于分bin的变量名,bins是其数值的划分,
    bin_labels是每个bin对应的文字.
    coordname是廓线所处的垂直坐标,可以为高度或温度.
    output_filepath指定存储图片的路径.

    组图形状为(2, nbin),行表示两种雨型,列表示不同分bin.
    '''
    coord = ds[coordname].data
    nbin = len(bins) - 1

    # 设置画CFAD图的bins.
    xbins = np.linspace(10, 60, 51)
    if coordname == 'height':
        ybins = np.linspace(1.5, 12, 43)
    elif coordname == 'temp':
        ybins = np.linspace(-60, 10, 71)
    nx = len(xbins) - 1
    ny = len(ybins) - 1
    xc = (xbins[1:] + xbins[:-1]) / 2
    yc = (ybins[1:] + ybins[:-1]) / 2

    # 简化计算CFAD的函数.
    func = lambda arr: calc_cfad(
        arr, y=coord, xbins=xbins, ybins=ybins, norm='sum'
    ) * 100
    # 计算两种雨型按bin_var分bin的CFAD.
    Hs = np.zeros((2, nbin, ny, nx))
    for i in range(2):
        cond = ds.rainType.data == i + 1
        ds_temp = ds.isel(npoint=cond)
        profile_var = ds_temp[profile_varname].data
        bin_var = ds_temp[bin_varname].data
        # 利用Binner进行分组计算.
        b = Binner(bin_var, profile_var, bins, axis=0)
        result = b.apply(func)
        for j, res in enumerate(result):
            if res is not None:
                Hs[i, j, :, :] = res

    Rstrs = ['Stratiform', 'Convective']
    fig, axes = plt.subplots(nrows=2, ncols=nbin, figsize=(4 * nbin, 8))
    # 在每张子图上画出CFAD图.
    for i in range(2):
        for j in range(nbin):
            ax = axes[i, j]
            H = Hs[i, j, :, :]
            levels = np.linspace(0, H.max(), 100)
            im = ax.contourf(
                xc, yc, H, levels=levels,
                cmap=cmaps.WhBlGrYeRe, extend='both'
            )
            cbar = fig.colorbar(
                im, ax=ax, aspect=30,
                ticks=mpl.ticker.MaxNLocator(6),
                format=mpl.ticker.PercentFormatter(decimals=2)
            )
            cbar.ax.tick_params(labelsize='x-small')

    # 设置子图的左右小标题.
    for i, Rstr in enumerate(Rstrs):
        for j, bin_label in enumerate(bin_labels):
            axes[i, j].set_title(Rstr, loc='left', fontsize='small')
            axes[i, j].set_title(bin_label, loc='right', fontsize='small')

    # 根据coord设置y轴所需的tick.
    if coordname == 'height':
        ylabel = 'Height (km)'
        ylim = (0, 12)
        major_locator = mpl.ticker.MultipleLocator(2)
        minor_locator = mpl.ticker.MultipleLocator(1)
    elif coordname == 'temp':
        ylabel = 'Temperature (℃)'
        ylim = (20, -60)
        major_locator = mpl.ticker.MultipleLocator(20)
        minor_locator = mpl.ticker.MultipleLocator(10)

    # 为每个子图设置相同的ticks.
    for ax in axes.flat:
        ax.set_xlim(10, 60)
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_locator(major_locator)
        ax.yaxis.set_minor_locator(minor_locator)
        ax.tick_params(labelsize='x-small')

    # 在最下面标上xlabel.
    for ax in axes[1, :]:
        ax.set_xlabel('Reflectivity (dBZ)', fontsize='small')
    # 在最左边标上ylabel.
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, fontsize='small')

    # 为子图标出字母标识.
    letter_subplots(axes, (0.06, 0.96), 'small')

    # 保存图片.
    fig.savefig(str(output_filepath), dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    # 读取dusty cases和clean cases.
    input_dirpath = Path(config['temp_dirpath']) / 'merged'
    ds = xr.load_dataset(str(input_dirpath / 'all_profile.nc'))

    # 若输出目录不存在,那么新建.
    output_dirpath = Path(config['result_dirpath']) / 'CFADs'
    if not output_dirpath.exists():
        output_dirpath.mkdir()

    # 纬度的分bin.
    DPR_extent = config['DPR_extent']
    lonmin, lonmax, latmin, latmax = DPR_extent
    lat_bins = np.linspace(latmin, latmax, 4)
    lat_bin_labels = [f'R{i + 1}' for i in range(len(lat_bins) - 1)]

    # 月份的分bin.
    month_bins = [2.5, 3.5, 4.5, 5.5]
    month_bin_labels = ['March', 'April', 'May']

    # 用纬度进行分组.
    draw_CFADs(
        ds=ds,
        profile_varname='zFactorCorrected',
        bin_varname='lat',
        bins=lat_bins,
        bin_labels=lat_bin_labels,
        coordname='height',
        output_filepath=(output_dirpath / 'CFADs_binned_by_lat.png')
    )
    draw_CFADs(
        ds=ds,
        profile_varname='zFactorCorrected_t',
        bin_varname='lat',
        bins=lat_bins,
        bin_labels=lat_bin_labels,
        coordname='temp',
        output_filepath=(output_dirpath / 'CFADs_t_binned_by_lat.png')
    )
    # 用月份进行分组.
    draw_CFADs(
        ds=ds,
        profile_varname='zFactorCorrected',
        bin_varname='month',
        bins=month_bins,
        bin_labels=month_bin_labels,
        coordname='height',
        output_filepath=(output_dirpath / 'CFADs_binned_by_month.png')
    )
    draw_CFADs(
        ds=ds,
        profile_varname='zFactorCorrected_t',
        bin_varname='month',
        bins=month_bins,
        bin_labels=month_bin_labels,
        coordname='temp',
        output_filepath=(output_dirpath / 'CFADs_t_binned_by_month.png')
    )
