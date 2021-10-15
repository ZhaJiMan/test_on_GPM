#----------------------------------------------------------------------------
# 2021/10/15
# 画出2014-2020年春季所有降水样本的平均降水速率廓线,并分别按月份和纬度分组.
#
# 图的形状为(1, 2),列表示雨型.每张子图内画有不同bin的平均廓线.
# 每张图可以指定不同的廓线变量,分bin变量和垂直坐标.
#----------------------------------------------------------------------------
import json
from pathlib import Path
import sys
sys.path.append('../modules')

import numpy as np
import xarray as xr
from scipy.stats import mstats
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from profile_funcs import Binner
from helper_funcs import letter_subplots

# 读取配置文件,作为全局变量使用.
with open('config.json', 'r') as f:
    config = json.load(f)

def draw_profiles(
    ds, profile_varname,
    bin_varname, bins, bin_labels,
    coordname, output_filepath
):
    '''
    画出廓线数据分bin后的平均廓线.

    ds时数据集.
    profile_varname是廓线的变量名.
    bin_varname是用于分bin的变量名,bins是其数值的划分,
    bin_labels是每个bin对应的文字.
    coordname是廓线所处的垂直坐标,可以为高度或温度.
    output_filepath指定存储图片的路径.

    组图形状为(1, 2),列表示两种雨型.
    每张子图中含有分bin平均的廓线.
    '''
    coord = ds[coordname].data
    nbin = len(bins) - 1
    ncoord = len(coord)

    # 计算两种雨型在不同bin中的平均廓线和标准误差廓线.
    npoints = np.zeros((2, nbin), dtype=int)    # bin中的廓线数量.
    avgs = np.ma.masked_all((2, nbin, ncoord))
    sems = np.ma.masked_all((2, nbin, ncoord))
    for i in range(2):
        cond = ds.rainType.data == i + 1
        ds_temp = ds.isel(npoint=cond)
        profile_var = ds_temp[profile_varname].to_masked_array()
        bin_var = ds_temp[bin_varname].to_masked_array()
        # 利用Binner进行分组计算.
        b = Binner(bin_var, profile_var, bins, axis=0)
        npoints[i, :] = b.counts
        avg_result = b.apply(np.ma.mean, axis=0)
        sem_result = b.apply(mstats.sem, axis=0)
        for j in range(nbin):
            if avg_result[j] is not None:
                avgs[i, j, :] = avg_result[j]
            if sem_result[j] is not None:
                sems[i, j, :] = sem_result[j]

    # 去掉1.5km或10℃高度以下的数据.
    cond = coord >= 1.5 if coordname == 'height' else coord <= 10
    avgs = avgs[:, :, cond]
    sems = sems[:, :, cond]
    coord = coord[cond]

    colors = [f'C{i}' for i in range(nbin)]
    titles = ['Stratiform', 'Convective']
    fig, axes = plt.subplots(1, 2, figsize=(4, 3))
    fig.subplots_adjust(wspace=0.25)

    # 在每张子图中画出平均廓线和标准误的阴影.
    for i, ax in enumerate(axes.flat):
        for j in range(nbin):
            ax.plot(
                avgs[i, j, :], coord, lw=1, c=colors[j],
                label=f'{bin_labels[j]} ({npoints[i, j]})'
            )
            ax.fill_betweenx(
                coord,
                avgs[i, j, :] - 1.96 * sems[i, j, :],
                avgs[i, j, :] + 1.96 * sems[i, j, :],
                color=colors[j], alpha=0.4
            )
        ax.legend(fontsize='xx-small', loc='upper right', handlelength=1.0)
        ax.set_title(titles[i], fontsize='x-small')

    # 设置坐标轴的label和ticks.
    # 根据提供的变量不同设置也会不同.
    for ax in axes.flat:
        # 设置x轴.
        if profile_varname in ['precipRate', 'precipRate_t']:
            ax.set_xlabel('Rain Rate (mm/h)', fontsize='x-small')
            ax.set_xlim(0, 1.2 * ax.get_xlim()[1])
        elif profile_varname in ['Dm', 'Dm_t']:
            ax.set_xlabel('Dm (mm)', fontsize='x-small')
            ax.set_xlim(None, 1.2 * ax.get_xlim()[1])
        elif profile_varname in ['Nw', 'Nw_t']:
            ax.set_xlabel('Nw (dBNw)', fontsize='x-small')
            ax.set_xlim(20, 40)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        # 设置y轴.
        if coordname == 'height':
            ax.set_ylabel('Height (km)', fontsize='x-small')
            ax.set_ylim(0, 12)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(2))
            ax.yaxis.set_minor_locator(mticker.MultipleLocator(1))
        elif coordname == 'temp':
            ax.set_ylabel('Temperature (℃)', fontsize='x-small')
            ax.set_ylim(20, -60)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
            ax.yaxis.set_minor_locator(mticker.MultipleLocator(10))
        ax.tick_params(labelsize='x-small')
    # 去除多余的ylabel.
    axes[1].set_ylabel(None)

    # 为子图标出字母标识.
    letter_subplots(axes, (0.08, 0.96), 'x-small')

    # 保存图片.
    fig.savefig(str(output_filepath), dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    # 读取dusty cases和clean cases.
    input_dirpath = Path(config['temp_dirpath']) / 'merged'
    ds = xr.load_dataset(str(input_dirpath / 'all_profile.nc'))

    # 若输出目录不存在,那么新建.
    output_dirpath = Path(config['result_dirpath']) / 'profiles'
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
    draw_profiles(
        ds=ds,
        profile_varname='precipRate',
        bin_varname='lat',
        bins=lat_bins,
        bin_labels=lat_bin_labels,
        coordname='height',
        output_filepath=(output_dirpath / 'precipRate_binned_by_lat.png')
    )
    draw_profiles(
        ds=ds,
        profile_varname='precipRate_t',
        bin_varname='lat',
        bins=lat_bins,
        bin_labels=lat_bin_labels,
        coordname='temp',
        output_filepath=(output_dirpath / 'precipRate_t_binned_by_lat.png')
    )
    # 用月份进行分组.
    draw_profiles(
        ds=ds,
        profile_varname='precipRate',
        bin_varname='month',
        bins=month_bins,
        bin_labels=month_bin_labels,
        coordname='height',
        output_filepath = (output_dirpath / 'precipRate_binned_by_month.png')
    )
    draw_profiles(
        ds=ds,
        profile_varname='precipRate_t',
        bin_varname='month',
        bins=month_bins,
        bin_labels=month_bin_labels,
        coordname='temp',
        output_filepath = (
            output_dirpath / 'precipRate_t_binned_by_month.png'
        )
    )
