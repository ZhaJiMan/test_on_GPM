import json
from pathlib import Path
from collections import namedtuple
import sys
sys.path.append('../modules')

import numpy as np
import xarray as xr
from scipy.stats import mstats
import matplotlib as mpl
import matplotlib.pyplot as plt

from profile_funcs import Binner, smooth_profiles

# 读取配置文件,作为全局变量使用.
with open('config.json', 'r') as f:
    config = json.load(f)

def calc_profiles(ds1, ds2, bins, xvarname, yvarname, smooth=False):
    '''
    计算两组数据中廓线数据根据另一个变量分bin后的平均廓线.
    并将廓线数据汇总到数组中,以方便比较.

    返回温度坐标,廓线条数,平均廓线和标准误差廓线.
    '''
    nbin = len(bins) - 1
    hgt = ds1.height_r.data
    # 提前准备好数组.
    npoints = np.zeros((2, nbin), dtype=int)
    mean_profiles = np.ma.masked_all((2, nbin, len(hgt)))
    sem_profiles = np.ma.masked_all((2, nbin, len(hgt)))
    # 对两组Dataset进行循环.
    for i, ds in enumerate([ds1, ds2]):
        x = ds[xvarname].to_masked_array()
        y = ds[yvarname].to_masked_array()
        # 分bin计算平均廓线和标准误差廓线.
        b = Binner(x, y, bins, axis=0)
        npoints[i, :] = b.counts
        mean_result = b.apply(np.ma.mean, axis=0)
        sem_result = b.apply(mstats.sem, axis=0)
        for j in range(nbin):
            if mean_result[j] is not None:
                mean_profiles[i, j, :] = mean_result[j]
            if sem_result[j] is not None:
                sem_profiles[i, j, :] = sem_result[j]

    # 去掉10℃高度以下的数据.
    # mean_profiles = mean_profiles[:, :, temp <= 10]
    # sem_profiles = sem_profiles[:, :, temp <= 10]
    # temp = temp[temp <= 10]

    # 进行平滑.
    if smooth:
        sigma = 2
        mean_profiles = smooth_profiles(mean_profiles, sigma=sigma)
        sem_profiles = smooth_profiles(sem_profiles, sigma=sigma)

    # 用namedtuple存储结果.
    Data = namedtuple(
        'Data', ['hgt', 'npoints', 'mean_profiles', 'sem_profiles']
    )

    return Data(hgt, npoints, mean_profiles, sem_profiles)

def get_bin_labels(bins):
    '''得到每个bin对应的标签.'''
    nbin = len(bins) - 1
    labels = [f'{bins[i]} ~ {bins[i + 1]}' for i in range(nbin)]

    return labels

def draw_Rr_profiles(
    dusty_ds, clean_ds,
    bins_stra, bins_conv,
    xvarname, yvarname, title,
    output_filepath):
    '''
    画出按一维变量分bin平均后的降水速率廓线.

    dusty_ds和clean_ds是数据集.
    xvarname是用于分bin的变量名,yvarname是廓线变量名.
    bins_stra和bins_conv分别是用于两种雨型的bins.
    title标出廓线是被什么变量分组的.
    output_filepath指定存储图片的路径.

    组图形状为(2, mbin),其中mbin是两个bins中最大的bin数.
    行表示雨型,列表示分bin.每张子图中含有清洁与污染两组的平均廓线.

    因为允许bins_stra与bins_conv不等长,所以代码中需要分开处理
    组图的两行,可能会写得有些繁琐.
    '''
    nbin_stra = len(bins_stra) - 1
    nbin_conv = len(bins_conv) - 1
    mbin = max(nbin_stra, nbin_conv)

    # 计算两种雨型对应的平均廓线数据.
    hgt = dusty_ds.height_r.data
    data_stra = calc_profiles(
        ds1=dusty_ds.isel(npoint=(dusty_ds.rainType == 1)),
        ds2=clean_ds.isel(npoint=(clean_ds.rainType == 1)),
        bins=bins_stra,
        xvarname=xvarname, yvarname=yvarname,
    )
    data_conv = calc_profiles(
        ds1=dusty_ds.isel(npoint=(dusty_ds.rainType == 2)),
        ds2=clean_ds.isel(npoint=(clean_ds.rainType == 2)),
        bins=bins_conv,
        xvarname=xvarname, yvarname=yvarname,
    )

    # 画图用的参数.
    groups = ['Dusty', 'Clean']
    colors = ['C1', 'C0']
    labels_stra = get_bin_labels(bins_stra)
    labels_conv = get_bin_labels(bins_conv)
    # 组图形状为(2, mbin),允许两种雨型分bin的数量不一样,
    # 组图列数取最长的mbin.组图大小也会自动随列数发生变化.
    fig, axes = plt.subplots(2, mbin, figsize=(2 * mbin, 6))
    fig.subplots_adjust(wspace=0.3, hspace=0.25)

    # 画出平均廓线和标准误差的阴影,并标出廓线数目.
    data_list = [data_stra, data_conv]
    nbin_list = [nbin_stra, nbin_conv]
    for i in range(2):
        data = data_list[i]
        nbin = nbin_list[i]
        for j, ax in enumerate(axes[i, :nbin]):
            for k in range(2):
                color = colors[k]
                group = groups[k]
                ax.plot(
                    data.mean_profiles[k, j, :],
                    data.hgt,
                    lw=1, color=color,
                    label=f'{group} ({data.npoints[k, j]})'
                )
                ax.fill_betweenx(
                    data.hgt,
                    data.mean_profiles[k, j, :] - 1.96 * data.sem_profiles[k, j, :],
                    data.mean_profiles[k, j, :] + 1.96 * data.sem_profiles[k, j, :],
                    color=color, alpha=0.4
                )
            ax.legend(
                fontsize='xx-small', loc='upper right', handlelength=1.0
            )

    # 添加雨型信息和分bin信息.
    for label, ax in zip(labels_stra, axes[0, :nbin_stra]):
        ax.set_title('Stratiform', loc='left', fontsize='x-small')
        ax.set_title(label, loc='right', fontsize='x-small')
    for label, ax in zip(labels_conv, axes[1, :nbin_conv]):
        ax.set_title('Convective', loc='left', fontsize='x-small')
        ax.set_title(label, loc='right', fontsize='x-small')

    # 根据垂直廓线的变量类型,设定x轴范围和xlabel.
    if yvarname == 'precipRate_r':
        set_xlim = lambda ax: ax.set_xlim(0, 1.2 * ax.get_xlim()[1])
        xlabel = 'Rain Rate (mm/h)'
    elif yvarname == 'Dm_r':
        set_xlim = lambda ax: ax.set_xlim(None, 1.2 * ax.get_xlim()[1])
        xlabel = 'Dm (mm)'
    elif yvarname == 'Nw_r':
        set_xlim = lambda ax: ax.set_xlim(20, 40)
        xlabel = 'Nw (dBNw)'

    # 设置ticks.
    for ax in axes.flat:
        set_xlim(ax)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
        ax.set_ylim(-4, 10)
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
        ax.tick_params(labelsize='x-small')

    # 在组图边缘添加label.
    for ax in axes[1, :]:
        ax.set_xlabel(xlabel, fontsize='x-small')
    for ax in axes[:, 0]:
        ax.set_ylabel('Relative Height [km]', fontsize='x-small')

    # 隐藏多余的子图.
    for ax in axes[0, nbin_stra:]:
        ax.set_visible(False)
    for ax in axes[0, nbin_conv:]:
        ax.set_visible(False)

    # 给组图添加标题.
    fig.suptitle(title, y=0.95, fontsize='medium')

    # 保存图片.
    fig.savefig(str(output_filepath), dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    # 读取dusty cases和clean cases.
    input_dirpath = Path(config['temp_dirpath']) / 'merged'
    ds = xr.load_dataset(str(input_dirpath / 'all_profile_relative.nc'))
    DIVIDE_LAT = config['DIVIDE_LAT']
    cond = ds.month.isin([3, 4]) & (ds.lat >= DIVIDE_LAT)
    dusty_ds = ds.isel(npoint=cond)
    cond = (ds.month == 5) & (ds.lat < DIVIDE_LAT)
    clean_ds = ds.isel(npoint=cond)

    # 若输出目录不存在,那么新建.
    output_dirpath = Path(config['result_dirpath']) / 'Rr_profiles_relative_dusty'
    if not output_dirpath.exists():
        output_dirpath.mkdir()

    # 依次画出按地表降水,雨顶温度,和PCT89分组的Rr.
    draw_Rr_profiles(
        dusty_ds, clean_ds,
        bins_stra=[0.5, 1, 2, 5],
        bins_conv=[1, 5, 10, 20],
        xvarname='precipRateNearSurface',
        yvarname='precipRate_r',
        title='Rain Rate Grouped by Surface Rain Rate (mm/h)',
        output_filepath=output_dirpath / 'Rr_profiles_groupby_surfRr.png'
    )
    draw_Rr_profiles(
        dusty_ds, clean_ds,
        bins_stra=[0, -10, -20, -40],
        bins_conv=[0, -20, -40, -60],
        xvarname='tempStormTop',
        yvarname='precipRate_r',
        title='Rain Rate Grouped by Surface Temperature of Storm Top (K)',
        output_filepath=output_dirpath / 'Rr_profiles_groupby_st.png'
    )
    # 依次画出按地表降水,雨顶温度,和PCT89分组的Dm.
    draw_Rr_profiles(
        dusty_ds, clean_ds,
        bins_stra=[0.5, 1, 2, 5],
        bins_conv=[1, 5, 10, 20],
        xvarname='precipRateNearSurface',
        yvarname='Dm_r',
        title='Dm Grouped by Surface Rain Rate (mm/h)',
        output_filepath=output_dirpath / 'Dm_profiles_groupby_surfRr.png'
    )
    draw_Rr_profiles(
        dusty_ds, clean_ds,
        bins_stra=[0, -10, -20, -40],
        bins_conv=[0, -20, -40, -60],
        xvarname='tempStormTop',
        yvarname='Dm_r',
        title='Dm Grouped by Surface Temperature of Storm Top (K)',
        output_filepath=output_dirpath / 'Dm_profiles_groupby_st.png'
    )
    # 依次画出按地表降水,雨顶温度,和PCT89分组的Nw.
    draw_Rr_profiles(
        dusty_ds, clean_ds,
        bins_stra=[0.5, 1, 2, 5],
        bins_conv=[1, 5, 10, 20],
        xvarname='precipRateNearSurface',
        yvarname='Nw_r',
        title='Nw Grouped by Surface Rain Rate (mm/h)',
        output_filepath=output_dirpath / 'Nw_profiles_groupby_surfRr.png'
    )
    draw_Rr_profiles(
        dusty_ds, clean_ds,
        bins_stra=[0, -10, -20, -40],
        bins_conv=[0, -20, -40, -60],
        xvarname='tempStormTop',
        yvarname='Nw_r',
        title='Nw Grouped by Surface Temperature of Storm Top (K)',
        output_filepath=output_dirpath / 'Nw_profiles_groupby_st.png'
    )