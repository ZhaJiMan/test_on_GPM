#----------------------------------------------------------------------------
# 2021/07/08
# 画出温度坐标下分组的潜热平均廓线.
#
# 可以用来分组的变量有:
# - precipRateNearSurface
# - tempStormTop
# - PCT89
# - CAPE
#
# 可以计算的平均廓线:
# - SLH
# - VPH
#
# 组图形状为(2,mbin),行表示层云和对流两种雨型,列表示分组.
# 两种雨型的分bin可以不一致,mbin表示两种分bin的最长的bin数.
#----------------------------------------------------------------------------
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
from helper_funcs import letter_subplots

# 读取配置文件,作为全局变量使用.
with open('config.json', 'r') as f:
    config = json.load(f)

# 全局的平滑参数.
SMOOTH = True
SIGMA = 2

def calc_profiles(ds1, ds2, bins, xvarname, yvarname):
    '''
    计算两组数据中廓线数据根据另一个变量分bin后的平均廓线.
    并将廓线数据汇总到数组中,以方便比较.

    返回温度坐标,廓线条数,平均廓线和标准误差廓线.
    '''
    nbin = len(bins) - 1
    temp = ds1.temp.data
    # 提前准备好数组.
    npoints = np.zeros((2, nbin), dtype=int)
    mean_profiles = np.ma.masked_all((2, nbin, len(temp)))
    sem_profiles = np.ma.masked_all((2, nbin, len(temp)))
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
    mean_profiles = mean_profiles[:, :, temp <= 10]
    sem_profiles = sem_profiles[:, :, temp <= 10]
    temp = temp[temp <= 10]

    # 进行平滑.
    if SMOOTH:
        mean_profiles = smooth_profiles(mean_profiles, sigma=SIGMA)
        sem_profiles = smooth_profiles(sem_profiles, sigma=SIGMA)

    # 用namedtuple存储结果.
    Data = namedtuple(
        'Data', ['temp', 'npoints', 'mean_profiles', 'sem_profiles']
    )

    return Data(temp, npoints, mean_profiles, sem_profiles)

def get_bin_labels(bins):
    '''得到每个bin对应的标签.'''
    nbin = len(bins) - 1
    labels = [f'{bins[i]} ~ {bins[i + 1]}' for i in range(nbin)]

    return labels

def draw_LH_profiles(
    dusty_ds, clean_ds,
    bins_stra, bins_conv,
    xvarname, yvarname, title,
    output_filepath):
    '''
    画出按一维变量分bin平均后的潜热廓线.

    dusty_ds和clean_ds是数据集.
    xvarname是用于分bin的变量名,yvarname是廓线变量名.
    bins_stra和bins_conv分别是用于两种雨型的bins.
    title标出廓线是被什么变量分组的.
    output_filepath指定存储图片的路径.

    组图形状为(2, mbin),其中mbin是两个bins中最大的bin数.
    行表示雨型,列表示分bin.每张子图中含有清洁与污染两组的平均廓线.
    '''
    nbin_stra = len(bins_stra) - 1
    nbin_conv = len(bins_conv) - 1
    mbin = max(nbin_stra, nbin_conv)

    # 计算两种雨型对应的平均廓线数据.
    temp = dusty_ds.temp.data
    data_stra = calc_profiles(
        ds1=dusty_ds.isel(npoint=(dusty_ds.rainType == 1)),
        ds2=clean_ds.isel(npoint=(clean_ds.rainType == 1)),
        bins=bins_stra,
        xvarname=xvarname, yvarname=yvarname
    )
    data_conv = calc_profiles(
        ds1=dusty_ds.isel(npoint=(dusty_ds.rainType == 2)),
        ds2=clean_ds.isel(npoint=(clean_ds.rainType == 2)),
        bins=bins_conv,
        xvarname=xvarname, yvarname=yvarname
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
                    data.temp,
                    lw=1, color=color,
                    label=f'{group} ({data.npoints[k, j]})'
                )
                ax.fill_betweenx(
                    data.temp,
                    data.mean_profiles[k, j, :] - 1.96 * data.sem_profiles[k, j, :],
                    data.mean_profiles[k, j, :] + 1.96 * data.sem_profiles[k, j, :],
                    color=color, alpha=0.4
                )
                # 添加0潜热处的垂直辅助线.
                ax.axvline(0, color='k', ls='--', lw=0.6)
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

    # 设置ticks.
    for ax in axes.flat:
        xmax = np.max(np.abs(ax.get_xlim()))
        ax.set_xlim(-0.8 * xmax, 1.4 * xmax)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
        ax.set_ylim(20, -60)
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(20))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(10))
        ax.tick_params(labelsize='x-small')

    # 在组图边缘添加label.
    for ax in axes[1, :]:
        ax.set_xlabel('Latent Heat (K/h)', fontsize='x-small')
    for ax in axes[:, 0]:
        ax.set_ylabel('Temperature (℃)', fontsize='x-small')

    # 为子图标出字母标识.
    letter_subplots(axes, (0.08, 0.96), 'x-small')

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
    input_dirpath = Path(config['input_dirpath'])
    with open(str(input_dirpath / 'found_cases.json'), 'r') as f:
        records = json.load(f)
    dusty_ds = xr.load_dataset(records['dusty']['profile_filepath'])
    clean_ds = xr.load_dataset(records['clean']['profile_filepath'])

    # 若输出目录不存在,那么新建.
    output_dirpath = Path(config['result_dirpath']) / 'LH_profiles'
    if not output_dirpath.exists():
        output_dirpath.mkdir()

    # 依次画出按地表降水,雨顶温度,和PCT89分组的SLH.
    draw_LH_profiles(
        dusty_ds, clean_ds,
        bins_stra=[0.5, 1, 2, 5],
        bins_conv=[1, 5, 10, 20],
        xvarname='precipRateNearSurface',
        yvarname='SLH_t',
        title='SLH Grouped by Surface Rain Rate (mm/h)',
        output_filepath=output_dirpath / 'SLH_profiles_groupby_surfRr.png'
    )
    draw_LH_profiles(
        dusty_ds, clean_ds,
        bins_stra=[0, -10, -20, -40],
        bins_conv=[0, -20, -40, -60],
        xvarname='tempStormTop',
        yvarname='SLH_t',
        title='SLH Grouped by Temperature of Storm Top (℃)',
        output_filepath=output_dirpath / 'SLH_profiles_groupby_st.png'
    )
    draw_LH_profiles(
        dusty_ds, clean_ds,
        bins_stra=[280, 270, 260, 240],
        bins_conv=[280, 260, 240, 220],
        xvarname='PCT89',
        yvarname='SLH_t',
        title='SLH Grouped by PCT89 (K)',
        output_filepath=output_dirpath / 'SLH_profiles_groupby_pct.png'
    )
    # 依次画出按地表降水,雨顶温度,和PCT89分组的VPH.
    draw_LH_profiles(
        dusty_ds, clean_ds,
        bins_stra=[0.5, 1, 2, 5],
        bins_conv=[1, 5, 10, 20],
        xvarname='precipRateNearSurface',
        yvarname='VPH_t',
        title='VPH Grouped by Surface Rain Rate (mm/h)',
        output_filepath=output_dirpath / 'VPH_profiles_groupby_surfRr.png'
    )
    draw_LH_profiles(
        dusty_ds, clean_ds,
        bins_stra=[0, -10, -20, -40],
        bins_conv=[0, -20, -40, -60],
        xvarname='tempStormTop',
        yvarname='VPH_t',
        title='VPH Grouped by Temperature of Storm Top (℃)',
        output_filepath=output_dirpath / 'VPH_profiles_groupby_st.png'
    )
    draw_LH_profiles(
        dusty_ds, clean_ds,
        bins_stra=[280, 270, 260, 240],
        bins_conv=[280, 260, 240, 220],
        xvarname='PCT89',
        yvarname='VPH_t',
        title='VPH Grouped by PCT89 (K)',
        output_filepath=output_dirpath / 'VPH_profiles_groupby_pct.png'
    )
