#----------------------------------------------------------------------------
# 2021/07/08
# 画出可能会用CAPE分组的图像.
#
# 实际画了:
# - CAPE的一维PDF分布.
# - CAPE与地表降水率和雨顶温度之间的散点关系图.
# - CAPE分组的降水廓线和潜热廓线.
#
# 画图的函数是从其它脚本中导入的.
#----------------------------------------------------------------------------
import json
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.stats import linregress
import matplotlib as mpl
import matplotlib.pyplot as plt

from draw_hist1D import hist1D
from draw_binned_Rr_profiles import draw_Rr_profiles
from draw_binned_LH_profiles import draw_LH_profiles

# 读取配置文件,作为全局变量使用.
with open('config.json', 'r') as f:
    config = json.load(f)

def draw_hist1D(dusty_ds, clean_ds, output_filepath):
    '''画出两组数组的两种雨型的CAPE的一维hist图.'''
    # 设置用于计算hist的bins.
    bins = np.linspace(0, 420, 41)
    nbin = len(bins) - 1

    # 用hs存储计算出的histogram.
    # 第一维表示两种雨型,第二维表示污染分组.
    hs = np.zeros((2, 2, nbin))
    for j, ds in enumerate([dusty_ds, clean_ds]):
        for i in range(2):
            cape = ds.cape.isel(npoint=(ds.rainType == i + 1)).data
            hs[i, j, :] = hist1D(cape, bins)

    # 画图用的参数.
    Rstrs = ['Stratiform', 'Convective']
    groups = ['Dusty', 'Clean']
    colors = ['C1', 'C0']

    # 组图形状为(1, 2).
    # 列表示两种雨型,每张子图中画有两组的histogram.
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.subplots_adjust(wspace=0.15)
    for i, ax in enumerate(axes):
        for j in range(2):
            color = colors[j]
            group = groups[j]
            h = hs[i, j, :]
            # 因为h经过处理,所以这里用bar手动画出histogram.
            ax.bar(
                bins[:-1], h, width=np.diff(bins),
                align='edge', alpha=0.5, color=color, label=group
            )
        # 用legend标出分组.
        ax.legend(fontsize='small', loc='upper right')

    # 为每张子图添加雨型标题.
    for Rstr, ax in zip(Rstrs, axes):
        ax.set_title(Rstr, fontsize='medium')

    # 设置子图的ticks.
    for ax in axes:
        ax.set_xlim(bins.min(), bins.max())
        ax.set_xlabel('CAPE (J/kg)', fontsize='small')
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(8))
        ax.set_ylim(0, None)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(8))
        ax.tick_params(labelsize='small')
    # 在最左边设置ylabel.
    axes[0].set_ylabel('PDF (%)', fontsize='small')

    fig.savefig(str(output_filepath), dpi=300, bbox_inches='tight')
    plt.close(fig)

def draw_scatter(dusty_ds, clean_ds, output_filepath):
    '''
    画出两组数据中地表降水率-CAPE,雨顶温度-CAPE的散点关系图.
    注意没有区分雨型.
    '''
    # 组图形状为(2, 2),行表示不同变量组合,列表示污染分组.
    # 每张子图中画出变量散点与线性拟合的结果.
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))
    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    for j, ds in enumerate([dusty_ds, clean_ds]):
        cape = ds.cape.data
        surfRr = ds.precipRateNearSurface.data
        st = ds.tempStormTop.data

        # 只选取地表降水率大于0,CAPE大于0的数据,同时利用分位数去掉异常高的值.
        mask = \
            (surfRr > 0) & (surfRr <= np.quantile(surfRr, 0.95)) & \
            (cape > 0) & (cape <= np.quantile(cape, 0.95))
        cape = cape[mask]
        surfRr = surfRr[mask]
        st = st[mask]

        # 在第一行画出地表降水率-CAPE的散点图.
        axes[0, j].plot(surfRr, cape, ls='', marker='.', ms=0.5, c='k')
        result = linregress(surfRr, cape)
        axes[0, j].plot(
            surfRr, result.slope * surfRr + result.intercept,
            lw=1, c='r', label='$R^2=$' + f'{result.rvalue**2:.2f}'
        )
        axes[0, j].legend(loc='upper right', fontsize='small')

        # 在第一行画出雨顶温度-CAPE的散点图.
        axes[1, j].plot(st, cape, ls='', marker='.', ms=0.5, c='k')
        result = linregress(st, cape)
        axes[1, j].plot(
            st, result.slope * st + result.intercept,
            lw=1, c='r', label='$R^2=$' + f'{result.rvalue**2:.2f}'
        )
        axes[1, j].legend(loc='upper right', fontsize='small')

    # 分别设置每张子图的ticks.
    for ax in axes[:, 0]:
        ax.set_ylabel('CAPE (J/kg)', fontsize='small')
    for ax in axes[0, :]:
        ax.set_xlim(0, None)
        ax.set_xlabel('Surface Rain Rate (mm/h)', fontsize='small')
    for ax in axes[1, :]:
        ax.set_xlim(10, -60)
        ax.set_xlabel('Temperature of Storm Top (℃)', fontsize='small')
    for ax in axes.flat:
        ax.set_ylim(0, None)
        ax.tick_params(labelsize='small')

    # 在每一列最上面标出污染分组.
    axes[0, 0].set_title('Dusty', y=1.05, fontsize='large')
    axes[0, 1].set_title('Clean', y=1.05, fontsize='large')

    fig.savefig(str(output_filepath), dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    input_dirpath = Path(config['input_dirpath'])
    with open(str(input_dirpath / 'found_cases.json'), 'r') as f:
        records = json.load(f)
    dusty_ds = xr.load_dataset(records['dusty']['profile_filepath'])
    clean_ds = xr.load_dataset(records['clean']['profile_filepath'])

    # 若输出目录不存在,则创建.
    output_dirpath = Path(config['result_dirpath']) / 'CAPE'
    if not output_dirpath.exists():
        output_dirpath.mkdir()

    output_filepath = output_dirpath / 'test.png'

    # 画出CAPE的一维PDF分布.
    draw_hist1D(dusty_ds, clean_ds, output_dirpath / 'hist1D.png')
    # 画出CAPE与其它变量的散点关系.
    draw_scatter(dusty_ds, clean_ds, output_dirpath / 'scatter.png')

    # 画出CAPE分组的Rr,Nw,和Dm廓线.
    draw_Rr_profiles(
        dusty_ds, clean_ds,
        bins_stra=[0, 10, 50, 100],
        bins_conv=[0, 100, 200, 300],
        xvarname='cape',
        yvarname='precipRate_t',
        title='Rain Rate Grouped by CAPE (J/kg)',
        output_filepath=output_dirpath / 'Rr_profiles_groupby_cape.png'
    )
    draw_Rr_profiles(
        dusty_ds, clean_ds,
        bins_stra=[0, 10, 50, 100],
        bins_conv=[0, 100, 200, 300],
        xvarname='cape',
        yvarname='Nw_t',
        title='Nw Grouped by CAPE (J/kg)',
        output_filepath=output_dirpath / 'Nw_profiles_groupby_cape.png'
    )
    draw_Rr_profiles(
        dusty_ds, clean_ds,
        bins_stra=[0, 10, 50, 100],
        bins_conv=[0, 100, 200, 300],
        xvarname='cape',
        yvarname='Dm_t',
        title='Dm Grouped by CAPE (J/kg)',
        output_filepath=output_dirpath / 'Dm_profiles_groupby_cape.png'
    )
    # 画出CAPE分组的SLH和VPH廓线.
    draw_LH_profiles(
        dusty_ds, clean_ds,
        bins_stra=[0, 10, 50, 100],
        bins_conv=[0, 100, 200, 300],
        xvarname='cape',
        yvarname='SLH_t',
        title='SLH Grouped by CAPE (J/kg)',
        output_filepath=output_dirpath / 'SLH_profiles_groupby_cape.png'
    )
    draw_LH_profiles(
        dusty_ds, clean_ds,
        bins_stra=[0, 10, 50, 100],
        bins_conv=[0, 100, 200, 300],
        xvarname='cape',
        yvarname='VPH_t',
        title='VPH Grouped by CAPE (J/kg)',
        output_filepath=output_dirpath / 'VPH_profiles_groupby_cape.png'
    )
