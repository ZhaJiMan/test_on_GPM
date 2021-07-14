#----------------------------------------------------------------------------
# 2021/05/08
# 画出两组个例间两个变量之间关系的散点图,计算分段平均并做T检验.
#
# 画出的变量有:
# - 地表降水率-雨顶温度
# - 地表降水率-PCT89
# - 雨顶温度-PCT89
#
# 对一对变量画一张组图.
# 组图形状为(1, 2),列表示雨型.每张子图内画有不同污染分组,并且下方还有
# 分段平均T检验的结果.
#
# 注意:这一脚本调用的Binner经过改写,暂时无法使用.所以本脚本也处于测试中.
#----------------------------------------------------------------------------
import json
from pathlib import Path
import sys
sys.path.append('../modules')
from profile_funcs import Binner, binned_tval

import numpy as np
import xarray as xr
import scipy.stats as stats

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 读取配置文件,作为全局变量使用.
with open('config.json', 'r') as f:
    config = json.load(f)

def draw_scatter_plot(
    dusty_ds, clean_ds,
    xvarname, yvarname, bins_list,
    plt_kwargs, output_filepath):
    '''
    画出两个一维变量之间的散点图.

    dusty_ds与clean_ds是数据集.
    xvarname与yvarname是x变量与y变量的变量名.
    bins_list是层云降水和对流降水的针对x的bins.
    plt_kwargs存有label和tick设置.
    output_filepath指定存储图片的路径.

    组图形状为(1, 2),列表示雨型.每张子图中含有两组散点,以及分段平均值.
    errorbar与bar函数可能会对MaskedArray警告,但不影响结果.
    '''
    ds_list = [dusty_ds, clean_ds]
    # 用一系列object数组存储画图用的变量.
    # 第一维表示雨型,第二维表示污染分组.
    scatter_data = np.empty((2, 2), dtype=object)
    binned_data = np.empty((2, 2,), dtype=object)
    line_data = np.empty((2, 2), dtype=object)
    for i in range(2):
        bins = bins_list[i]
        for j, ds in enumerate(ds_list):
            # 根据雨型截取数据.
            ds = ds.isel(npoint=(ds.rainType == i + 1))
            xvar = ds[xvarname].data
            yvar = ds[yvarname].data
            # 根据xvar进行分组,并计算平均值与标准误.
            b = Binner(xvar, yvar, bins)
            avgs = b.mean()
            sems = b.sem()

            # 将这些量存储到数组中.
            scatter_data[i, j] = (xvar, yvar)
            binned_data[i, j] = b.data
            line_data[i, j] = (avgs, sems)

    # 计算污染组与清洁组之间的分组t value.
    tvals_list = []
    for i in range(2):
        binned_dusty = binned_data[i, 0]
        binned_clean = binned_data[i, 1]
        tvals = binned_tval(binned_dusty, binned_clean)
        tvals_list.append(tvals)

    # 用于画图的参数.
    Rtypes = ['Stratiform', 'Convective']
    groups = ['Dusty', 'Clean']
    scatter_colors = ['C3', 'C0']
    line_colors = ['r', 'b']

    # 画出散点图.
    fig, axes = plt.subplots(1, 2, figsize=(7, 5))
    fig.subplots_adjust(wspace=0.2)
    for i, ax in enumerate(axes):
        tvals = tvals_list[i]
        bins = bins_list[i]
        Rtype = Rtypes[i]

        # 利用bins的中点作为线图的x.
        x_line = (bins[1:] + bins[:-1]) / 2
        for j in range(2):
            xvar, yvar = scatter_data[i, j]
            avgs, sems = line_data[i, j]
            group = groups[j]
            scatter_color = scatter_colors[j]
            line_color = line_colors[j]

            # 先画出散点图.
            ax.plot(
                xvar, yvar, ls='', marker='.', ms=0.5, alpha=0.8,
                color=scatter_color, label=group
            )
            # 再画出分bin后的yp的平均值,加上误差棒.
            ax.errorbar(
                x_line, avgs, yerr=(1.96 * sems),
                color=line_color, lw=1, capsize=3
            )

        # 设定x与y轴.
        ax.legend(fontsize='x-small', markerscale=10, loc='upper right')
        ax.set_xlim(bins[0], bins[-1])
        ax.set_ylim(*plt_kwargs['ylim'])
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(6))
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(6))
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
        ax.tick_params(labelsize='x-small')
        ax.set_title(Rtype, fontsize='small')

        # 在ax下方再分出一个axb用于绘制t value.
        divider = make_axes_locatable(ax)
        axb = divider.append_axes('bottom', size=1, pad=0.3)

        # 绘制t value的柱状图,并标出显著性的虚线.
        width = (bins[1] - bins[0]) / 3
        axb.bar(x_line, tvals, width, fc='none', ec='k', lw=1)
        axb.axhline(1.96, lw=1, ls='--', c='k', label='95%')
        axb.legend(fontsize='x-small', loc='upper right')

        # 需要保证ax和axb的x轴完全一致.
        axb.set_xlim(bins[0], bins[-1])
        axb.set_ylim(0, max(4, 1.3 * tvals.max()))
        axb.set_xlabel(plt_kwargs['xlabel'], fontsize='x-small')
        axb.xaxis.set_major_locator(mpl.ticker.MaxNLocator(6))
        axb.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
        axb.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
        axb.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
        axb.tick_params(labelsize='x-small')

        # 在左边缘添加ylabel.
        if i == 0:
            ax.set_ylabel(plt_kwargs['ylabel'], fontsize='x-small')
            axb.set_ylabel('| t |', fontsize='x-small')

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
    output_dirpath = Path(config['result_dirpath']) / 'statistics'
    if not output_dirpath.exists():
        output_dirpath.mkdir()

    # 两个bins分别对应于层云降水和对流降水.
    bins_list = [
        np.linspace(0, 5, 8),
        np.linspace(0, 12, 8)
    ]
    # 画出地表降水率-雨顶温度图.
    draw_scatter_plot(
        dusty_ds, clean_ds,
        xvarname='precipRateNearSurface',
        yvarname='tempStormTop',
        bins_list=bins_list,
        plt_kwargs={
            'ylim': (15, -60),
            'xlabel': 'Surface Rain Rate (mm/h)',
            'ylabel': 'Temperature of Storm Top (℃)'
        },
        output_filepath=output_dirpath / 'surfRr_st_scatter.png'
    )
    # 画出地表降水率-Tb89H图.
    draw_scatter_plot(
        dusty_ds, clean_ds,
        xvarname='precipRateNearSurface',
        yvarname='PCT89',
        bins_list=bins_list,
        plt_kwargs={
            'ylim': (180, 300),
            'xlabel': 'Surface Rain Rate (mm/h)',
            'ylabel': 'PCT89 (K)'
        },
        output_filepath=output_dirpath / 'surfRr_PCT89_scatter.png'
    )

    # 两个bins分别对应于层云降水和对流降水.
    bins_list = [
        np.linspace(10, -50, 8),
        np.linspace(0, -60, 8)
    ]
    # 画出雨顶温度-PCT89图.
    draw_scatter_plot(
        dusty_ds, clean_ds,
        xvarname='tempStormTop',
        yvarname='PCT89',
        bins_list=bins_list,
        plt_kwargs={
            'ylim': (180, 300),
            'xlabel': 'Temperature of Storm Top (℃)',
            'ylabel': 'PCT89 (K)'
        },
        output_filepath=output_dirpath / 'st_PCT89_scatter.png'
    )
