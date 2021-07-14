#----------------------------------------------------------------------------
# 2021/07/08
# 画出两组个例的两种雨型下一些一维变量的histogram(PDF)分布图.
#
# 画出的变量有:
# - 地表降水率
# - 雨顶温度
# - PCT89
# 其中地表降水率采用对数坐标表示.
#
# 组图形状为(2, nvar),行表示雨型,列表示不同变量,每张子图内画有不同污染分组的
# 变量的分布.
#----------------------------------------------------------------------------
import json
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.special import exp10

import matplotlib as mpl
import matplotlib.pyplot as plt

# 读取配置文件,作为全局变量使用.
with open('config.json', 'r') as f:
    config = json.load(f)

def hist1D(x, bins):
    '''
    计算一维histogram,并用总和进行归一化.

    要求bins单调递增.
    '''
    h = np.histogram(x, bins)[0]
    s = h.sum()
    if s > 0:
        h = h / s * 100

    return h

if __name__ == '__main__':
    input_dirpath = Path(config['input_dirpath'])
    with open(str(input_dirpath / 'found_cases.json'), 'r') as f:
        records = json.load(f)
    dusty_ds = xr.load_dataset(records['dusty']['profile_filepath'])
    clean_ds = xr.load_dataset(records['clean']['profile_filepath'])
    ds_list = [dusty_ds, clean_ds]

    # 若输出目录不存在,则创建.
    output_dirpath = Path(config['result_dirpath']) / 'statistics'
    if not output_dirpath.exists():
        output_dirpath.mkdir()

    # 用data存储计算出的histogram.
    # 第一维表示两种雨型,第二维表示三个变量,第三维表示污染分组.
    data = np.empty((2, 3, 2), dtype=object)
    for k, ds in enumerate(ds_list):
        rainType = ds.rainType.data
        for i in range(2):
            # 根据雨型进行筛选.
            flag = rainType == (i + 1)
            var1 = ds.precipRateNearSurface.data[flag]
            var2 = ds.tempStormTop.data[flag]
            var3 = ds.PCT89.data[flag]

            # 去掉降水率为0的点.
            var1 = var1[var1 > 0]

            # 计算histogram.
            bins1 = np.linspace(-2, 2, 40)
            h1 = hist1D(np.log10(var1), bins1)
            bins2 = np.linspace(-60, 20, 40)
            h2 = hist1D(var2, bins2)
            bins3 = np.linspace(150, 300, 40)
            h3 = hist1D(var3, bins3)

            # 存储到data中.
            data[i, 0, k] = (exp10(bins1), h1)
            data[i, 1, k] = (bins2, h2)
            data[i, 2, k] = (bins3, h3)

    # 画图用的参数.
    Rstrs = ['Stratiform', 'Convective']
    groups = ['Dusty', 'Clean']
    colors = ['C1', 'C0']

    # 组图形状为(2, 3).
    # 行表示两种雨型,列表示三种变量,每张子图中画有两组的histogram.
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    # 在每张子图里画出两组histogram.
    for i in range(2):
        for j in range(3):
            for k in range(2):
                group = groups[k]
                color = colors[k]
                bins, h = data[i, j, k]
                # 因为h经过处理,所以这里用bar手动画出histogram.
                axes[i, j].bar(
                    bins[:-1], h, width=np.diff(bins),
                    align='edge', alpha=0.5, color=color, label=group
                )
            # 用legend标出分组.
            axes[i, j].legend(fontsize='x-small', loc='upper right')

    # 设置第一列的xticks.
    for ax in axes[:, 0]:
        ax.set_xscale('log')
        ax.set_xlim(1E-2, 1E2)
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    # 设置第二列的xticks.
    for ax in axes[:, 1]:
        ax.set_xlim(20, -60)
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(10))
    # 设置第三列的xticks.
    for ax in axes[:, 2]:
        ax.set_xlim(150, 300)
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(30))
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(15))
    # 为每个ax设置相同的yticks和labelsize.
    for ax in axes.flat:
        ax.set_ylim(0, 1.2 * ax.get_ylim()[1])
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(6))
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
        ax.tick_params(labelsize='x-small')

    # 在最下面标出xlabel.
    axes[1, 0].set_xlabel('Rain Rate (mm/h)', fontsize='small')
    axes[1, 1].set_xlabel('Temperature of Storm Top (℃)', fontsize='small')
    axes[1, 2].set_xlabel('PCT89 (K)', fontsize='small')
    # 在最左边标出ylabel.
    for ax in axes[:, 0]:
        ax.set_ylabel('PDF (%)', fontsize='small')

    # 标出雨型.
    for i in range(2):
        Rstr = Rstrs[i]
        for ax in axes[i, :]:
            ax.set_title(Rstr, loc='left', fontsize='small')

    # 保存图片.
    output_filepath = output_dirpath / 'hist1D.png'
    fig.savefig(str(output_filepath), dpi=300, bbox_inches='tight')
    plt.close(fig)
