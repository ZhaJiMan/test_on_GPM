#----------------------------------------------------------------------------
# 2021/07/08
# 画出两组个例的两种雨型下一些一维变量间构成的二维histogram(PDF)分布图.
#
# 画出的变量有:
# - 地表降水率-雨顶温度
# - 地表降水率-PCT89
# - 雨顶温度-PCT89
# 其中地表降水率采用对数坐标表示.
#
# 画出两张组图,分别代表两种雨型.
# 组图形状为(nvar, 3),行表示变量组合,列表示污染,清洁,污染-清洁差值.
#----------------------------------------------------------------------------
import json
from pathlib import Path
import sys
sys.path.append('../modules')
from profile_funcs import Binner

import numpy as np
import xarray as xr
from scipy.special import exp10

import matplotlib as mpl
import matplotlib.pyplot as plt
import cmaps

# 读取配置文件,作为全局变量使用.
with open('config.json', 'r') as f:
    config = json.load(f)

def hist2D(x, y, bins):
    '''
    统计x与y构成的二维histogram.

    要求bins中的数组都单调递增.
    返回xbins与ybins构成的网格,以及百分比为单位的histogram.
    '''
    H = np.histogram2d(x, y, bins)[0].T
    # 若H的总计数大于0,那么将H换算为百分比.
    s = H.sum()
    if s > 0:
        H = H / s * 100
    # X和Y是画图用的网格.
    X, Y = np.meshgrid(*bins)

    return X, Y, H

def add_colorbar(ax, im):
    '''为一个pcolor图添加colorbar.'''
    cbar = plt.colorbar(
        im, ax=ax, extend='both', pad=0.03,
        ticks=mpl.ticker.MaxNLocator(6),
        format=mpl.ticker.PercentFormatter()
    )
    cbar.ax.tick_params(labelsize='x-small')

def add_pcolor_normal(ax, X, Y, Z):
    '''绘制普通的pcolor图.'''
    im = ax.pcolormesh(X, Y, Z, cmap=cmaps.WhBlGrYeRe)
    add_colorbar(ax, im)

def add_pcolor_diff(ax, X, Y, Z):
    '''绘制作差后得到的正负对称的pcolor图.'''
    zmax = 0.95 * np.abs(Z).max()
    im = ax.pcolormesh(X, Y, Z, cmap=cmaps.BlWhRe, vmin=-zmax, vmax=zmax)
    add_colorbar(ax, im)

def draw_plot(ds_list, Rtype, output_filepath):
    '''
    画出一种雨型的二维histogram图像.

    组图形状为3*3,每一行表示不同的变量组合,
    前两列表示污染分组,第三列是两组的差值.
    地表降水率使用对数坐标处理.
    '''
    if Rtype == 'stra':
        Rnum = 1
        Rstr = 'Stratiform Rains'
    elif Rtype == 'conv':
        Rnum = 2
        Rstr = 'Convective Rains'

    # data用于存储作图的histogram.
    # 第一维表示三种变量组合.
    # 第二维表示dusty组,clean组,以及dusty-clean的差值.
    data = np.empty((3, 3), dtype=object)
    for j, ds in enumerate(ds_list):
        # 要求地表降水率大于0,且雨顶温度不缺测.
        cond = \
            (ds.rainType == Rnum) & \
            (ds.precipRateNearSurface > 0) & \
            (~ds.tempStormTop.isnull())
        ds = ds.isel(npoint=cond)

        # 这里降水使用对数值.
        Rr = np.log10(ds.precipRateNearSurface.data)
        st = ds.tempStormTop.data
        pct = ds.PCT89

        # 统计histogram.
        X1, Y1, Z1 = hist2D(
            Rr, st,
            bins=[
                np.linspace(-2, 2, 50),
                np.linspace(-60, 20, 50)
            ]
        )
        X2, Y2, Z2 = hist2D(
            Rr, pct,
            bins=[
                np.linspace(-2, 2, 50),
                np.linspace(200, 300, 50)
            ]
        )
        X3, Y3, Z3 = hist2D(
            st, pct,
            bins=[
                np.linspace(-60, 20, 50),
                np.linspace(200, 300, 50)
            ]
        )
        # 为了方便后面作图,这里提前对X1和X2进行指数运算.
        data[0, j] = (exp10(X1), Y1, Z1)
        data[1, j] = (exp10(X2), Y2, Z2)
        data[2, j] = (X3, Y3, Z3)

    # 计算dusty与clean组的差值,并存入data中.
    for i in range(3):
        Xd, Yd, Zd = data[i, 0]
        Xc, Yc, Zc = data[i, 1]
        # Zdiff = normalize_by_x(Zd) - normalize_by_x(Zc)
        data[i, 2] = (Xd, Yd, Zd - Zc)

    # 组图形状为(3, 3).
    # 行表示三种变量,第一列表示dusty,第二列表示clean,第三列表示二者差值.
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 8))
    fig.subplots_adjust(hspace=0.35, wspace=0.5)

    # 前两列绘制普通图像,第三列绘制差值.并附上colorbar.
    for i in range(3):
        add_pcolor_normal(axes[i, 0], *data[i, 0])
        add_pcolor_normal(axes[i, 1], *data[i, 1])
        add_pcolor_diff(axes[i, 2], *data[i, 2])

    # 为第一行设置ticks和label.
    for ax in axes[0, :]:
        ax.set_xscale('log')
        ax.set_xlim(1E-2, 1E2)
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.set_xlabel('Rain Rate (mm/h)', fontsize='small')
        ax.set_ylim(20, -60)
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(20))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(10))
        ax.set_ylabel('Temperature of Storm Top (℃)', fontsize='small')
    # 为第二行设置ticks和label.
    for ax in axes[1, :]:
        ax.set_xscale('log')
        ax.set_xlim(1E-2, 1E2)
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.set_xlabel('Rain Rate (mm/h)', fontsize='small')
        ax.set_ylim(200, 300)
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(20))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(10))
        ax.set_ylabel('PCT89 (K)', fontsize='small')
    # 为第三行设置ticks和label.
    for ax in axes[2, :]:
        ax.set_xlim(20, -60)
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(10))
        ax.set_xlabel('Temperature of Storm Top (℃)', fontsize='small')
        ax.set_ylim(200, 300)
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(20))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(10))
        ax.set_ylabel('PCT89 (K)', fontsize='small')
    # 为每个子图设置labelsize.
    for ax in axes.flat:
        ax.tick_params(labelsize='x-small')

    # 在第一行添加每列的说明.
    axes[0, 0].set_title('Dusty', fontsize='medium')
    axes[0, 1].set_title('Clean', fontsize='medium')
    axes[0, 2].set_title('Dusty - Clean', fontsize='medium')

    # 为整张图添加雨型说明.
    fig.suptitle(Rstr, y=0.95, fontsize='large')

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
    output_dirpath = Path(config['result_dirpath']) / 'statistics'
    if not output_dirpath.exists():
        output_dirpath.mkdir()

    ds_list = [dusty_ds, clean_ds]
    draw_plot(ds_list, 'stra', output_dirpath / 'hist2D_stra.png')
    draw_plot(ds_list, 'conv', output_dirpath / 'hist2D_conv.png')
