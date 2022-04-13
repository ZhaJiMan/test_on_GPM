'''
2022-04-12
画出所有个例样本的空间分布频率图.
组图形状为(1, 2), 分别表示污染个例和清洁个例.

注意:
- 虽然没有画出研究区域外的样本, 但后续的统计中是含有的.
'''
import json
from pathlib import Path
import sys
sys.path.append('../modules')

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cmaps
import cartopy.crs as ccrs

import profile_tools
import plot_tools

# 读取配置文件, 将config作为全局变量.
with open('config.json') as f:
    config = json.load(f)

if __name__ == '__main__':
    # 读取两组个例的记录和合并后的数据.
    dirpath_input = Path(config['dirpath_input'])
    dirpath_result = Path(config['dirpath_result'])
    dirpath_merged = Path(config['dirpath_data'], 'DPR_case', 'merged')
    with open(str(dirpath_input / 'cases_dusty.json')) as f:
        cases_dusty = json.load(f)
    with open(str(dirpath_input / 'cases_clean.json')) as f:
        cases_clean = json.load(f)
    ds_dusty = xr.load_dataset(str(dirpath_merged / 'data_dusty.nc'))
    ds_clean = xr.load_dataset(str(dirpath_merged / 'data_clean.nc'))

    # 读取地图范围.
    extents_map = config['extents_map']
    extents_DPR = config['extents_DPR']

    # 读取两组个例的降水中心位置.
    centers_list = []
    for cases in [cases_dusty, cases_clean]:
        centers = []
        for case in cases:
            center = case['rain_center']
            centers.append(center)
        centers_list.append(centers)

    # 设定二维histogram的bins.
    # 其实少量样本会超出extents_DPR的范围, 但这里选择不画出.
    lonmin, lonmax, latmin, latmax = extents_DPR
    xbins = np.linspace(lonmin, lonmax, 101)
    ybins = np.linspace(latmin, latmax, 101)
    x = (xbins[1:] + xbins[:-1]) / 2
    y = (ybins[1:] + ybins[:-1]) / 2

    # 计算二维histogram.
    Hs = np.zeros((2, len(y), len(x)))
    for i, ds in enumerate([ds_dusty, ds_clean]):
        lon = ds.lon.values
        lat = ds.lat.values
        Hs[i, :, :] = profile_tools.hist2d(
            lon, lat, xbins, ybins, sigma=1, norm='sum'
        ) * 100

    # 组图含两张子图, 表示污染分组.
    titles = ['Dusty Cases', 'Clean Cases']
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(1, 2, subplot_kw={'projection': proj})
    fig.subplots_adjust(wspace=0.3)
    # 画出地图.
    for ax in axes:
        plot_tools.add_Chinese_provinces(ax, lw=0.3, fc='none', zorder=1.5)
        ax.coastlines(resolution='10m', lw=0.3, zorder=1.5)
        plot_tools.set_map_extent_and_ticks(
            ax, extents_map,
            xticks=np.arange(-180, 190, 10),
            yticks=np.arange(-90, 100, 10),
            nx=1, ny=1
        )
        ax.tick_params(labelsize='x-small')

    # 画出分布.
    for i, ax in enumerate(axes):
        im = ax.contourf(
            x, y, Hs[i, :, :], levels=40,
            cmap=cmaps.WhBlGrYeRe, extend='both', transform=proj
        )
        cbar = fig.colorbar(
            im, ax=ax, pad=0.1, orientation='horizontal',
            ticks=mticker.LinearLocator(5), format='%.3f'
        )
        cbar.ax.tick_params(labelsize='x-small')
        ax.set_title(titles[i], fontsize='x-small')

    # 标出每个降水个例的中心.
    for centers, ax in zip(centers_list, axes):
        for center in centers:
            clon, clat = center
            marker, = ax.plot(
                clon, clat, 'r+', ms=3, mew=0.45,
                transform=proj, label='rain center'
            )
        # 手动添加图例.
        ax.legend(
            handles=[marker], loc='upper right', markerscale=1.5,
            fontsize='x-small', fancybox=False, handletextpad=0
        )

    # 标出extents_DPR的范围.
    lonmin, lonmax, latmin, latmax = extents_DPR
    x = (lonmin + lonmax) / 2
    y = latmax + 0.8
    for ax in axes:
        plot_tools.add_box_on_map(ax, extents_DPR, lw=1, ec='C3', fc='none')
        ax.text(
            x, y, 'Region for DPR', color='C3', fontsize='xx-small',
            ha='center', va='center', transform=proj
        )

    # 为子图标出字母标识.
    plot_tools.letter_axes(axes, 0.05, 0.95, fontsize='x-small')

    # 保存图片.
    filepath_output = dirpath_result / 'horizontal_distribution.png'
    fig.savefig(str(filepath_output), dpi=300, bbox_inches='tight')
    plt.close(fig)
