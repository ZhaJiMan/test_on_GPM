#----------------------------------------------------------------------------
# 2021/05/08
# 画出污染组与清洁组中含有的降水廓线在地图上的水平分布频次.
#----------------------------------------------------------------------------
import json
from pathlib import Path
import sys
sys.path.append('../modules')

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmaps
import cartopy.crs as ccrs

from helper_funcs import letter_subplots
from map_funcs import (
    add_Chinese_provinces,
    set_map_extent_and_ticks,
    add_box_on_map
)

# 读取配置文件,作为全局变量.
with open('config.json', 'r') as f:
    config = json.load(f)

def draw_horizontal_distribution(dusty_ds, clean_ds, output_filepath):
    '''画出污染组和清洁组降水样本在地图上的分布概率.'''
    # 设定histogram2D的范围和分辨率.
    # 选用map_extent是因为会有少量数据超出DPR_extent的范围.
    DPR_extent = config['DPR_extent']
    map_extent = config['map_extent']
    lonmin, lonmax, latmin, latmax = map_extent
    nlon = 100
    nlat = 100
    lon_bins = np.linspace(lonmin, lonmax, nlon + 1)
    lat_bins = np.linspace(latmin, latmax, nlat + 1)

    # 读取经纬度数据并计算.
    Hs = np.zeros((2, nlat, nlon))
    for i, ds in enumerate([dusty_ds, clean_ds]):
        lon = ds.lon.data
        lat = ds.lat.data
        H = np.histogram2d(lon, lat, [lon_bins, lat_bins])[0].T
        Hs[i, :, :] = H / H.sum() * 100     # 单位为百分比.

    # 画出两组个例的图像.
    titles = ['Dusty Cases', 'Clean Cases']
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(1, 2, subplot_kw={'projection': proj})
    fig.subplots_adjust(wspace=0.3)
    for i, ax in enumerate(axes):
        # 画出地图.
        add_Chinese_provinces(ax, lw=0.3, ec='k', fc='none')
        ax.coastlines(resolution='10m', lw=0.3)
        set_map_extent_and_ticks(
            ax,
            extent=map_extent,
            xticks=np.arange(-180, 190, 10),
            yticks=np.arange(-90, 100, 10),
            nx=1, ny=1
        )
        ax.tick_params(labelsize='x-small')

        # 画出分布.
        im = ax.pcolormesh(
            lon_bins, lat_bins, Hs[i, :, :],
            cmap=cmaps.WhBlGrYeRe, vmin=0, transform=proj
        )
        cbar = fig.colorbar(
            im, ax=ax, extend='both', pad=0.1, orientation='horizontal'
        )
        cbar.set_label('Frequency (%)', fontsize='x-small')
        cbar.ax.tick_params(labelsize='x-small')
        cbar.ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))

        # 标出DPR_extent的范围.
        add_box_on_map(ax, DPR_extent, color='C3', lw=1)
        x = (DPR_extent[0] + DPR_extent[1]) / 2
        y = DPR_extent[3] + 0.8
        ax.text(
            x, y, 'Region for DPR', color='C3', fontsize='xx-small',
            ha='center', va='center', transform=proj
        )
        ax.set_title(titles[i], fontsize='x-small')

    # 为子图标出字母标识.
    letter_subplots(axes, (0.05, 0.95), 'x-small')

    # 保存图片.
    fig.savefig(str(output_filepath), dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    # 读取两组个例.
    result_dirpath = Path(config['result_dirpath'])
    with open(str(result_dirpath / 'found_cases.json'), 'r') as f:
        records = json.load(f)
    # 读取总的廓线数据.
    dusty_ds = xr.load_dataset(records['dusty']['profile_filepath'])
    clean_ds = xr.load_dataset(records['clean']['profile_filepath'])

    output_filepath = result_dirpath / 'rain_distribution.png'
    draw_horizontal_distribution(dusty_ds, clean_ds, output_filepath)
