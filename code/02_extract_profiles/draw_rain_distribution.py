#----------------------------------------------------------------------------
# 2021/05/08
# 画出污染组与清洁组中含有的降水廓线在地图上的水平分布频次.
#----------------------------------------------------------------------------
import json
from pathlib import Path
import sys
sys.path.append('../modules')
from map_funcs import *

import numpy as np
import xarray as xr

import matplotlib as mpl
import matplotlib.pyplot as plt
import cmaps
import cartopy.crs as ccrs

# 读取配置文件,作为全局变量.
with open('config.json', 'r') as f:
    config = json.load(f)

def calc_rain_distribution(ds):
    '''计算一组个例的数据点在DPR_extent范围内的频率分布.'''
    # 设定histogram2D的范围和分辨率.
    lonmin, lonmax, latmin, latmax = config['DPR_extent']
    nlon = 50
    nlat = 50
    lon_bins = np.linspace(lonmin, lonmax, nlon + 1)
    lat_bins = np.linspace(latmin, latmax, nlat + 1)
    # 读取经纬度数据并计算.
    lon = ds.lon.data
    lat = ds.lat.data
    hist2D = np.histogram2d(lon, lat, bins=[lon_bins, lat_bins])[0]
    hist2D = hist2D / hist2D.sum() * 100

    return lon_bins, lat_bins, hist2D

if __name__ == '__main__':
    # 读取两组个例.
    result_dirpath = Path(config['result_dirpath'])
    with open(str(result_dirpath / 'found_cases.json'), 'r') as f:
        records = json.load(f)
    # 读取总的廓线数据.
    dusty_ds = xr.open_dataset(records['dusty']['profile_filepath'])
    clean_ds = xr.open_dataset(records['clean']['profile_filepath'])

    # 计算降水分布.
    dusty_data = calc_rain_distribution(dusty_ds)
    clean_data = calc_rain_distribution(clean_ds)
    data_list = [dusty_data, clean_data]

    map_extent = config['map_extent']
    DPR_extent = config['DPR_extent']
    proj = ccrs.PlateCarree()
    titles = ['Dusty Cases', 'Clean Cases']

    # 画出两组个例的图像.
    fig, axes = plt.subplots(1, 2, subplot_kw={'projection': proj})
    fig.subplots_adjust(wspace=0.3)
    for i in range(2):
        ax = axes[i]
        title = titles[i]
        lon_bins, lat_bins, hist2D = data_list[i]

        add_Chinese_provinces(ax, lw=0.3, ec='k', fc='none')
        ax.coastlines(resolution='10m', lw=0.3)
        set_map_ticks(ax, dx=10, dy=10, nx=1, ny=1, labelsize='x-small')
        ax.set_extent(map_extent, crs=proj)
        draw_box_on_map(ax, DPR_extent, color='C0', lw=1)

        im = ax.pcolormesh(
            lon_bins, lat_bins, hist2D.T, cmap=cmaps.WhBlGrYeRe,
            vmin=0.001, transform=proj
        )
        cbar = fig.colorbar(
            im, ax=ax, extend='both', pad=0.1, orientation='horizontal'
        )
        cbar.set_label('Frequency (%)', fontsize='x-small')
        cbar.ax.tick_params(labelsize='x-small')
        cbar.ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))

        ax.set_title(title, fontsize='x-small')

    # 保存图片.
    output_filepath = result_dirpath / 'rain_distribution.png'
    fig.savefig(str(output_filepath), dpi=300, bbox_inches='tight')
    plt.close(fig)
