#----------------------------------------------------------------------------
# 2021/10/15
# 画出2014-2020年春季研究区域内降水廓线数和平均降水速率的水平分布图.
#
# 组图形状为(1, 2),第一张是廓线数量的histogram图,第二张是binavg的降水速率图.
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
from map_funcs import add_Chinese_provinces, set_map_extent_and_ticks

# 读取配置文件,作为全局变量使用.
with open('config.json', 'r') as f:
    config = json.load(f)

def binavg2d(x, y, z, xbins, ybins):
    '''
    计算二维binning后的平均值.

    若某个bin内没有数据落入,则平均值记作NaN.
    返回数组的形状为(ny, nx),方便画出二维图像.
    '''
    nx = len(xbins) - 1
    ny = len(ybins) - 1
    count = np.histogram2d(x, y, [xbins, ybins])[0]
    total = np.histogram2d(x, y, [xbins, ybins], weights=z)[0]
    avg = np.full((nx, ny), np.nan)
    np.divide(total, count, out=avg, where=(count > 0))

    return avg.T

if __name__ == '__main__':
    # 读取数据.
    input_dirpath = Path(config['temp_dirpath']) / 'merged'
    output_dirpath = Path(config['result_dirpath'])
    ds = xr.load_dataset(str(input_dirpath / 'all_profile.nc'))

    # 根据DPR_extent的范围设置bins.
    DPR_extent = config['DPR_extent']
    lonmin, lonmax, latmin, latmax = DPR_extent
    dlon, dlat = 0.25, 0.25
    nlon = int((lonmax - lonmin) / dlon)
    nlat = int((latmax - latmin) / dlat)
    lon_bins = np.linspace(lonmin, lonmax, nlon + 1)
    lat_bins = np.linspace(latmin, latmax, nlat + 1)

    lon = ds.lon.data
    lat = ds.lat.data
    surfRr = ds.precipRateNearSurface.data

    # 计算廓线数量分布和平均降水速率分布.
    H = np.histogram2d(lon, lat, [lon_bins, lat_bins])[0].T
    A = binavg2d(lon, lat, surfRr, lon_bins, lat_bins)

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(1, 2, subplot_kw={'projection': proj})
    fig.subplots_adjust(wspace=0.3)

    # 画出地图.
    for ax in axes:
        add_Chinese_provinces(ax, lw=0.3, ec='k', fc='none')
        ax.coastlines(resolution='10m', lw=0.3)
        set_map_extent_and_ticks(
            ax,
            extent=DPR_extent,
            xticks=np.arange(-180, 180 + 4, 4),
            yticks=np.arange(-90, 90 + 4, 4),
            nx=1, ny=1
        )
        ax.tick_params(labelsize='x-small')

    # 画出廓线数目的分布.
    im1 = axes[0].pcolormesh(
        lon_bins, lat_bins, H, cmap=cmaps.WhBlGrYeRe,
        vmin=0, vmax=0.95*H.max(), transform=proj
    )
    cbar1 = fig.colorbar(
        im1, ax=axes[0], extend='both',
        orientation='horizontal', pad=0.1
    )
    cbar1.set_label('Profile Number', fontsize='x-small')
    cbar1.ax.tick_params(labelsize='x-small')

    # 画出平均降水速率的分布.
    cmap = cmaps.WhiteBlueGreenYellowRed
    cmap.set_bad('grey')    # 缺测值设为灰色.
    im2 = axes[1].pcolormesh(
        lon_bins, lat_bins, A, cmap=cmap,
        vmin=0, vmax=5, transform=proj
    )
    cbar2 = fig.colorbar(
        im2, ax=axes[1], extend='both',
        orientation='horizontal', pad=0.1
    )
    cbar2.set_label('Mean Rain Rate (mm/h)', fontsize='x-small')
    cbar2.ax.tick_params(labelsize='x-small')

    # 为子图标出字母标识.
    letter_subplots(axes, (0.05, 0.95), 'x-small')

    # 保存图片.
    output_filepath = output_dirpath / 'horizontal_distribution.png'
    fig.savefig(str(output_filepath), dpi=300, bbox_inches='tight')
    plt.close(fig)
