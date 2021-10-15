#----------------------------------------------------------------------------
# 2021/07/26
# 画出find_rain_cases找出的所有降水个例.
#
# 图中画有DPR文件的近地表降水率,个例的部分不透明,非个例的部分
# 设为半透明,以显示二者之间的区别.
#----------------------------------------------------------------------------
import json
import itertools
from pathlib import Path
from multiprocessing import Pool
import sys
sys.path.append('../modules')

import h5py
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from helper_funcs import recreate_dir
from region_funcs import region_mask
from map_funcs import (
    add_Chinese_provinces,
    set_map_extent_and_ticks,
    add_box_on_map
)

# 读取配置文件,作为全局变量.
with open('config.json', 'r') as f:
    config = json.load(f)

def draw_rain_case(case, output_dirpath):
    '''画出一个降水个例.'''
    map_extent = config['map_extent']
    DPR_extent = config['DPR_extent']
    # 读取DPR数据.
    with h5py.File(case['DPR_filepath'], 'r') as f:
        lon2D = f['NS/Longitude'][:]
        lat2D = f['NS/Latitude'][:]
        surfRr = f['NS/SLV/precipRateNearSurface'][:]
    # 读取个例的mask.
    with xr.open_dataset(case['mask_filepath']) as ds:
        case_mask = ds.case_mask.data
    # 创建一个只显示个例的地表降水数组.
    caseRr = surfRr.copy()
    caseRr[~case_mask] = 0

    # 用map_extent截取数据,加快画图速度.
    nscan, nray = surfRr.shape
    midray = nray // 2
    scan_mask = region_mask(lon2D[:, midray], lat2D[:, midray], map_extent)
    lon2D = lon2D[scan_mask, :]
    lat2D = lat2D[scan_mask, :]
    surfRr = surfRr[scan_mask, :]
    caseRr = caseRr[scan_mask, :]

    # 设置降水速率的cmap和norm.
    bounds = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0]
    nbin = len(bounds) - 1
    norm = mpl.colors.BoundaryNorm(bounds, nbin)
    cmap = mpl.cm.get_cmap('jet', nbin)
    cmap.set_under(color='lavender', alpha=0.2)

    # 画出地图.
    proj = ccrs.PlateCarree()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=proj)
    add_Chinese_provinces(ax, lw=0.3, ec='k', fc='none')
    ax.coastlines(resolution='10m', lw=0.3)
    set_map_extent_and_ticks(
        ax,
        extent=map_extent,
        xticks=np.arange(-180, 190, 10),
        yticks=np.arange(-90, 100, 10),
        nx=1, ny=1
    )
    ax.tick_params(labelsize='small')

    # 完整的数据设为半透明.
    ax.pcolormesh(
        lon2D, lat2D, surfRr, cmap=cmap, norm=norm,
        alpha=0.6, shading='nearest', transform=proj
    )
    # 个例的数据不透明.
    im = ax.pcolormesh(
        lon2D, lat2D, caseRr, cmap=cmap, norm=norm,
        shading='nearest', transform=proj
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, extend='both')
    cbar.set_label('Rain Rate (mm/h)', fontsize='small')
    cbar.ax.tick_params(labelsize='small')

    # 标出DPR_extent的范围.
    add_box_on_map(ax, DPR_extent, lw=1, color='C3')
    x = (DPR_extent[0] + DPR_extent[1]) / 2
    y = DPR_extent[3] + 0.6
    ax.text(
        x, y, 'Region for DPR', color='C3', fontsize='small',
        ha='center', va='center', transform=proj
    )

    # 标出降水中心.
    clon, clat = case['rain_center']
    ax.plot(
        clon, clat, 'r*', ms=4, mew=0.4, alpha=0.8,
        transform=proj, label='rain center'
    )
    ax.legend(
        loc='upper right', markerscale=2, fontsize='x-small',
        fancybox=False, handletextpad=0.5
    )
    # 用降水时间作为标题.
    ax.set_title(case['rain_time'], fontsize='medium')

    # 用个例编号作为输出的图片名.
    output_filepath = output_dirpath / (case['case_number'] + '.png')
    fig.savefig(str(output_filepath), dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    result_dirpath = Path(config['result_dirpath'])
    with open(str(result_dirpath / 'rain_cases.json'), 'r') as f:
        cases = json.load(f)

    # 重新创建输出目录.
    output_dirpath = result_dirpath / 'rain_cases'
    recreate_dir(output_dirpath)

    # 画出每个个例的图像.
    p = Pool(8)
    args = itertools.product(cases, [output_dirpath])
    p.starmap(draw_rain_case, args)
    p.close()
    p.join()
