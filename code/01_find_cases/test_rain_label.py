#----------------------------------------------------------------------------
# 2021/07/26
# 选取某个DPR文件,用连通域算法标记出map_extent内的降水个例,并画在地图上.
#
# 用地表降水率大于0作为二值图像.
# 过滤掉降水像元少于RAIN_PIXEL_NUM的个例.
#----------------------------------------------------------------------------
import json
from pathlib import Path
import copy
import sys
sys.path.append('../modules')

import h5py
import numpy as np
from scipy.stats import rankdata
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from data_reader import read_GPM_time
from region_funcs import region_mask
from label_funcs import two_pass
from map_funcs import add_Chinese_provinces, set_map_ticks, draw_box_on_map

# 读取配置文件,作为全局变量.
with open('config.json', 'r') as f:
    config = json.load(f)

if __name__ == '__main__':
    result_dirpath = Path(config['result_dirpath'])
    # 读取DPR数据.
    DPR_filepath = Path('/data00/0/GPM/DPR/V06/2016/201605/2A.GPM.DPR.V8-20180723.20160502-S030027-E043300.012361.V06A.HDF5')
    with h5py.File(str(DPR_filepath), 'r') as f:
        lon2D = f['NS/Longitude'][:]
        lat2D = f['NS/Latitude'][:]
        surfRr = f['NS/SLV/precipRateNearSurface'][:]
        time = read_GPM_time(f, group='NS')

    # 根据map_extent截取数据.
    map_extent = config['map_extent']
    nscan, nray = surfRr.shape
    midray = nray // 2
    scan_mask = region_mask(lon2D[:, midray], lat2D[:, midray], map_extent)
    lon2D = lon2D[scan_mask, :]
    lat2D = lat2D[scan_mask, :]
    surfRr = surfRr[scan_mask, :]
    time = time[scan_mask]


    # 标记每个降水个例.
    RAIN_RADIUS = config['RAIN_RADIUS']
    RAIN_PIXEL_NUM = config['RAIN_PIXEL_NUM']
    labelled, nlabel = two_pass(surfRr > 0, radius=RAIN_RADIUS)
    # 筛选掉像元太少的个例.
    for label in range(1, nlabel + 1):
        case_mask = labelled == label
        if np.count_nonzero(case_mask) < RAIN_PIXEL_NUM:
            labelled[case_mask] = 0
        else:
            continue
    # 重排标签.
    labelled = rankdata(labelled, method='dense').reshape(labelled.shape) - 1
    nlabel = labelled.max()

    # 设置离散的cmap.
    # nlabel太大时就不应该使用tab10了.
    lavender = np.array([0.9, 0.9, 0.98, 1])
    cmap = mpl.colors.ListedColormap(
        np.vstack([lavender, mpl.cm.tab10(np.arange(nlabel))])
    )
    norm = mpl.colors.Normalize(vmin=0, vmax=nlabel+1)
    ticks = np.arange(nlabel + 1) + 0.5
    ticklabels = ['no rain'] + list(range(1, nlabel + 1))

    # 画出地图.
    proj = ccrs.PlateCarree()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=proj)
    add_Chinese_provinces(ax, lw=0.3, ec='k', fc='none')
    ax.coastlines(resolution='10m', lw=0.3)
    set_map_ticks(ax, dx=10, dy=10, nx=1, ny=1, labelsize='small')
    ax.set_extent(map_extent, crs=proj)

    # 画出label.
    im = ax.pcolormesh(
        lon2D, lat2D, labelled, cmap=cmap, norm=norm,
        shading='nearest', transform=proj
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticklabels)
    cbar.ax.tick_params(length=0, labelsize='small')

    time_str = time.mean().strftime('%Y-%m-%d %H:%M')
    ax.set_title(time_str, fontsize='medium')

    output_filepath = result_dirpath / 'test_rain_label.png'
    fig.savefig(str(output_filepath), dpi=300, bbox_inches='tight')
    plt.close(fig)
