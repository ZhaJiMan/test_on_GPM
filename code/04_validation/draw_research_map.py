#----------------------------------------------------------------------------
# 2021/10/15
# 画出研究中用于选取DPR资料的范围,同时画出地形高度和纬度分区.
#----------------------------------------------------------------------------
import json
from pathlib import Path
import sys
sys.path.append('../modules')

import numpy as np
import xarray as xr

import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from region_funcs import region_mask
from map_funcs import add_Chinese_provinces, set_map_extent_and_ticks

# 读取config.json作为全局参数.
with open('config.json', 'r') as f:
    config = json.load(f)

if __name__ == '__main__':
    topo_filepath = Path(config['data_dirpath']) / 'ETOPO2v2g_f4.nc'
    DPR_extent = config['DPR_extent']
    lonmin, lonmax, latmin, latmax = DPR_extent

    # 读取ETOPO2的地形数据.
    ds = xr.load_dataset(str(topo_filepath))
    z = ds.z.sel(x=slice(lonmin, lonmax), y=slice(latmin, latmax))
    z /= 1000   # 单位换成km.

    proj = ccrs.PlateCarree()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=proj)

    # 画出地图.
    add_Chinese_provinces(ax, lw=0.3, ec='k', fc='none')
    ax.coastlines(resolution='10m', lw=0.3)
    set_map_extent_and_ticks(
        ax,
        extent=DPR_extent,
        xticks=np.arange(-180, 180 + 4, 4),
        yticks=np.arange(-90, 90 + 4, 4),
        nx=1, ny=1
    )

    # 只选取terrain中绿色以上的部分(0.2~1).
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'new_terrain', plt.cm.terrain(np.linspace(0.2, 1, 256))
    )
    # 高度小于0的数据设为海蓝色.
    cmap.set_under('lightblue')

    # 画出地形.
    im = ax.contourf(
        z.x, z.y, z, levels=np.linspace(0, 3, 21),
        cmap=cmap, extend='both', transform=proj
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label('Elevation (km)', fontsize='small')
    cbar.set_ticks(np.linspace(0, 5, 6))
    cbar.ax.tick_params(labelsize='small')

    # 画出纬度分区.
    bins = np.linspace(latmin, latmax, 4)
    for edge in bins[1:-1]:
        ax.plot(
            [lonmin, lonmax], [edge] * 2,
            lw=1, c='r', transform=proj
        )
    centers = (bins[1:] + bins[:-1]) / 2
    for i, center in enumerate(centers):
        xc = (lonmin + lonmax) / 2
        ax.text(
            xc, center, f'R{i + 1}', fontsize=12, color='r',
            ha='center', va='center', transform=proj
        )
    # 设置标题.
    ax.set_title('Region for Data Selection', fontsize='medium')

    # 保存图片.
    output_filepath = Path(config['result_dirpath']) / 'research_zone.png'
    fig.savefig(str(output_filepath), dpi=300, bbox_inches='tight')
    plt.close(fig)
