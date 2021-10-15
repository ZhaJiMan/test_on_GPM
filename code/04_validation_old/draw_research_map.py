#----------------------------------------------------------------------------
# 2021/05/08
# 画出研究中用于选取DPR资料的范围,同时画出地形高度.
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

from map_funcs import add_Chinese_provinces, set_map_ticks, draw_box_on_map
from region_funcs import region_mask

# 读取config.json作为全局参数.
with open('config.json', 'r') as f:
    config = json.load(f)

if __name__ == '__main__':
    map_extent = config['map_extent']
    DPR_extent = config['DPR_extent']
    topo_filepath = Path(config['data_dirpath']) / 'ETOPO2v2g_f4.nc'

    # 读取ETOPO2的地形数据.
    with xr.open_dataset(str(topo_filepath)) as ds:
        z = ds.z.sel(x=slice(*map_extent[:2]), y=slice(*map_extent[2:]))
        z /= 1000   # 单位换成km.

    proj = ccrs.PlateCarree()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=proj)

    # 画出地图.
    add_Chinese_provinces(ax, lw=0.3, ec='k', fc='none')
    ax.coastlines(resolution='10m', lw=0.3)
    set_map_ticks(ax, dx=10, dy=10, nx=1, ny=1, labelsize='small')
    ax.set_extent(map_extent, crs=proj)

    # 只选取terrain中绿色以上的部分(0.2~1).
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'new_terrain', plt.cm.terrain(np.linspace(0.2, 1, 256))
    )
    # 高度小于0的数据设为海蓝色.
    cmap.set_under('lightblue')

    # 画出地形.
    im = ax.contourf(
        z.x, z.y, z, levels=np.linspace(0, 5, 21),
        cmap=cmap, extend='both', transform=proj
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label('Elevation (km)', fontsize='small')
    cbar.set_ticks(np.linspace(0, 5, 6))
    cbar.ax.tick_params(labelsize='small')

    # 标出DPR_extent的范围.
    x = (DPR_extent[0] + DPR_extent[1]) / 2
    y = DPR_extent[3] + 0.6
    draw_box_on_map(ax, DPR_extent, color='C3', lw=1)
    ax.text(
        x, y, 'Region for DPR', color='C3', fontsize='small',
        ha='center', va='center', transform=proj
    )
    # 设置标题.
    ax.set_title('Region for Data Selection', fontsize='medium')

    # 保存图片.
    output_filepath = Path(config['result_dirpath']) / 'research_zone.png'
    fig.savefig(str(output_filepath), dpi=300, bbox_inches='tight')
    plt.close(fig)