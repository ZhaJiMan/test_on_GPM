'''
2022-02-10
画出研究区域及其周边的地形高度.

流程:
- 画出地形高度.
- 用方框标出研究区域

输入:
- ETOPO2的地形高度文件.

输出:
- 地形高度图片.

参数:
- extents_DPR: 研究区域的范围
- extents_map: 这里表示比研究区域稍大的范围.
'''
import json
from pathlib import Path
import sys
sys.path.append('../modules')

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import cartopy.crs as ccrs

import plot_tools

# 读取配置文件, 将config作为全局变量.
with open('config.json') as f:
    config = json.load(f)

if __name__ == '__main__':
    extents_map = config['extents_map']
    extents_DPR = config['extents_DPR']
    filepath_topo = Path(config['dirpath_data'], 'ETOPO2v2g_f4.nc')

    # 读取ETOPO2的地形数据.
    lonmin, lonmax, latmin, latmax = extents_map
    with xr.open_dataset(str(filepath_topo)) as ds:
        z = ds.z.sel(
            x=slice(lonmin, lonmax),
            y=slice(latmin, latmax)
        ).load()
    z /= 1000   # 单位换成km.

    proj = ccrs.PlateCarree()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=proj)

    # 画出地图.
    plot_tools.add_Chinese_provinces(ax, lw=0.3, fc='none', zorder=1.5)
    ax.coastlines(resolution='10m', lw=0.3, zorder=1.5)
    plot_tools.set_map_extent_and_ticks(
        ax, extents_map,
        xticks=np.arange(-180, 190, 10),
        yticks=np.arange(-90, 100, 10),
        nx=1, ny=1
    )
    ax.tick_params(labelsize='small')

    # 只选取terrain中绿色以上的部分(0.2~1).
    cmap = mcolors.ListedColormap(cm.terrain(np.linspace(0.2, 1, 256)))
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

    # 标出extents_DPR的范围.
    lonmin, lonmax, latmin, latmax = extents_DPR
    x = (lonmin + lonmax) / 2
    y = latmax + 0.6
    plot_tools.add_box_on_map(ax, extents_DPR, lw=1, ec='C3', fc='none')
    ax.text(
        x, y, 'Region for DPR', color='C3', fontsize='small',
        ha='center', va='center', transform=proj
    )
    # 设置标题.
    ax.set_title('Region for Data Selection', fontsize='medium')

    # 保存图片.
    filepath_output = Path(config['dirpath_result'], 'research_region.png')
    fig.savefig(str(filepath_output), dpi=300, bbox_inches='tight')
    plt.close(fig)
