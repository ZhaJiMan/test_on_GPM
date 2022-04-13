'''
2021-07-26
画出find_rain_cases找出的所有降水个例.

流程:
- 读取降水个例的json文件.
- 每个个例对应一个DPR文件和掩膜文件, 以半透明的形式画出地图范围内DPR观测到的
  近地表降水率, 再以不透明的形式强调掩膜文件标记的近地表降水率.
- 用方框标记研究区域, 用星星标记降水中心.
- 对每个个例画出这样的图片.

输入:
- cases_rain.json文件

输出:
- 每个降水个例水平分布的图片.

参数:
- extents_DPR: 研究区域.
- extents_map: 画图时的地图范围.

注意:
- 脚本使用了多进程.
'''
import json
from pathlib import Path
from multiprocessing import Pool
import sys
sys.path.append('../modules')

import h5py
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import helper_tools
import region_tools
import data_tools
import plot_tools

# 读取配置文件, 作为全局变量.
with open('config.json') as f:
    config = json.load(f)

def draw_rain_case(case, dirpath_output):
    '''画出一个降水个例.'''
    extents_map = config['extents_map']
    extents_DPR = config['extents_DPR']
    # 读取DPR数据.
    with data_tools.ReaderDPR(case['filepath_DPR']) as f:
        lon, lat = f.read_lonlat()
        Rr_all = f.read_ds('SLV/precipRateNearSurface')
    # 读取个例的mask.
    mask_case = np.load(case['filepath_mask'])
    # 创建一个只显示个例的地表降水数组.
    Rr_case = Rr_all.copy()
    Rr_case[~mask_case] = 0

    # 用extents_map截取数据, 加快画图速度.
    nscan, nray = Rr_all.shape
    midray = nray // 2
    scan_mask = region_tools.region_mask(
        lon[:, midray], lat[:, midray], extents_map
    )
    lon = lon[scan_mask, :]
    lat = lat[scan_mask, :]
    Rr_all = Rr_all[scan_mask, :]
    Rr_case = Rr_case[scan_mask, :]

    # 画出地图.
    proj = ccrs.PlateCarree()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=proj)
    plot_tools.add_Chinese_provinces(ax, lw=0.3, fc='none', zorder=1.5)
    ax.coastlines(resolution='10m', lw=0.3, zorder=1.5)
    plot_tools.set_map_extent_and_ticks(
        ax, extents_map,
        xticks=np.arange(-180, 190, 10),
        yticks=np.arange(-90, 100, 10),
        nx=1, ny=1
    )
    ax.tick_params(labelsize='small')

    # 完整的数据设为半透明.
    cmap, norm = plot_tools.get_rain_cmap()
    ax.pcolormesh(
        lon, lat, Rr_all[:-1, :-1], cmap=cmap, norm=norm,
        alpha=0.6, shading='flat', transform=proj
    )
    # 个例的数据不透明.
    im = ax.pcolormesh(
        lon, lat, Rr_case[:-1, :-1], cmap=cmap, norm=norm,
        shading='flat', transform=proj
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, extend='both')
    cbar.set_label('Rain Rate (mm/hr)', fontsize='small')
    cbar.ax.tick_params(labelsize='small')

    # 标出extents_DPR的范围.
    lonmin, lonmax, latmin, latmax = extents_DPR
    plot_tools.add_box_on_map(ax, extents_DPR, lw=1, ec='C3', fc='none')
    x = (lonmin + lonmax) / 2
    y = latmax + 0.6
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
    filepath_output = dirpath_output / (case['case_number'] + '.png')
    fig.savefig(str(filepath_output), dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    dirpath_result = Path(config['dirpath_result'])
    with open(str(dirpath_result / 'cases_rain.json')) as f:
        cases = json.load(f)

    # 重新创建输出目录.
    dirpath_output = dirpath_result / 'cases_rain'
    helper_tools.renew_dir(dirpath_output)

    # 画出每个个例的图像.
    p = Pool(10)
    args = [(case, dirpath_output) for case in cases]
    p.starmap(draw_rain_case, args)
    p.close()
    p.join()
