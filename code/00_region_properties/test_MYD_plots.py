'''
2022-02-10
画出每一天的格点化后的MODIS AOD图像.
'''
import json
from pathlib import Path
from multiprocessing import Pool
import sys
sys.path.append('../modules')

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs

import helper_tools
import plot_tools

# 读取配置文件, 作为全局变量.
with open('config.json') as f:
    config = json.load(f)

def draw_one_file(filepath_MYD, dirpath_output):
    '''画出一个MYD文件内多种AOD的水平分布.'''
    extents_map = config['extents_map']
    ds = xr.load_dataset(str(filepath_MYD))

    # 将用于画图的参数打包到列表中.
    da_list = [
        ds.aod_dt_plot, ds.aod_db_plot, ds.aod_combined,
        ds.aod_dt_best, ds.aod_db_best, ds.ae_db_best
    ]
    labels = ['AOD'] * 5 + ['AE']

    # 组图形状为(2, 3).
    # 第一列是DT AOD, 第二列是DB AOD, 第三列是合并AOD和AE.
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        2, 3, figsize=(12, 6), subplot_kw={'projection': proj}
    )
    fig.subplots_adjust(wspace=0.2, hspace=0.1)
    # 绘制地图.
    for ax in axes.flat:
        plot_tools.add_Chinese_provinces(ax, lw=0.3, fc='none', zorder=1.5)
        ax.coastlines(resolution='10m', lw=0.3, zorder=1.5)
        plot_tools.set_map_extent_and_ticks(
            ax, extents_map,
            xticks=np.arange(-180, 190, 10),
            yticks=np.arange(-90, 100, 10),
            nx=1, ny=1
        )
        ax.tick_params(labelsize='x-small')

    # 将数据绘制在每张子图上.
    for i, ax in enumerate(axes.flat):
        da = da_list[i]
        im = ax.pcolormesh(
            da.lon, da.lat, da, cmap='jet', vmin=0, vmax=2,
            shading='nearest', transform=proj
        )
        cbar = fig.colorbar(
            im, ax=ax, pad=0.05, shrink=0.85, extend='both',
            ticks=mticker.MaxNLocator(5)
        )
        cbar.set_label(labels[i], fontsize='x-small')
        cbar.ax.tick_params(labelsize='x-small')
        ax.set_title(da.name, fontsize='medium')

    # 用文件名作为输出的图片名.
    filename_output = filepath_MYD.stem + '.png'
    filepath_output = dirpath_output / filename_output
    fig.savefig(str(filepath_output), dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    dirpath_result = Path(config['dirpath_result'])
    dirpath_MYD = Path(config['dirpath_data'], 'MYD', 'single')
    # 创建输出目录.
    dirpath_output = dirpath_result / 'AOD' / 'test_plots'
    helper_tools.renew_dir(dirpath_output, parents=True)

    # 画出dirpath_MYD内的所有文件.
    p = Pool(10)
    for filepath_MYD in dirpath_MYD.iterdir():
        p.apply_async(
            draw_one_file,
            args=(filepath_MYD, dirpath_output)
        )
    p.close()
    p.join()
