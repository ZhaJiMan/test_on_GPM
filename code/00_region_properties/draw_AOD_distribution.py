'''
2022-02-10
画出地图范围内多年春季平均的AOD值, 和研究区域内AOD的一维PDF.

流程:
- 读取合并版本的MODIS和MERRA2文件, 选取多种AOD变量, 对时间维做平均, 画出
  地图范围内的填色图. 在地图上用方框标记研究区域, 并把研究区域内的平均值
  标记在方框上面.
- 读取合并版本的MODIS和MERRA2文件, 选取多种AOD变量, 截取研究区域内的数据,
  将所有数据点用于计算PDF(核密度估计), 最后画成曲线图.

输入:
- data_MYD.nc
- data_MERRA2.nc

输出:
- 多年平均AOD的水平分布组图.
- AOD的一维PDF分布图.

注意:
- 具体画出的变量请见代码.
'''
import json
from pathlib import Path
import sys
sys.path.append('../modules')

import numpy as np
import xarray as xr
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs

import helper_tools
import plot_tools

# 读取配置文件, 作为全局变量.
with open('config.json') as f:
    config = json.load(f)

def sel_data(da, extents):
    '''根据经纬度方框选取DataArray或Dataset.'''
    lonmin, lonmax, latmin, latmax = extents
    return da.sel(
        lon=slice(lonmin, lonmax),
        lat=slice(latmin, latmax)
    )

def pdf1d(x, data, bw_method=None):
    '''利用核密度估计计算一维PDF分布.'''
    data = np.ravel(data)
    data = data[~np.isnan(data)]
    kernel = gaussian_kde(data, bw_method)

    return kernel(x)

def draw_horizontal_distribution(ds_MYD, ds_MERRA2, filepath_output):
    '''画出MYD和MERRA2数据多年平均的AOD水平分布.'''
    extents_map = config['extents_map']
    extents_DPR = config['extents_DPR']
    lonmin, lonmax, latmin, latmax = extents_DPR

    # 为了循环画图, 将数据打包至列表中.
    da_list = [
        ds_MYD.aod_dt_plot, ds_MYD.aod_db_plot, ds_MYD.aod_combined,
        ds_MYD.aod_dt_best, ds_MYD.aod_db_best, ds_MYD.ae_db_best,
        ds_MERRA2.aod_total, ds_MERRA2.aod_fine, ds_MERRA2.aod_coarse
    ]
    vmaxs = [1] * 5 + [2] + [1] * 2 + [0.25]
    levels_list = [np.linspace(0, vmax, 101) for vmax in vmaxs]
    labels = ['AOD'] * 5 + ['AE'] + ['AOD'] * 3

    # 提前计算标注用的文字位置.
    x = (lonmin + lonmax) / 2
    y = latmax + 0.8

    # 组图形状维3x3.
    # 前两行绘制MYD的五种AOD和AE, 第二行绘制MERRA2的AOD.
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        3, 3, figsize=(16, 12), subplot_kw={'projection': proj}
    )
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
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
        da_map = da.mean(dim='time')
        da_avg = sel_data(da_map, extents_DPR).mean()

        # 画出da_map.
        im = ax.contourf(
            da_map.lon, da_map.lat, da_map, levels=levels_list[i],
            cmap='jet', extend='both', transform=proj
        )
        cbar = fig.colorbar(
            im, ax=ax, pad=0.05, shrink=0.9,
            ticks=mticker.MaxNLocator(5)
        )
        cbar.set_label(labels[i], fontsize='x-small')
        cbar.ax.tick_params(labelsize='x-small')
        ax.set_title(da_map.name, fontsize='medium')

        # 标出extents_DPR的范围和da_avg的值.
        plot_tools.add_box_on_map(ax, extents_DPR, lw=1, ec='C3', fc='none')
        ax.text(
            x, y, f'mean={float(da_avg):.2f}',
            color='C3', fontsize='small', ha='center', va='center',
            transform=proj
        )

    # 保存图片.
    fig.savefig(str(filepath_output), dpi=300, bbox_inches='tight')
    plt.close(fig)

def draw_pdf1d(ds_MYD, ds_MERRA2, filepath_output):
    '''画出研究区域内MYD和MERRA2的AOD数据的一维PDF分布.'''
    # 预先截取extents_DPR中的数据.
    extents_DPR = config['extents_DPR']
    ds_MYD = sel_data(ds_MYD, extents_DPR)
    ds_MERRA2 = sel_data(ds_MERRA2, extents_DPR)

    # 为了循环画图, 将数据打包至列表中.
    da_list = [
        ds_MYD.aod_dt_plot,
        ds_MYD.aod_dt_best,
        ds_MYD.aod_db_plot,
        ds_MYD.aod_db_best,
        ds_MYD.aod_combined,
        ds_MERRA2.aod_total
    ]

    # 画出da_list中每种AOD的一维PDF.
    x = np.linspace(0, 2, 201)
    fig, ax = plt.subplots()
    for i, da in enumerate(da_list):
        color = f'C{i}'
        y = pdf1d(x, da.mean(dim=['lon', 'lat']))
        ax.plot(x, y, color=color, lw=1, label=da.name)
        ax.fill_between(x, y, color=color, alpha=0.4)
    ax.legend(fontsize='small', loc='upper right')

    # 设置坐标轴.
    ax.set_xlim(0, 2)
    ax.set_ylim(0, None)
    ax.set_xlabel('AOD', fontsize='medium')
    ax.set_ylabel('PDF', fontsize='medium')
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.set_title('Probability Density Function of AODs', fontsize='medium')

    # 保存图片.
    fig.savefig(str(filepath_output), dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    dirpath_result = Path(config['dirpath_result'])
    dirpath_data = Path(config['dirpath_data'])
    filepath_MYD = dirpath_data / 'MYD' / 'data_MYD.nc'
    filepath_MERRA2 = dirpath_data / 'MERRA2' / 'data_MERRA2.nc'

    # 创建输出目录.
    dirpath_output = dirpath_result / 'AOD'
    helper_tools.new_dir(dirpath_output)

    # 读取AOD数据.
    ds_MYD = xr.load_dataset(str(filepath_MYD))
    ds_MERRA2 = xr.load_dataset(str(filepath_MERRA2))

    # 画出AOD的水平分布.
    filepath_output = dirpath_output / 'horizontal_distribution.png'
    draw_horizontal_distribution(ds_MYD, ds_MERRA2, filepath_output)

    # 画出AOD的一维PDF.
    filepath_output = dirpath_output / 'pdf1d.png'
    draw_pdf1d(ds_MYD, ds_MERRA2, filepath_output)
