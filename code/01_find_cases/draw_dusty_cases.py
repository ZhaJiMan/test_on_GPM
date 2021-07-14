#----------------------------------------------------------------------------
# 2021/05/08
# 画出之前找出的污染个例和清洁个例的图像.
#
# 每张图中含有地表降水率的水平分布,与之匹配的CALIPSO VFM轨道,当天的AOD
# 水平分布,以及VFM的纬度-高度截面图.如果一个降水事件与多个VFM轨道匹配,
# 那么有几条轨道就画出几张图.
#----------------------------------------------------------------------------
import json
from pathlib import Path
from multiprocessing import Pool
import sys
sys.path.append('../modules')
from data_reader import *
from map_funcs import add_Chinese_provinces, set_map_ticks
from region_funcs import get_extent_flag_both
from helper_funcs import recreate_dir
from find_dusty_cases import get_CAL_flag

import h5py
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LatitudeFormatter

# 读取配置文件,作为全局变量.
with open('config.json', 'r') as f:
    config = json.load(f)

def get_AOD_cmap():
    '''返回适用于AOD的cmap.'''
    rgb = np.loadtxt('../modules/NEO_modis_aer_od.csv', delimiter=',')
    cmap = mpl.colors.ListedColormap(rgb / 256)

    return cmap

def get_MYD_filepath(date):
    '''根据年月日,返回文件名中含有的时间与之匹配的MYD文件路径.'''
    MYD_dirpath = Path('/data00/0/MODIS/MYD04_L2/061')
    yy = date.strftime('%Y')
    jj = date.strftime('%j')
    yyjj = date.strftime('%Y%j')

    for MYD_filepath in (MYD_dirpath / yy / jj).glob(f'*A{yyjj}*.hdf'):
        yield MYD_filepath

def draw_one_case(case, output_dirpath):
    '''
    画出一个个例的图像.

    分为两张子图,图一画出地表降水率和MODIS AOD的分布,并加上CALIPSO轨迹.
    图二画出CALIPSO VFM的结果.
    若个例对应于多个CALIPSO文件,那么画出多张图像.
    '''
    # 读取DPR数据
    with h5py.File(case['DPR_filepath'], 'r') as f:
        lon_DPR = f['NS/Longitude'][:]
        lat_DPR = f['NS/Latitude'][:]
        time = read_DPR_time(f)
        surfRr = f['NS/SLV/precipRateNearSurface'][:]
    # 根据scan_flag截取数据.
    nscan, nray = surfRr.shape
    midray = nray // 2
    scan_flag = get_extent_flag_both(
        lon_DPR[:, midray], lat_DPR[:, midray], config['map_extent']
    )
    lon_DPR = lon_DPR[scan_flag, :]
    lat_DPR = lat_DPR[scan_flag, :]
    DPR_time = time[scan_flag].mean()
    surfRr = surfRr[scan_flag, :]

    # 读取CALIPSO数据.
    CAL_lines = []
    for CAL_filepath in case['CAL_filepaths']:
        f = reader_for_CAL(CAL_filepath)
        lon_CAL, lat_CAL = f.read_lonlat()
        time = f.read_time()
        vfm = f.read_vfm()
        hgt = f.read_hgt()
        f.close()
        # 截取数据.
        scan_flag = get_CAL_flag(
            (lon_CAL, lat_CAL), case['rain_center']
        )
        lon_CAL = lon_CAL[scan_flag]
        lat_CAL = lat_CAL[scan_flag]
        CAL_time = time[scan_flag].mean()
        vfm = vfm[scan_flag, :]
        CAL_lines.append((lon_CAL, lat_CAL, CAL_time, hgt, vfm))

    # 读取MODIS数据.
    MYD_granules = []
    for MYD_filepath in get_MYD_filepath(DPR_time):
        f = reader_for_MYD(str(MYD_filepath))
        lon_MYD, lat_MYD = f.read_lonlat()
        aod = f.read_sds('AOD_550_Dark_Target_Deep_Blue_Combined')
        f.close()
        # 若没有数据点落入map_extent中,跳过该文件.
        extent_flag = get_extent_flag_both(
            lon_MYD, lat_MYD, config['map_extent']
        )
        if not extent_flag.any():
            continue
        else:
            MYD_granules.append((lon_MYD, lat_MYD, aod))

    # 设置降水率用的norm和cmap.
    bounds1 = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0]
    nbin = len(bounds1) - 1
    norm1 = mpl.colors.BoundaryNorm(bounds1, nbin)
    cmap1 = mpl.cm.get_cmap('jet', nbin)
    # 设置<0.1的颜色,以画出swath.
    cmap1.set_under(color='lavender', alpha=0.2)

    # 设置VFM用的norm和cmap.
    colors = [
        'white', 'lightcyan', 'skyblue', 'gold', 'red',
        'seagreen', 'palegreen', 'black', 'darkgoldenrod'
    ]
    labels = [
        'invalid', 'clear air', 'cloud', 'aerosol', 'strato.',
        'surface', 'subsurf.', 'no signal', 'dust'
    ]
    cmap2 = mpl.colors.ListedColormap(colors)
    bounds2 = np.arange(cmap2.N + 1) - 0.5
    norm2 = mpl.colors.BoundaryNorm(bounds2, cmap2.N)
    ticks = np.arange(cmap2.N)

    # 有几个CALIPSO文件,就画出几张图.
    for i, CAL_line in enumerate(CAL_lines):
        lon_CAL, lat_CAL, CAL_time, hgt, vfm = CAL_line
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(6, 6))
        ax1 = fig.add_axes([0.28, 0.4, 0.5, 0.5], projection=proj)
        cax1 = fig.add_axes([0.18, 0.28, 0.02, 0.5])
        cax2 = fig.add_axes([0.82, 0.28, 0.02, 0.5])
        ax2 = fig.add_axes([0.28, 0.16, 0.5, 0.2])

        # 绘制地图.
        add_Chinese_provinces(ax1, lw=0.3, ec='k', fc='none')
        ax1.coastlines(resolution='10m', lw=0.3)
        set_map_ticks(
            ax1, dx=10, dy=10, nx=1, ny=1, labelsize='x-small'
        )
        ax1.set_extent(config['map_extent'], crs=proj)

        # 画出MODIS AOD的水平分布.
        for MYD_granule in MYD_granules:
            lon_MYD, lat_MYD, aod = MYD_granule
            im = ax1.pcolormesh(
                lon_MYD, lat_MYD, aod[:-1, :-1], cmap=get_AOD_cmap(),
                vmin=0, vmax=2, transform=proj
            )
        cbar = fig.colorbar(im, cax=cax1, extend='both')
        cbar.set_label('AOD', fontsize='x-small')
        cbar.ax.tick_params(labelsize='x-small')
        # 把ticks和label移到左边.
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')

        # 画出地表降水.
        im = ax1.pcolormesh(
            lon_DPR, lat_DPR, surfRr[:-1, :-1],
            cmap=cmap1, norm=norm1, transform=proj
        )
        cbar = fig.colorbar(im, cax=cax2, extend='both')
        cbar.set_label('Rain Rate (mm/h)', fontsize='x-small')
        cbar.ax.tick_params(labelsize='x-small')

        # 画出CALIPSO轨道.
        ax1.plot(lon_CAL, lat_CAL, 'r-', lw=1, alpha=0.5, transform=proj)
        ax1.plot(
            lon_CAL[[0, -1]], lat_CAL[[0, -1]],
            'ro', ms=2, alpha=0.5, transform=proj
        )

        # 标出DPR和CAL的时间.
        ax1.set_title(DPR_time.strftime('%Y-%m-%d %H:%M'), fontsize='x-small')
        ax2.set_title(CAL_time.strftime('%Y-%m-%d %H:%M'), fontsize='x-small')

        # 画出VFM截面.
        im = ax2.pcolormesh(
            lat_CAL, hgt, vfm[:-1, :-1].T, cmap=cmap2, norm=norm2
        )
        # 在右上角标出沙尘比例.
        dust_ratio = np.count_nonzero(vfm == 8) / vfm.size * 100
        ax2.set_title(f'{dust_ratio:.2f}%', fontsize='x-small', loc='right')
        # 设置colorbar.
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('bottom', size='10%', pad=0.3)
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(labels)
        cbar.ax.tick_params(length=0, labelsize=5)
        # 设置坐标轴.
        ax2.set_ylim(0, 20)
        ax2.set_ylabel('Height [km]', fontsize='x-small')
        ax2.xaxis.set_major_formatter(LatitudeFormatter())
        ax2.xaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
        ax2.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
        ax2.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
        ax2.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
        ax2.tick_params(labelsize='x-small')

        # 设置输出路径.对同个个例画多张图时,在文件名后追加数字.
        time_str = DPR_time.strftime('%Y%m%d_%H%M')
        if len(CAL_lines) == 1:
            output_filepath = output_dirpath / (time_str + '.png')
        else:
            output_filepath = output_dirpath / (time_str + f'_{i+1}' + '.png')
        fig.savefig(str(output_filepath), dpi=300, bbox_inches='tight')
        plt.close(fig)

if __name__ == '__main__':
    result_dirpath = Path(config['result_dirpath'])
    with open(str(result_dirpath / 'found_cases.json'), 'r') as f:
        records = json.load(f)
    dusty_cases = records['dusty']['cases']
    clean_cases = records['clean']['cases']

    # 若输出路径已存在,那么重新创建.
    dusty_dirpath = result_dirpath / 'dusty_cases'
    clean_dirpath = result_dirpath / 'clean_cases'
    recreate_dir(dusty_dirpath)
    recreate_dir(clean_dirpath)

    # 画出每个个例的图像.
    p = Pool(16)
    for dusty_case in dusty_cases:
        p.apply_async(draw_one_case, args=(dusty_case, dusty_dirpath))
    for clean_case in clean_cases:
        p.apply_async(draw_one_case, args=(clean_case, clean_dirpath))
    p.close()
    p.join()
