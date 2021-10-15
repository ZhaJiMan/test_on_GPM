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
from collections import namedtuple
from multiprocessing import Pool
import sys
sys.path.append('../modules')

import h5py
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LatitudeFormatter

from region_funcs import region_mask
from helper_funcs import recreate_dir
from data_reader import Reader_for_MYD, Reader_for_CAL
from find_dusty_cases import get_CAL_mask, dust_ratio
from map_funcs import add_Chinese_provinces, set_map_extent_and_ticks

# 读取配置文件,作为全局变量.
with open('config.json', 'r') as f:
    config = json.load(f)

def get_AOD_cmap():
    '''返回适用于AOD的cmap.'''
    rgb = np.loadtxt('../modules/NEO_modis_aer_od.csv', delimiter=',')
    cmap = mpl.colors.ListedColormap(rgb / 256)

    return cmap

def get_MYD_filepaths(date):
    '''根据年月日,返回文件名中含有的时间与之匹配的MYD文件路径.'''
    MYD_dirpath = Path('/data00/0/MODIS/MYD04_L2/061')
    yy = date.strftime('%Y')
    jj = date.strftime('%j')
    yyjj = date.strftime('%Y%j')

    for MYD_filepath in (MYD_dirpath / yy / jj).glob(f'*A{yyjj}*.hdf'):
        yield MYD_filepath

def read_data(case):
    '''读取DPR,CAL,MYD三种卫星产品的数据.'''
    map_extent = config['map_extent']
    rain_center = case['rain_center']
    rain_time = pd.to_datetime(case['rain_time'])
    # 创建三种卫星数据所需的namedtuple.
    Data_DPR = namedtuple('Data_DPR', 'lon2D lat2D surfRr time')
    Data_CAL = namedtuple('Data_CAL', 'lon1D lat1D vfm hgt time')
    Data_MYD = namedtuple('Data_MYD', 'lon2D lat2D aod')

    # 读取DPR数据
    with h5py.File(case['DPR_filepath'], 'r') as f:
        lon2D = f['NS/Longitude'][:]
        lat2D = f['NS/Latitude'][:]
        surfRr = f['NS/SLV/precipRateNearSurface'][:]
    with xr.open_dataset(case['mask_filepath']) as ds:
        case_mask = ds.case_mask.data
    # 去除个例以外的降水.
    surfRr[~case_mask] = 0

    # 根据scan_mask截取数据.
    nscan, nray = surfRr.shape
    midray = nray // 2
    scan_mask = region_mask(lon2D[:, midray], lat2D[:, midray], map_extent)
    lon2D = lon2D[scan_mask, :]
    lat2D = lat2D[scan_mask, :]
    surfRr = surfRr[scan_mask, :]

    # 存储DPR数据.
    DPR_data = Data_DPR(lon2D, lat2D, surfRr, rain_time)

    # 读取CALIPSO数据.
    CAL_data_list = []
    for CAL_filepath in case['CAL_filepaths']:
        f = Reader_for_CAL(CAL_filepath)
        lon1D, lat1D = f.read_lonlat()
        time = f.read_time()
        vfm = f.read_vfm()
        hgt = f.read_hgt()
        f.close()
        # 截取降水中心南北宽度为CAL_width的数据.
        scan_mask = get_CAL_mask((lon1D, lat1D), rain_center)
        lon1D = lon1D[scan_mask]
        lat1D = lat1D[scan_mask]
        time = time[scan_mask].mean()
        vfm = vfm[scan_mask, :]
        ratio = dust_ratio(vfm)
        # 存储CALIPSO数据.
        CAL_data = Data_CAL(lon1D, lat1D, vfm, hgt, time)
        CAL_data_list.append(CAL_data)

    # 读取MODIS数据.
    MYD_data_list = []
    for MYD_filepath in get_MYD_filepaths(rain_time):
        f = Reader_for_MYD(str(MYD_filepath))
        lon2D, lat2D = f.read_lonlat()
        aod = f.read_sds('AOD_550_Dark_Target_Deep_Blue_Combined')
        f.close()
        # 若没有数据点落入map_extent中,跳过该文件.
        MYD_mask = region_mask(lon2D, lat2D, map_extent)
        if not MYD_mask.any():
            continue
        # 存储MODIS数据.
        MYD_data = Data_MYD(lon2D, lat2D, aod)
        MYD_data_list.append(MYD_data)

    return DPR_data, CAL_data_list, MYD_data_list

def draw_one_plot(DPR_data, CAL_data, MYD_data_list, output_filepath):
    '''画出一个个例的一张图.'''
    map_extent = config['map_extent']
    # 设置降水率用的norm和cmap.
    bounds = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0]
    nbin = len(bounds) - 1
    norm_Rr = mpl.colors.BoundaryNorm(bounds, nbin)
    cmap_Rr = mpl.cm.get_cmap('jet', nbin)
    cmap_Rr.set_under(color='lavender', alpha=0.2)

    # 设置VFM用的norm和cmap.
    colors = [
        'white', 'lightcyan', 'skyblue', 'gold', 'red',
        'seagreen', 'palegreen', 'black', 'darkgoldenrod'
    ]
    labels = [
        'invalid', 'clear air', 'cloud', 'aerosol', 'strato.',
        'surface', 'subsurf.', 'no signal', 'dust'
    ]
    ncolor = len(colors)
    cmap_vfm = mpl.colors.ListedColormap(colors)
    norm_vfm = mpl.colors.Normalize(vmin=0, vmax=ncolor)
    ticks = np.linspace(0, ncolor - 1, ncolor) + 0.5

    # 设置画布.
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_axes([0.28, 0.4, 0.5, 0.5], projection=proj)
    cax1 = fig.add_axes([0.18, 0.28, 0.02, 0.5])
    cax2 = fig.add_axes([0.82, 0.28, 0.02, 0.5])
    ax2 = fig.add_axes([0.28, 0.16, 0.5, 0.2])

    # 绘制地图.
    add_Chinese_provinces(ax1, lw=0.3, ec='k', fc='none')
    ax1.coastlines(resolution='10m', lw=0.3)
    set_map_extent_and_ticks(
        ax1,
        extent=map_extent,
        xticks=np.arange(-180, 190, 10),
        yticks=np.arange(-90, 100, 10),
        nx=1, ny=1
    )
    ax1.tick_params(labelsize='x-small')

    # 画出MODIS AOD的水平分布.
    for MYD_data in MYD_data_list:
        im = ax1.pcolormesh(
            MYD_data.lon2D, MYD_data.lat2D, MYD_data.aod,
            cmap=get_AOD_cmap(), vmin=0, vmax=2,
            shading='nearest', transform=proj
        )
    cbar = fig.colorbar(im, cax=cax1, extend='both')
    cbar.set_label('AOD', fontsize='x-small')
    cbar.ax.tick_params(labelsize='x-small')
    # 把ticks和label移到左边.
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')

    # 画出地表降水.
    im = ax1.pcolormesh(
        DPR_data.lon2D, DPR_data.lat2D, DPR_data.surfRr,
        cmap=cmap_Rr, norm=norm_Rr, shading='nearest', transform=proj
    )
    cbar = fig.colorbar(im, cax=cax2, extend='both')
    cbar.set_label('Rain Rate (mm/h)', fontsize='x-small')
    cbar.ax.tick_params(labelsize='x-small')

    # 标出降水中心.
    clon = DPR_data.lon2D[DPR_data.surfRr > 0].mean()
    clat = DPR_data.lat2D[DPR_data.surfRr > 0].mean()
    ax1.plot(
        clon, clat, 'r*', ms=4, mew=0.4, alpha=0.8,
        transform=proj, label='rain center'
    )

    # 画出CALIPSO轨道.
    ax1.plot(
        CAL_data.lon1D, CAL_data.lat1D,
        'r-', lw=1, alpha=0.8, transform=proj,
        label='CALIPSO track'
    )
    ax1.plot(
        CAL_data.lon1D[[0, -1]], CAL_data.lat1D[[0, -1]],
        'ro', ms=2, alpha=0.5, transform=proj
    )

    # 设置降水中心和CALIPSO轨道的legend.
    ax1.legend(
        loc='upper right', markerscale=2, fontsize='xx-small',
        fancybox=False, handletextpad=0.5
    )

    # 标出DPR和CAL的时间.
    ax1.set_title(DPR_data.time.strftime('%Y-%m-%d %H:%M'), fontsize='x-small')
    ax2.set_title(CAL_data.time.strftime('%Y-%m-%d %H:%M'), fontsize='x-small')

    # 画出VFM截面.
    im = ax2.pcolormesh(
        CAL_data.lat1D, CAL_data.hgt, CAL_data.vfm.T,
        cmap=cmap_vfm, norm=norm_vfm, shading='nearest'
    )
    # 在右上角标出沙尘比例.
    ratio = dust_ratio(CAL_data.vfm)
    ax2.set_title(f'{ratio:.2f}%', fontsize='x-small', loc='right')
    # 设置colorbar.
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('bottom', size='10%', pad=0.3)
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)
    cbar.ax.tick_params(length=0, labelsize=5)

    # 设置ax2的ticks.
    ax2.set_ylim(0, 20)
    ax2.set_ylabel('Height [km]', fontsize='x-small')
    ax2.xaxis.set_major_formatter(LatitudeFormatter())
    ax2.xaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
    ax2.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    ax2.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
    ax2.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    ax2.tick_params(labelsize='x-small')

    # 为子图标出字母标识.
    ax1.text(
        0.04, 0.96, '(a)', fontsize='x-small',
        ha='center', va='center', transform=ax1.transAxes
    )
    ax2.text(
        0.04, 0.9, '(b)', fontsize='x-small',
        ha='center', va='center', transform=ax2.transAxes
    )

    # 保存图片.
    fig.savefig(str(output_filepath), dpi=300, bbox_inches='tight')
    plt.close(fig)

def draw_one_case(case, output_dirpath):
    '''
    画出一个个例的所有图像.

    当个例对应于多个CALIPSO文件时,画出多张图像.
    每场图中含有两个子图,图一画出地表降水率和MODIS AOD的分布,
    并加上CALIPSO轨迹.图二画出CALIPSO VFM的结果.
    '''
    DPR_data, CAL_data_list, MYD_data_list = read_data(case)
    # 有几个CALIPSO文件就画出几张图.图片名采用case_number+CAL_number的形式.
    case_number = case['case_number']
    for i, CAL_data in enumerate(CAL_data_list):
        output_filename = f'{case_number}_{i+1:02}.png'
        output_filepath = output_dirpath / output_filename
        draw_one_plot(DPR_data, CAL_data, MYD_data_list, output_filepath)

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
    p = Pool(8)
    for dusty_case in dusty_cases:
        p.apply_async(draw_one_case, args=(dusty_case, dusty_dirpath))
    for clean_case in clean_cases:
        p.apply_async(draw_one_case, args=(clean_case, clean_dirpath))
    p.close()
    p.join()
