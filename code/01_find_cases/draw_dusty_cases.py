'''
2022-04-12
画出所有污染个例和清洁个例的图片.

流程:
- 读取个例的json文件. 准备一张水平图和剖面图构成的组图.
- 利用掩膜文件在水平图上画出DPR的地表降水, 标出降水中心.
- 在水平图上画出ERA5在500hPa高度的水平风场.
- 利用add_CAL_record_to_rain_cases中的方法截取VFM剖面, 在水平图上画出
  轨道路径, 在剖面图中画出剖面, 并标出像元占比.
- 利用add_AOD_records_to_rain_cases中的方法在水平图上画出对AOD做平均的
  方框, 再画出MODIS AOD的水平分布, 保存图片. 然后清理掉MODIS AOD的图像,
  再画出MERRA2 AOD的水平分布, 保存图片.
- 将两种AOD的图片水平拼接在一起, 再删除拼接前的临时图片.
- CAL, MODIS和MERRA2数据都可以缺测不画出.

输入:
- cases_dusty.json
- cases_clean.json

输出:
- 每个个例的图片.

参数:
- extents_map: 地图范围.
- LAT_LENGTH: 截取VFM剖面时取降水中心上下共LAT_LENGTH度的纬向条带.
- AOD_BOX_LENGTH: 为降水个例计算区域平均AOD时方框的边长.

注意:
- 脚本使用了多进程.
'''
import json
from pathlib import Path
from multiprocessing import Pool
import sys
sys.path.append('../modules')

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LatitudeFormatter
from PIL import Image

import helper_tools
import data_tools
import region_tools
import plot_tools
from add_AOD_records_to_rain_cases import AOD_BOX_LENGTH
from add_CAL_record_to_rain_cases import LAT_LENGTH

# 读取配置文件, 作为全局变量.
with open('config.json') as f:
    config = json.load(f)

def draw_one_case(case, dirpath_output):
    '''画出污染或清洁个例的图像.'''
    # 读取参数.
    extents_map = config['extents_map']
    case_number = case['case_number']
    rain_time = pd.to_datetime(case['rain_time'])
    clon, clat = case['rain_center']

    # 设置cmap和norm.
    cmap_aod = plot_tools.get_aod_cmap()
    cmap_rain, norm_rain = plot_tools.get_rain_cmap()
    colors = [
        'white', 'lightcyan', 'skyblue', 'gold', 'red',
        'seagreen', 'palegreen', 'black', 'darkgoldenrod'
    ]
    labels = [
        'invalid', 'clear air', 'cloud', 'aerosol', 'strato.',
        'surface', 'subsurf.', 'no signal', 'dust'
    ]
    cmap_ftype, norm_ftype, ticks = plot_tools.make_qualitative_cmap(colors)

    # 组图由ax1, ax2, cax1和cax2组成,
    # 分别表示地图, 剖面图和两个colorbar.
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_axes([0.25, 0.4, 0.5, 0.5], projection=proj)
    # 绘制地图.
    plot_tools.add_Chinese_provinces(ax1, lw=0.3, fc='none', zorder=1.5)
    ax1.coastlines(resolution='10m', lw=0.3, zorder=1.5)
    plot_tools.set_map_extent_and_ticks(
        ax1, extents_map,
        xticks=np.arange(-180, 190, 10),
        yticks=np.arange(-90, 100, 10),
        nx=1, ny=1
    )
    ax1.tick_params(labelsize='x-small')

    # 根据地图ax1的位置设定其它Axes.
    ax2 = plot_tools.add_equal_axes(ax1, loc='bottom', pad=0.08, width=0.15)
    cax1 = plot_tools.add_equal_axes(ax1, loc='left', pad=0.07, width=0.02)
    cax2 = plot_tools.add_equal_axes(ax1, loc='right', pad=0.05, width=0.02)
    cax3 = plot_tools.add_equal_axes(ax2, loc='bottom', pad=0.05, width=0.02)

    # 读取DPR数据.
    with data_tools.ReaderDPR(case['filepath_DPR']) as f:
        lon, lat = f.read_lonlat()
        surfRr = f.read_ds('SLV/precipRateNearSurface')
    mask_case = np.load(case['filepath_mask'])
    # 去除个例以外的降水.
    surfRr[~mask_case] = 0
    # 截取DPR数据.
    nscan, nray = surfRr.shape
    midray = nray // 2
    mask_scan = region_tools.region_mask(
        lon[:, midray], lat[:, midray], extents_map
    )
    lon = lon[mask_scan, :]
    lat = lat[mask_scan, :]
    surfRr = surfRr[mask_scan, :]

    # 画出地表降水.
    im = ax1.pcolormesh(
        lon, lat, surfRr,
        cmap=cmap_rain, norm=norm_rain,
        shading='nearest', transform=proj
    )
    cbar = fig.colorbar(im, cax=cax2, extend='both')
    cbar.set_label('Rain Rate (mm/h)', fontsize='x-small')
    cbar.ax.tick_params(labelsize='x-small')
    # 标出降水中心.
    ax1.plot(
        clon, clat, 'r*', ms=4, mew=0.4,
        transform=proj, label='Rain Center'
    )
    # 左上角标出降水时间.
    ax1.set_title(
        rain_time.strftime('%Y-%m-%d %H:%M'),
        loc='left', fontsize='x-small'
    )

    # 读取ERA5数据.
    ds = xr.load_dataset(case['filepath_ERA5'])
    # 截取ERA5数据.
    lonmin, lonmax, latmin, latmax = extents_map
    level = 500
    ds = ds.sel(
        level=level,
        longitude=slice(lonmin, lonmax),
        latitude=slice(latmin, latmax)
    )

    # 画出水平风场.
    Q = ax1.quiver(
        ds.longitude.values, ds.latitude.values,
        ds.u.values, ds.v.values,
        scale_units='inches', scale=180, angles='uv',
        units='inches', width=0.008, headwidth=4,
        regrid_shape=15, transform=proj, zorder=1.6
    )
    # 右下角添加风箭头图例的背景方块.
    w, h = 0.12, 0.1
    rect = mpatches.Rectangle(
        (1 - w, 0), w, h, transform=ax1.transAxes,
        fc='white', ec='k', lw=0.5, zorder=1.6
    )
    ax1.add_patch(rect)
    # 设置风箭头图例.
    wnd = 30
    qk = ax1.quiverkey(
        Q, X=1-w/2, Y=0.7*h, U=wnd,
        label=f'{wnd} m/s', labelpos='S', labelsep=0.05,
        fontproperties={'size': 'xx-small'}
    )

    # 设置截取CAL轨道数据的范围.
    half = LAT_LENGTH / 2
    lonmin, lonmax, _, _ = extents_map
    latmin = clat - half
    latmax = clat + half
    extents_scan = [lonmin, lonmax, latmin, latmax]

    # 读取CAL数据并绘制水平和垂直分布.
    record_CAL = case['record_CAL']
    if record_CAL:
        with data_tools.ReaderCAL(record_CAL['filepath_CAL']) as f:
            lon, lat = f.read_lonlat()
            hgt = f.read_hgt()
            time = f.read_time()
            ftype = f.read_ftype()
        ftype = data_tools.get_ftype_with_dust(ftype)
        # 截取数据.
        mask_scan = region_tools.region_mask(lon, lat, extents_scan)
        ftype = ftype[mask_scan, :]
        ftype = ftype[:, hgt <= 15]
        hgt = hgt[hgt <= 15]
        lon = lon[mask_scan]
        lat = lat[mask_scan]
        time = time[mask_scan].mean()

        # 画出CAL轨道.
        ax1.plot(
            lon, lat, 'r-', lw=1,
            transform=proj, label='CALIPSO Track'
        )
        ax1.plot(
            lon[[0, -1]], lat[[0, -1]],
            'ro', ms=2, transform=proj
        )
        # 画出VFM截面.
        im = ax2.pcolormesh(
            lat, hgt, ftype.T,
            cmap=cmap_ftype, norm=norm_ftype, shading='nearest'
        )
        # 左上角标出时间.
        ax2.set_title(
            time.strftime('%Y-%m-%d %H:%M'),
            loc='left', fontsize='x-small'
        )
        # 右上角标出沙尘和气溶胶比例.
        str_dust = 'Dust Ratio: {:.1f}%'.format(record_CAL['ratio_dust'])
        str_aerosol = 'Aerosol Ratio: {:.1f}%'.format(
            record_CAL['ratio_aerosol']
        )
        ax2.set_title(
            str_dust + '\n' + str_aerosol,
            fontsize='x-small', loc='right'
        )
    else:
        # 创建填充colorbar的对象.
        im = cm.ScalarMappable(norm_ftype, cmap_ftype)
        ax2.set_xlim(latmin, latmax)
    cbar = fig.colorbar(im, cax=cax3, orientation='horizontal')
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)
    cbar.ax.tick_params(length=0, labelsize=5)

    # 设置ax2的刻度.
    ax2.set_ylim(0, 15)
    ax2.set_ylabel('Height (km)', fontsize='x-small')
    ax2.xaxis.set_major_formatter(LatitudeFormatter())
    ax2.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax2.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(5))
    ax2.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax2.tick_params(labelsize='x-small')

    # 设置选取AOD范围的经纬度方框.
    half = AOD_BOX_LENGTH / 2
    lonmin = clon - half
    lonmax = clon + half
    latmin = clat - half
    latmax = clat + half
    extents_box = [lonmin, lonmax, latmin, latmax]

    # 画出计算平均AOD的方框.
    plot_tools.add_box_on_map(
        ax1, extents_box, lw=1, ec='magenta', fc='none',
        label='AOD Box', zorder=1.8
    )

    # 画出ax1的图例.
    ax1.legend(
        loc='upper right', markerscale=2, fontsize='xx-small',
        fancybox=False, handletextpad=0.5
    )

    # 为子图标出字母标识.
    plot_tools.letter_axes(
        [ax1, ax2], x=[0.04, 0.04], y=[0.96, 0.9],
        fontsize='x-small'
    )

    # 读取MYD数据.
    record_MYD = case['record_MYD']
    if record_MYD:
        with xr.open_dataset(record_MYD['filepath_MYD']) as ds:
            aod = ds.aod_combined.load()
        # 画出AOD的水平分布.
        im = ax1.pcolormesh(
            aod.lon, aod.lat, aod,
            cmap=cmap_aod, vmin=0, vmax=2,
            shading='nearest', transform=proj, zorder=0.9
        )
        # 右上角标出AOD方框内的平均值.
        str_aod = 'AOD: {:.2f}'.format(record_MYD['aod_combined'])
        str_ae = 'AE: {:.2f}'.format(record_MYD['ae_db'])
        ax1.set_title(
            str_aod + '\n' + str_ae,
            loc='right', fontsize='x-small'
        )
    else:
        # 创建填充colorbar的对象.
        im = cm.ScalarMappable(norm_aod, cmap_aod)
    cbar = fig.colorbar(im, cax=cax1, extend='both')
    cbar.set_ticks(mticker.MaxNLocator(5))
    cbar.set_label('AOD', fontsize='x-small')
    cbar.ax.tick_params(labelsize='x-small')
    # 把cbar的刻度和标签移到左边.
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')

    # 临时保存图片.
    filepath_output = dirpath_output / (case_number + '.png')
    filepath_temp1 = dirpath_output / (case_number + '_MYD.png')
    filepath_temp2 = dirpath_output / (case_number + '_MERRA2.png')
    fig.savefig(str(filepath_temp1), dpi=300, bbox_inches='tight')
    # 去掉MYD的填色图并重新创建cax1.
    im.remove()
    cax1.remove()
    cax1 = plot_tools.add_equal_axes(ax1, loc='left', pad=0.07, width=0.02)

    # 读取MERRA2数据.
    record_MERRA2 = case['record_MERRA2']
    if record_MERRA2:
        with xr.open_dataset(record_MERRA2['filepath_MERRA2']) as ds:
            aod = ds.aod_dust.sel(time=rain_time, method='nearest').load()
        # 画出AOD的水平分布.
        im = ax1.contourf(
            aod.lon, aod.lat, aod,
            levels=np.linspace(0, 0.5, 11), cmap=cmap_aod,
            extend='both', transform=proj, zorder=0.9
        )
        # 在ax1右上角标出AOD方框内的平均值.
        str_total = 'Total AOD: {:.2f}'.format(record_MERRA2['aod_total'])
        str_dust = 'Dust AOD: {:.2f}'.format(record_MERRA2['aod_dust'])
        ax1.set_title(
            str_total + '\n' + str_dust,
            loc='right', fontsize='x-small'
        )
        cbar = fig.colorbar(im, cax=cax1)
    else:
        # 创建填充colorbar的对象.
        im = cm.ScalarMappable(norm_aod, cmap_aod)
        cbar = fig.colorbar(im, cax=cax1, extend='both')
    cbar.set_ticks(mticker.MaxNLocator(5))
    cbar.set_label('Dust AOD', fontsize='x-small')
    cbar.ax.tick_params(labelsize='x-small')
    # 把cbar的刻度和标签移到左边.
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')

    # 临时保存图片.
    fig.savefig(str(filepath_temp2), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 拼接两张临时图片.
    image1 = Image.open(str(filepath_temp1))
    image2 = Image.open(str(filepath_temp2))
    size = (image1.width + image2.width, image1.height)
    merged = Image.new(image1.mode, size)
    merged.paste(image1, (0, 0))
    merged.paste(image2, (image1.width, 0))
    merged.save(str(filepath_output))
    # 删除临时图片.
    filepath_temp1.unlink()
    filepath_temp2.unlink()

if __name__ == '__main__':
    # 读取两组个例.
    dirpath_result = Path(config['dirpath_result'])
    filepath_dusty = dirpath_result / 'cases_dusty.json'
    filepath_clean = dirpath_result / 'cases_clean.json'
    with open(str(filepath_dusty)) as f:
        cases_dusty = json.load(f)
    with open(str(filepath_clean)) as f:
        cases_clean = json.load(f)

    # 重新创建输出目录.
    dirpath_dusty = dirpath_result / 'cases_dusty'
    dirpath_clean = dirpath_result / 'cases_clean'
    helper_tools.renew_dir(dirpath_dusty)
    helper_tools.renew_dir(dirpath_clean)

    # 画出每个个例的图像.
    p = Pool(10)
    for case in cases_dusty:
        p.apply_async(draw_one_case, args=(case, dirpath_dusty))
    for case in cases_clean:
        p.apply_async(draw_one_case, args=(case, dirpath_clean))
    p.close()
    p.join()
