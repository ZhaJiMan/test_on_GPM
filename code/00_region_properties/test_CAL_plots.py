'''
2022-02-10
画出所有经过研究区域的CALIPSO VFM剖面图, 并标出像元占比.
'''
import json
from pathlib import Path
from multiprocessing import Pool
import sys
sys.path.append('../modules')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import LatitudeFormatter

import helper_tools
import data_tools
import region_tools
import plot_tools

# 读取配置文件, 将config作为全局变量.
with open('config.json') as f:
    config = json.load(f)

def draw_one_day(date, dirpath_output):
    extents_DPR = config['extents_DPR']
    datasets = []
    for filepath_CAL in data_tools.get_CAL_filepaths_one_day(date):
        f = data_tools.ReaderCAL(str(filepath_CAL))
        lon, lat = f.read_lonlat()
        # 若没有轨道数据落入extents_DPR, 则结束读取.
        mask_scan = region_tools.region_mask(lon, lat, extents_DPR)
        if not mask_scan.any():
            f.close()
            continue

        # 继续读取feature type数据.
        hgt = f.read_hgt()
        time = f.read_time()
        ftype = f.read_ftype()
        ftype = data_tools.get_ftype_with_dust(ftype, polluted=False)
        f.close()
        # 截取数据.
        lat = lat[mask_scan]
        time = time[mask_scan].mean()
        ftype = ftype[mask_scan, :]
        ftype = ftype[:, hgt <= 15]
        hgt = hgt[hgt <= 15]

        # 计算dust ratio和aerosol ratio.
        num_all = ftype.size
        num_dust = np.count_nonzero(ftype == 8)
        num_aerosol = np.count_nonzero(ftype == 3) + num_dust
        ratio_dust = num_dust / num_all * 100
        ratio_aerosol = num_aerosol / num_all * 100

        # 存入列表中.
        data = {
            'lat': lat,
            'time': time,
            'hgt': hgt,
            'ftype': ftype,
            'ratio_dust': ratio_dust,
            'ratio_aerosol': ratio_aerosol
        }
        datasets.append(data)

    # 若没有数据则不画图.
    if not datasets:
        return None
    nfile = len(datasets)

    colors = [
        'white', 'lightcyan', 'skyblue', 'gold', 'red',
        'seagreen', 'palegreen', 'black', 'darkgoldenrod'
    ]
    labels = [
        'invalid', 'clear air', 'cloud', 'aerosol', 'strato.',
        'surface', 'subsurf.', 'no signal', 'dust'
    ]
    cmap, norm, ticks = plot_tools.make_qualitative_cmap(colors)

    # 组图形状为(nfile, 1).
    # 研究区域内有几条VFM剖面就画几条.
    fig, axes = plt.subplots(nfile, 1, figsize=(6, 2.5 * nfile))
    fig.subplots_adjust(hspace=0.5)
    axes = np.atleast_1d(axes)

    for data, ax in zip(datasets, axes):
        im = ax.pcolormesh(
            data['lat'], data['hgt'], data['ftype'].T,
            cmap=cmap, norm=norm, shading='nearest'
        )
        ax.set_title(
            data['time'].strftime('%Y-%m-%d %H:%M'),
            loc='left', fontsize='x-small'
        )
        str_dust = 'Dust Ratio: {:.1f}%'.format(data['ratio_dust'])
        str_aerosol = 'Aerosol Ratio: {:.1f}%'.format(data['ratio_aerosol'])
        ax.set_title(
            str_dust + '\n' + str_aerosol,
            fontsize='x-small', loc='right'
        )
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', aspect=30)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)
    cbar.ax.tick_params(length=0, labelsize=5)

    # 设置刻度.
    for ax in axes:
        ax.set_ylim(0, 15)
        ax.set_ylabel('Height (km)', fontsize='x-small')
        ax.xaxis.set_major_formatter(LatitudeFormatter())
        ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.tick_params(labelsize='x-small')

    # 保存图片.
    filename_output = date.strftime('%Y%m%d') + '.png'
    filepath_output = dirpath_output / filename_output
    fig.savefig(str(filepath_output), dpi=200, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    # 设置时间.
    time_start = config['time_start']
    time_end = config['time_end']
    dates = pd.date_range(time_start, time_end, freq='D')
    # 只选取春季的数据.
    dates = dates[dates.month.isin([3, 4, 5])]

    # 创建输出目录.
    dirpath_output = Path(config['dirpath_result'], 'CAL_plots')
    helper_tools.renew_dir(dirpath_output)

    p = Pool(10)
    args = [(date, dirpath_output) for date in dates]
    p.starmap(draw_one_day, args)
    p.close()
    p.join()
