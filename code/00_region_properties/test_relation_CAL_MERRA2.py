'''
2022-02-10
提取落入研究区域内的所有VFM剖面, 计算气溶胶像元占比和沙尘像元占比.
再根据VFM的过境时间选取匹配的MERRA2数据, 计算区域平均的AOD.
最后将这些结果保存到npz文件中.
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
import matplotlib.ticker as mticker
import cartopy.crs as ccrs

import helper_tools
import data_tools
import region_tools
import plot_tools

# 读取配置文件, 作为全局变量.
with open('config.json') as f:
    config = json.load(f)

if __name__ == '__main__':
    # 设置时间.
    time_start = config['time_start']
    time_end = config['time_end']
    dates = pd.date_range(time_start, time_end, freq='D')
    # 只选取春季的数据.
    dates = dates[dates.month.isin([3, 4, 5])]

    extents_DPR = config['extents_DPR']
    dirpath_result = Path(config['dirpath_result'])
    dirpath_MERRA2 = Path(config['dirpath_data'], 'MERRA2', 'single')

    list_ratio_dust = []
    list_ratio_aerosol = []
    list_aod_dust = []
    list_aod_total = []
    for date in dates:
        for filepath_CAL in data_tools.get_CAL_filepaths_one_day(date):
            # 分阶段读取数据, 先读取经纬度和时间.
            f = data_tools.ReaderCAL(str(filepath_CAL))
            lon, lat= f.read_lonlat()
            time = f.read_time()

            # 若没有轨道数据落入extents_DPR, 则结束读取.
            mask_scan = region_tools.region_mask(lon, lat, extents_DPR)
            if not mask_scan.any():
                f.close()
                continue

            # 继续读取feature type数据.
            hgt = f.read_hgt()
            ftype = f.read_ftype()
            ftype = data_tools.get_ftype_with_dust(ftype)
            f.close()
            # 截取数据.
            lon = lon[mask_scan]
            lat = lat[mask_scan]
            time = time[mask_scan].mean()
            ftype = ftype[mask_scan, :]
            ftype = ftype[:, hgt <= 15]

            # 计算dust ratio和aerosol ratio.
            num_all = ftype.size
            num_dust = np.count_nonzero(ftype == 8)
            num_aerosol = np.count_nonzero(ftype == 3) + num_dust
            ratio_dust = num_dust / num_all * 100
            ratio_aerosol = num_aerosol / num_all * 100

            # 获取对应MERRA2文件中的AOD.
            filename_MERRA2 = time.strftime('%Y%m%d') + '.nc'
            filepath_MERRA2 = dirpath_MERRA2 / filename_MERRA2
            if not filepath_MERRA2.exists():
                continue

            with xr.open_dataset(str(filepath_MERRA2)) as ds:
                ds = ds.sel(time=time, method='nearest')
                ds = ds.interp(
                    lon=xr.DataArray(lon, dims='npoint'),
                    lat=xr.DataArray(lat, dims='npoint')
                )
                aod_dust = float(ds.aod_dust.mean())
                aod_total = float(ds.aod_total.mean())

            list_ratio_dust.append(ratio_dust)
            list_ratio_aerosol.append(ratio_aerosol)
            list_aod_dust.append(aod_dust)
            list_aod_total.append(aod_total)

    ratio_dust = np.array(list_ratio_dust)
    ratio_aerosol = np.array(list_ratio_aerosol)
    aod_dust = np.array(list_aod_dust)
    aod_total = np.array(list_aod_total)

    np.savez(
        'data.npz',
        ratio_dust=ratio_dust,
        ratio_aerosol=ratio_aerosol,
        aod_dust=aod_dust,
        aod_total=aod_total
    )



