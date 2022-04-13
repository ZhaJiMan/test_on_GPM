'''
2022-02-10
给定地图范围和时间范围, 格点化每一天的MYD04_L2产品的AOD值, 保存为nc文件.

流程:
- 读取某天所有的MYD04_L2文件, 收集落入地图区域内的数据点, 用格点平均的方式
  格点化所需的变量. 将结果保存到单个文件中. 对每一天进行同样的操作.
- 为方便统计, 沿时间维连接所有文件生成单个文件.

输入:
- 服务器上的MYD04_L2文件.

输出:
- 以日期命名的nc文件, 含有的格点化变量包括:
  - DT AOD
  - DB AOD
  - combined AOD
  - DB AE
- 上一条中所有文件连接起来的单个nc文件.

参数:
- extents_map: 地图范围.
- dlon, dlat: 格点分辨率.

注意:
- 时间范围为春季(3~5月), 硬编码在脚本中.
- 合并版的文件里时间维并不连续.
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

import data_tools
import helper_tools
import region_tools

# 读取配置文件,作为全局变量.
with open('config.json') as f:
    config = json.load(f)

def grid_data_one_day(date, dirpath_output):
    '''
    读取一天所有的MYD04_L2文件, 格点化研究区域内的AOD数据并保存.

    输出文件以日期命名, 保存为nc格式.
    若研究区域内无像元落入, 那么会缺少当天的文件.
    '''
    # 选取格点化的区域.
    extents_map = config['extents_map']
    lonmin, lonmax, latmin, latmax = extents_map

    # 收集落入区域内的散点数据.
    list_lon = []
    list_lat = []
    list_aod_dt_plot = []
    list_aod_dt_best = []
    list_aod_db_plot = []
    list_aod_db_best = []
    list_aod_combined = []
    list_ae_db_plot = []
    list_ae_db_best = []
    for filepath_MYD in data_tools.get_MYD_filepaths_one_day(date):
        # 分阶段读取数据.先读取经纬度.
        f = data_tools.ReaderMYD(str(filepath_MYD))
        lon, lat = f.read_lonlat()
        mask_map = region_tools.region_mask(lon, lat, extents_map)
        # 若没有数据落在区域内,跳过该文件.
        if not mask_map.any():
            f.close()
            continue
        # 继续读取AOD.
        aod_dt_plot = f.read_sds('Image_Optical_Depth_Land_And_Ocean')
        aod_dt_best = f.read_sds('Optical_Depth_Land_And_Ocean')
        aod_db_plot = f.read_sds('Deep_Blue_Aerosol_Optical_Depth_550_Land')
        aod_db_best = f.read_sds('Deep_Blue_Aerosol_Optical_Depth_550_Land_Best_Estimate')
        aod_combined = f.read_sds('AOD_550_Dark_Target_Deep_Blue_Combined')
        ae_db_plot = f.read_sds('Deep_Blue_Angstrom_Exponent_Land')
        qa_db = f.read_sds('Deep_Blue_Aerosol_Optical_Depth_550_Land_QA_Flag', scale=False)
        f.close()
        # 用QA筛选AE.
        ae_db_best = np.where(qa_db > 1, ae_db_plot, np.nan)

        # 收集散点数据.
        list_lon.append(lon[mask_map])
        list_lat.append(lat[mask_map])
        list_aod_dt_plot.append(aod_dt_plot[mask_map])
        list_aod_dt_best.append(aod_dt_best[mask_map])
        list_aod_db_plot.append(aod_db_plot[mask_map])
        list_aod_db_best.append(aod_db_best[mask_map])
        list_aod_combined.append(aod_combined[mask_map])
        list_ae_db_plot.append(ae_db_plot[mask_map])
        list_ae_db_best.append(ae_db_best[mask_map])

    # 若没有文件落入区域, 那么停止处理.
    if not list_lon:
        return None

    # 合并成一维数组.
    lon = np.concatenate(list_lon)
    lat = np.concatenate(list_lat)
    aod_dt_plot = np.concatenate(list_aod_dt_plot)
    aod_dt_best = np.concatenate(list_aod_dt_best)
    aod_db_plot = np.concatenate(list_aod_db_plot)
    aod_db_best = np.concatenate(list_aod_db_best)
    aod_combined = np.concatenate(list_aod_combined)
    ae_db_plot = np.concatenate(list_ae_db_plot)
    ae_db_best = np.concatenate(list_ae_db_best)

    # 格点化数据.
    dlon = dlat = 0.5
    bins_lon = np.arange(lonmin, lonmax + 0.5 * dlon, dlon)
    bins_lat = np.arange(latmin, latmax + 0.5 * dlat, dlat)
    glon, glat, gridded = region_tools.grid_data(
        lon=lon, lat=lat,
        data=[
            aod_dt_plot, aod_dt_best,
            aod_db_plot, aod_db_best,
            aod_combined,
            ae_db_plot, ae_db_best
        ],
        bins_lon=bins_lon, bins_lat=bins_lat
    )

    # 保存为nc文件.
    dims = ['lat', 'lon']
    attrs = {'units': 'none'}
    ds = xr.Dataset(
        data_vars={
            'aod_dt_plot': (dims, gridded[0, :, :], attrs),
            'aod_dt_best': (dims, gridded[1, :, :], attrs),
            'aod_db_plot': (dims, gridded[2, :, :], attrs),
            'aod_db_best': (dims, gridded[3, :, :], attrs),
            'aod_combined': (dims, gridded[4, :, :], attrs),
            'ae_db_plot': (dims, gridded[5, :, :], attrs),
            'ae_db_best': (dims, gridded[6, :, :], attrs)
        },
        coords={
            'lon': ('lon', glon, {'units': 'degrees_east'}),
            'lat': ('lat', glat, {'units': 'degrees_north'})
        }
    )
    filename_output = date.strftime('%Y%m%d') + '.nc'
    filepath_output = dirpath_output / filename_output
    ds.to_netcdf(str(filepath_output))

def merge_files(dirpath_input, filepath_output):
    '''合并grid_data_one_day函数生成的所有文件.'''
    datasets = []
    for filepath_input in sorted(dirpath_input.iterdir()):
        ds = xr.load_dataset(str(filepath_input))
        ds.coords['time'] = pd.to_datetime(filepath_input.stem)
        datasets.append(ds)
    merged = xr.concat(datasets, dim='time')
    merged.to_netcdf(str(filepath_output))

if __name__ == '__main__':
    time_start = config['time_start']
    time_end = config['time_end']
    dates = pd.date_range(time_start, time_end, freq='D')
    # 只选取春季的数据.
    dates = dates[dates.month.isin([3, 4, 5])]

    # 创建输出目录.
    dirpath_output = Path(config['dirpath_data'], 'MYD')
    helper_tools.renew_dir(dirpath_output)
    dirpath_single = dirpath_output / 'single'
    helper_tools.new_dir(dirpath_single)

    # 对每一天的MYD文件进行格点化.
    p = Pool(10)
    args = [(date, dirpath_single) for date in dates]
    p.starmap(grid_data_one_day, args)
    p.close()
    p.join()

    # 合并前面产生的所有文件.
    merge_files(dirpath_single, dirpath_output / 'data_MYD.nc')
