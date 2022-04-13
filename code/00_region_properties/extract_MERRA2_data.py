'''
2022-02-10
给定地图范围和时间范围, 提取MERRA2文件中的AOD值, 保存为nc文件.

流程:
- 读取某天对应的MERRA2文件, 截取地图范围内的AOD数据. 将AOD变量名改成更简单
  的形式, 然后加和计算细模态AOD和粗模态AOD. 将结果保存到单个文件中.
  对每一天进行同样的操作.
- 为方便统计, 沿时间维连接所有文件生成单个文件.

输入:
- 服务器上的MERRA2文件.

输出:
- 以日期命名的nc文件, 含有的变量包括:
  - total AOD
  - fine AOD
  - coarse AOD
  - black carbon AOD
  - dust AOD
  - organic carbon AOD
  - sea salt AOD
  - sulfate AOD
- 上一条中所有文件连接起来的单个nc文件.

参数:
- extents_map: 地图范围.

注意:
- 时间范围为春季(3~5月), 硬编码在脚本中.
- 合并版的文件里时间维不仅不连续, 还是逐小时的.
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

import helper_tools
import data_tools

# 读取配置文件, 作为全局变量.
with open('config.json') as f:
    config = json.load(f)

def extract_data_from_one_file(filepath_MERRA2, dirpath_output):
    '''
    提取一个MERRA2文件落入研究区域内的AOD数据并保存.

    输出文件以日期命名, 保存为nc格式.
    '''
    extents_map = config['extents_map']
    lonmin, lonmax, latmin, latmax = extents_map

    # 截取需要的数据并更改变量名.
    varname_dict = {
        'TOTEXTTAU': 'aod_total',
        'BCEXTTAU': 'aod_black_carbon',
        'DUEXTTAU': 'aod_dust',
        'OCEXTTAU': 'aod_organic_carbon',
        'SSEXTTAU': 'aod_sea_salt',
        'SUEXTTAU': 'aod_sulfate'
    }
    with xr.open_dataset(str(filepath_MERRA2)) as ds:
        ds = ds.sel(
            lon=slice(lonmin, lonmax),
            lat=slice(latmin, latmax)
        )
        ds = ds[list(varname_dict)]
        ds = ds.rename(varname_dict)
        ds.load()

    # 计算两种模态的AOD.
    ds['aod_fine'] = (
        ds.aod_black_carbon + ds.aod_organic_carbon + ds.aod_sulfate
    )
    ds['aod_coarse'] = ds.aod_dust + ds.aod_sea_salt

    # 设置元数据.
    for var in ds.data_vars.values():
        var.attrs = {'units': 'none'}
        var.encoding.clear()
    for coord in ds.coords.values():
        if coord.name == 'time':
            coord.attrs.clear()
        elif coord.name == 'lat':
            coord.attrs = {'units': 'degrees_north'}
        elif coord.name == 'lon':
            coord.attrs = {'units': 'degrees_east'}
        coord.encoding.clear()
    ds.attrs.clear()

    # 保存为nc文件.
    filename_output = filepath_MERRA2.stem.split('.')[-1] + '.nc'
    filepath_output = dirpath_output / filename_output
    ds.to_netcdf(str(filepath_output))

def merge_files(dirpath_input, filepath_output):
    '''合并extract_data_from_one_file函数生成的所有文件.'''
    datasets = []
    for filepath_input in sorted(dirpath_input.iterdir()):
        ds = xr.load_dataset(str(filepath_input))
        datasets.append(ds)
    merged = xr.concat(datasets, dim='time')
    merged.to_netcdf(str(filepath_output))

if __name__ == '__main__':
    # 设置时间.
    time_start = config['time_start']
    time_end = config['time_end']
    dates = pd.date_range(time_start, time_end, freq='D')
    # 只选取春季的数据.
    dates = dates[dates.month.isin([3, 4, 5])]

    # 创建输出目录.
    dirpath_output = Path(config['dirpath_data'], 'MERRA2')
    helper_tools.renew_dir(dirpath_output)
    dirpath_single = dirpath_output / 'single'
    helper_tools.new_dir(dirpath_single)

    # 从每一天的MERRA2文件中提取数据.
    p = Pool(10)
    for date in dates:
        filepath_MERRA2 = data_tools.get_MERRA2_filepath_one_day(date)
        p.apply_async(
            extract_data_from_one_file,
            args=(filepath_MERRA2, dirpath_single)
        )
    p.close()
    p.join()

    # 合并前面产生的所有文件.
    merge_files(dirpath_single, dirpath_output / 'data_MERRA2.nc')
