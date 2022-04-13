'''
2022-04-12
向每个个例写入MYD和MERRA2的AOD信息.

流程:
- 选取降水当天的格点化后的MYD文件, 以降水中心为中心的方框计算平均AOD.
- 选取降水时刻前的MERRA2文件, 以类似的方法计算平均AOD.
- 将MYD和MERRA2的信息写入到个例记录当中. 对每个个例进行同样的操作.

输入:
- cases_rain.json文件.
- 00目录里制作的MYD和MERRA2文件.

输出:
- 写入了MYD和MERRA2记录的个例json文件, 其中MYD记录包括:
  - MYD文件路径
  - combined AOD
  - DB AE
  MERRA2记录包括:
  - MERRA2文件路径
  - total AOD
  - dust AOD

参数:
- AOD_TIME_ADVANCE: MERRA2文件的时间提前多少小时.
- AOD_BOX_LENGTH: 为降水个例计算区域平均AOD时方框的边长.
'''
import json
from pathlib import Path
import sys
sys.path.append('../modules')

import numpy as np
import pandas as pd
import xarray as xr

# 读取配置文件, 将config作为全局变量.
with open('config.json') as f:
    config = json.load(f)
# 其它全局参数.
AOD_TIME_ADVANCE = 2
AOD_BOX_LENGTH = 10

def add_records_one_case(case):
    '''向个例写入MYD和MERRA2的AOD记录, 返回写入了记录的case的拷贝.'''
    dirpath_data = Path(config['dirpath_data'])
    dirpath_MYD = dirpath_data / 'MYD' / 'single'
    dirpath_MERRA2 = dirpath_data / 'MERRA2' / 'single'

    # 根据降水时间选取MYD和MERRA2文件.
    time_rain = pd.to_datetime(case['rain_time'])
    time_advance = time_rain - pd.Timedelta(hours=AOD_TIME_ADVANCE)
    filepath_MYD = dirpath_MYD / time_rain.strftime('%Y%m%d.nc')
    filepath_MERRA2 = dirpath_MERRA2 / time_advance.strftime('%Y%m%d.nc')

    # 设置选取AOD范围的经纬度方框.
    clon, clat = case['rain_center']
    half = AOD_BOX_LENGTH / 2
    lonmin = clon - half
    lonmax = clon + half
    latmin = clat - half
    latmax = clat + half

    # 计算MYD记录.
    record_MYD = {}
    if filepath_MYD.exists():
        with xr.open_dataset(str(filepath_MYD)) as ds:
            ds = ds.sel(
                lon=slice(lonmin, lonmax),
                lat=slice(latmin, latmax)
            )
            aod_combined = ds.aod_combined.mean()
            ae_db = ds.ae_db_best.mean()
        record_MYD['filepath_MYD'] = str(filepath_MYD)
        record_MYD['aod_combined'] = float(aod_combined)
        record_MYD['ae_db'] = float(ae_db)

    # 计算MERRA2记录.
    record_MERRA2 = {}
    if filepath_MERRA2.exists():
        with xr.open_dataset(str(filepath_MERRA2)) as ds:
            ds = ds.sel(
                lon=slice(lonmin, lonmax),
                lat=slice(latmin, latmax)
            ).sel(time=time_advance, method='nearest')
            aod_total = ds.aod_total.mean()
            aod_dust = ds.aod_dust.mean()
        record_MERRA2['filepath_MERRA2'] = str(filepath_MERRA2)
        record_MERRA2['aod_total'] = float(aod_total)
        record_MERRA2['aod_dust'] = float(aod_dust)

    # 写入到个例中.
    case = case.copy()
    case['record_MYD'] = record_MYD
    case['record_MERRA2'] = record_MERRA2

    return case

if __name__ == '__main__':
    # 读取降水个例.
    filepath_input = Path(config['dirpath_result'], 'cases_rain.json')
    with open(str(filepath_input)) as f:
        cases = json.load(f)

    # 为每个个例添加信息.
    cases = [add_records_one_case(case) for case in cases]

    # 重新写成json文件.
    with open(str(filepath_input), 'w') as f:
        json.dump(cases, f, indent=4)
