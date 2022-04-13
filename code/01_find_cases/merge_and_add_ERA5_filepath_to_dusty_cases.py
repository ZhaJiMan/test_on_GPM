'''
2022-04-12
为每个污染和清洁个例添加降水时刻的ERA5文件路径.

流程:
- 搜索个例当天的所有ERA5文件(具体变量见代码), 截取降水时刻的数据,
  然后将不同文件的数据合并到单个文件中.
- 将合并文件的路径写入到个例记录中.

输入:
- cases_dusty.json
- cases_clean.json

输出:
- 合并后的ERA5文件
- cases_dusty.json
- cases_clean.json
'''
import json
from pathlib import Path
import sys
sys.path.append('../modules')

import xarray as xr
import pandas as pd

import data_tools
import helper_tools

# 读取配置文件, 将config作为全局变量.
with open('config.json') as f:
    config = json.load(f)

if __name__ == '__main__':
    # 读取两组个例.
    dirpath_result = Path(config['dirpath_result'])
    filepath_dusty = dirpath_result / 'cases_dusty.json'
    filepath_clean = dirpath_result / 'cases_clean.json'
    with open(str(filepath_dusty)) as f:
        cases_dusty = json.load(f)
    with open(str(filepath_clean)) as f:
        cases_clean = json.load(f)

    # 创建输出目录.
    dirpath_output = Path(config['dirpath_data'], 'ERA5')
    helper_tools.new_dir(dirpath_output)

    # 需要的变量.
    varnames = [
        'u_component_of_wind',
        'v_component_of_wind',
        'vertical_velocity',
        'specific_humidity',
        'convective_available_potential_energy'
    ]

    # 根据个例时间寻找对应的ERA5文件, 并合并成单个文件保存.
    for case in cases_dusty + cases_clean:
        rain_time = pd.to_datetime(case['rain_time'])
        time_str = rain_time.strftime('%Y%m%d_%H%M')
        filename_output = f'era5.{time_str}.nc'
        filepath_output = dirpath_output / filename_output
        case['filepath_ERA5'] = str(filepath_output)
        # 避免重复创建文件.
        if filepath_output.exists():
            continue

        # 合并varnames对应的Dataset.
        datasets = []
        for varname in varnames:
            filepath_ERA5 = data_tools.get_ERA5_filepath_one_day(
                rain_time, varname
            )
            ds = xr.load_dataset(str(filepath_ERA5))
            # 为节省存储空间, 仅选取降水时刻的数据.
            ds = ds.sel(time=rain_time, method='nearest')
            datasets.append(ds)
        merged = xr.merge(datasets)
        merged = merged.sortby('latitude')  # 颠倒纬度.
        merged.to_netcdf(str(filepath_output))

    # 重新写成json文件.
    with open(str(filepath_dusty), 'w') as f:
        json.dump(cases_dusty, f, indent=4)
    with open(str(filepath_clean), 'w') as f:
        json.dump(cases_clean, f, indent=4)

