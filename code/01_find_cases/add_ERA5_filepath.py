#----------------------------------------------------------------------------
# 2021/07/08
# 为找出的污染个例和清洁个例添加含有ERA5 CAPE数据的文件路径.
#----------------------------------------------------------------------------
import json
from pathlib import Path

import pandas as pd

# 读取配置文件,作为全局变量.
with open('config.json', 'r') as f:
    config = json.load(f)

if __name__ == '__main__':
    input_filepath = Path(config['result_dirpath']) / 'found_cases.json'
    with open(str(input_filepath), 'r') as f:
        records = json.load(f)
    dusty_cases = records['dusty']['cases']
    clean_cases = records['clean']['cases']

    # 根据降水事件发生的日期来设置ERA5文件的路径.
    # 额外检查ERA5文件是否存在,否则报错.
    ERA5_dirpath = Path('/data04/0/ERA5_NANJING/ERA-new/convective_available_potential_energy')
    for case in dusty_cases + clean_cases:
        rain_time = pd.to_datetime(case['rain_time'])
        date_str = rain_time.strftime('%Y%m%d')
        ERA5_filepath = ERA5_dirpath / f'era5.convective_available_potential_energy.{date_str}.nc'
        if ERA5_filepath.exists():
            case['ERA5_filepath'] = str(ERA5_filepath)
        else:
            raise ValueError(f'File for case on {date_str} does not exist!')

    # 重新保存修改后的case.
    with open(str(input_filepath), 'w') as f:
        json.dump(records, f, indent=4)
