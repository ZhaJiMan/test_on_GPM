#----------------------------------------------------------------------------
# 2021/07/08
# 为找出的污染个例和清洁个例添加ENV,SLH和ERA5 CAPE数据的文件路径.
#
# 其中ERA5文件对应于降水个例当天.
#----------------------------------------------------------------------------
import json
from pathlib import Path

import pandas as pd

# 读取配置文件,将config作为全局变量.
with open('config.json', 'r') as f:
    config = json.load(f)

def to_ENV_filepath(DPR_filepath):
    '''根据DPR文件的路径获取对应的ENV文件的路径.'''
    DPR_filename = DPR_filepath.name
    parts = DPR_filename.split('.')
    yy = parts[4][:4]

    ENV_dirpath = DPR_filepath.parents[4] / 'ENV' / 'V06'
    parts[0] = '2A-ENV'
    ENV_filename = '.'.join(parts)
    ENV_filepath = ENV_dirpath / yy / ENV_filename

    # 若文件不存在,报错并提示.
    if not ENV_filepath.exists():
        raise FileNotFoundError(str(ENV_filepath))

    return ENV_filepath

def to_SLH_filepath(DPR_filepath):
    '''根据DPR文件的路径获取对应的SLH文件的路径.'''
    DPR_filename = DPR_filepath.name
    parts = DPR_filename.split('.')
    yy = parts[4][:4]

    # 两个服务器上的SLH文件的版本存在差异,需要分开处理.
    GPM_dirpath = DPR_filepath.parents[4]
    if str(GPM_dirpath) == '/data00/0/GPM':
        parts[-2] = 'V06A'
    elif str(GPM_dirpath)  == '/data04/0/gpm':
        parts[-2] = 'V06B'

    parts[3] = 'GPM-SLH'
    SLH_filename = '.'.join(parts)
    SLH_filepath = GPM_dirpath / 'SLH' / 'V06' / yy / SLH_filename

    # 若文件不存在,报错并提示.
    if not SLH_filepath.exists():
        raise FileNotFoundError(str(SLH_filepath))

    return SLH_filepath

def get_ERA5_filepath(time):
    '''根据时间的日期找到对应的ERA5 CAPE文件.'''
    ERA5_dirpath = Path('/data04/0/ERA5_NANJING/ERA-new/convective_available_potential_energy')
    date_str = time.strftime('%Y%m%d')
    ERA5_filepath = ERA5_dirpath / f'era5.convective_available_potential_energy.{date_str}.nc'

    # 若文件不存在,报错并提示.
    if not ERA5_filepath.exists():
        raise FileNotFoundError(str(ERA5_filepath))

    return ERA5_filepath

if __name__ == '__main__':
    result_dirpath = Path(config['result_dirpath'])
    with open(str(result_dirpath / 'found_cases.json'), 'r') as f:
        records = json.load(f)
    dusty_cases = records['dusty']['cases']
    clean_cases = records['clean']['cases']

    # 为每个个例添加新文件路径.
    for case in dusty_cases + clean_cases:
        DPR_filepath = Path(case['DPR_filepath'])
        ENV_filepath = to_ENV_filepath(DPR_filepath)
        SLH_filepath = to_SLH_filepath(DPR_filepath)

        rain_time = pd.to_datetime(case['rain_time'])
        ERA5_filepath = get_ERA5_filepath(rain_time)

        case['ENV_filepath'] = str(ENV_filepath)
        case['SLH_filepath'] = str(SLH_filepath)
        case['ERA5_filepath'] = str(ERA5_filepath)

    # 重新保存修改后的case.
    with open(str(result_dirpath / 'found_cases.json'), 'w') as f:
        json.dump(records, f, indent=4)
