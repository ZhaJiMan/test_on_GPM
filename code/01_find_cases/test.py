import json
from pathlib import Path
from multiprocessing import Pool
import sys
sys.path.append('../modules')
from region_funcs import get_extent_flag_both
from data_reader import read_DPR_time

import h5py
import numpy as np
import pandas as pd

# 读取配置文件,将config作为全局变量.
with open('config.json', 'r') as f:
    config = json.load(f)

def is_rain_enough(flagPre):
    '''
    判断flagPre中有降水的点是否够多.
    flagPre的取值为[0, 1, 10, 11],其中0表示无降水.
    '''
    RAIN_PIXEL_NUM = config['RAIN_PIXEL_NUM']
    if np.count_nonzero(flagPre > 0) >= RAIN_PIXEL_NUM:
        return True
    else:
        return False

def calc_rain_center(lon1D, lat1D, flagPre):
    '''
    计算降水的大致中心.
    方法是计算有降水的点的平均经纬度.
    '''
    flag = flagPre > 0
    # 将np.float32转为原生的float,否则无法导入到json中.
    lon_avg = float(lon1D[flag].mean())
    lat_avg = float(lat1D[flag].mean())

    return lon_avg, lat_avg

def to_ENV_filepath(DPR_filepath):
    '''根据DPR文件的路径获取对应的ENV文件的路径.'''
    DPR_filename = DPR_filepath.name
    parts = DPR_filename.split('.')
    yy = parts[4][:4]

    ENV_dirpath = DPR_filepath.parents[4] / 'ENV' / 'V06'
    parts[0] = '2A-ENV'
    ENV_filename = '.'.join(parts)
    ENV_filepath = ENV_dirpath / yy / ENV_filename

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

    return SLH_filepath

results = []
def collect_results(result):
    '''收集apply_async的结果,并筛除None.'''
    global results
    if result is not None:
        results.append(result)

def process_one_file(DPR_filepath):
    '''
    处理一个DPR文件.若文件在DPR_extent区域内有足够多的降水像元,
    返回文件路径,大致的降水时间和位置.否则返回None.
    '''
    DPR_extent = config['DPR_extent']
    f = h5py.File(str(DPR_filepath), 'r')
    lon2D = f['NS/Longitude'][:]
    lat2D = f['NS/Latitude'][:]

    extent_flag = get_extent_flag_both(lon2D, lat2D, DPR_extent)
    # 如果没有点落入DPR_extent中,返回None.
    if np.count_nonzero(extent_flag) == 0:
        f.close()
        return None

    # 如果降水点数太少,那么返回None.
    lon1D = lon2D[extent_flag]
    lat1D = lat2D[extent_flag]
    flagPre = f['NS/PRE/flagPrecip'][:][extent_flag]
    if not is_rain_enough(flagPre):
        f.close()
        return None

    # 读取降水事件的平均时间与位置.
    time = read_DPR_time(f)
    scan_flag = np.any(extent_flag, axis=1)
    rain_time = time[scan_flag].mean().strftime('%Y-%m-%d %H:%M')
    rain_center = calc_rain_center(lon1D, lat1D, flagPre)
    f.close()

    # 以字典形式存储.
    case = {
        'DPR_filepath': str(DPR_filepath),
        'ENV_filepath': str(to_ENV_filepath(DPR_filepath)),
        'SLH_filepath': str(to_SLH_filepath(DPR_filepath)),
        'rain_time': rain_time,
        'rain_center': rain_center
    }

    return case

def get_DPR_filepath(date):
    '''根据年月日,返回文件名中含有的时间与之匹配的DPR文件路径.'''
    yy = date.strftime('%Y')
    yymm = date.strftime('%Y%m')
    yymmdd = date.strftime('%Y%m%d')

    # DPR数据在服务器上是分两个位置存储的,需要分开处理.
    if date < pd.to_datetime('2019-05-30'):
        DPR_dirpath = Path('/data00/0/GPM/DPR/V06')
    else:
        DPR_dirpath = Path('/data04/0/gpm/DPR/V06')

    # 搜索对应的文件.
    for DPR_filepath in (DPR_dirpath / yy / yymm).glob(
        f'*{yymmdd}-S*.HDF5'):
        yield DPR_filepath

if __name__ == '__main__':
    start_time = config['start_time']
    end_time = config['end_time']
    dates = pd.date_range(start_time, end_time, freq='D')
    # 只选取春季的数据.
    dates = dates[(dates.month >= 3) & (dates.month <= 5)]

    class Found(Exception):
        pass

    # 从所有DPR文件中寻找降水事件.
    try:
        for date in dates:
            for DPR_filepath in get_DPR_filepath(date):
                case = process_one_file(DPR_filepath)
                if case is not None:
                    raise Found
    except Found:
        print(case)
