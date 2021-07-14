#----------------------------------------------------------------------------
# 2021/05/08
# 对于每一个降水事件,找出在时间与空间上与之相近的CALIPSO VFM轨道,
# 并将对应的文件路径写入到事件记录中.
#
# 寻找匹配轨道的算法为:对一个降水事件,读取当天前后各一天共三天的VFM数据,
# 若VFM轨道离降水事件中心的经纬度距离小于给定的阈值,视作空间相近;若轨道
# 的平均时间与降水时间的差值小于给定的阈值,视作时间相近.将符合条件的
# 所有VFM文件路径都记录下来.
#
# 注:CALIPSO VFM原始数据的空间范围为[70, 140, 0, 60].
#----------------------------------------------------------------------------
import json
from pathlib import Path
from multiprocessing import Pool
import sys
sys.path.append('../modules')
from data_reader import reader_for_CAL

import numpy as np
import pandas as pd

# 读取配置文件,将config作为全局变量.
with open('config.json', 'r') as f:
    config = json.load(f)

def get_CAL_filepath(date):
    '''根据年月日,返回文件名中含有的时间与之匹配的CAL文件路径.'''
    yy = date.strftime('%Y')
    yymm = date.strftime('%Y%m')
    yymmdd = date.strftime('%Y-%m-%d')

    CAL_dirpath = Path('/d4/wangj/dust_precipitation/data/CALIPSO')
    # 搜索对应的文件.
    for CAL_filepath in (CAL_dirpath / yy / yymm).glob(
        f'*{yymmdd}*.hdf'):
        yield CAL_filepath

def is_time_near(t1, t2):
    '''比较两个卫星的过境时间的差值是否小于TIME_DIFF.'''
    TIME_DIFF = pd.Timedelta(config['TIME_DIFF'], unit='h')
    dt = abs(t1 - t2)
    if dt <= TIME_DIFF:
        return True
    else:
        return False

def is_space_near(track, center):
    '''判断CALIPSO的轨道track与降水的中心是否邻近.'''
    SPACE_DIFF = config['SPACE_DIFF']
    lon, lat = track
    clon, clat = center
    dist = np.sqrt((lon - clon)**2 + (lat - clat)**2)
    if dist.min() <= SPACE_DIFF:
        return True
    else:
        return False

def process_one_case(case):
    '''
    为一个降水事件寻找匹配的CALIPSO轨道.若匹配成功,返回添加了CALIPSO文件
    路径的case;若失败,返回None.
    '''
    rain_time = pd.to_datetime(case['rain_time'])
    rain_center = case['rain_center']
    dt = pd.Timedelta(days=1)
    matched_filepaths = []
    # 读取降水事件前后共三天的CALIPSO数据.
    for date in [rain_time - dt, rain_time, rain_time + dt]:
        for CAL_filepath in get_CAL_filepath(date):
            f = reader_for_CAL(str(CAL_filepath))
            track = f.read_lonlat()
            CAL_time = f.read_time().mean()
            f.close()

            # 若时间与空间都接近,那么认为匹配.
            if is_space_near(track, rain_center) and \
                is_time_near(CAL_time, rain_time):
                matched_filepaths.append(str(CAL_filepath))
            else:
                continue

    # 若有CALIPSO轨道与降水事件匹配,返回添加了CALIPSO文件路径的case.
    if len(matched_filepaths) > 0:
        case_new = case.copy()
        case_new['CAL_filepaths'] = matched_filepaths
        return case_new
    else:
        return None

if __name__ == '__main__':
    # 读取降水事件.
    result_dirpath = Path(config['result_dirpath'])
    input_filepath = result_dirpath / 'rain_events.json'
    with open(str(input_filepath), 'r') as f:
        cases = json.load(f)

    # 寻找能与CALIPSO文件匹配的降水事件.
    p = Pool(16)
    results = p.map(process_one_case, cases)
    p.close()
    p.join()
    results = [r for r in results if r is not None]

    # 把找到的结果写到json文件中.
    output_filepath = result_dirpath / 'rain_events_matched.json'
    with open(str(output_filepath), 'w') as f:
        json.dump(results, f, indent=4)
