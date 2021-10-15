#----------------------------------------------------------------------------
# 2021/05/08
# 对每一个降水个例,若存在时间和空间上与之相近的CALIPSO VFM产品,则认为该个例
# 是一个"匹配"个例,将这样的个例记录下来.
#
# 寻找匹配个例的算法为:
# - 对一个降水个例,读取降水时间前后各一天范围内的VFM数据.
# - 若VFM轨道里降水中心的经纬度距离小于给定的阈值,视作空间相近.
# - 若VFM轨道的平均时间与降水时间的差值小于给定的阈值,视作时间相近.
# - 存在这样VFM数据的降水个例即匹配个例,同时将其对应的VFM文件路径记录下来.
#
# 注:
# - 一个匹配个例可能对应多个CALIPSO文件.
# - CALIPSO VFM原始数据的空间范围为[70, 140, 0, 60],所以不需要根据空间截取.
#----------------------------------------------------------------------------
import json
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

from data_reader import Reader_for_CAL

# 读取配置文件,将config作为全局变量.
with open('config.json', 'r') as f:
    config = json.load(f)

def get_CAL_filepaths(date):
    '''根据年月日,返回文件名中含有的时间与之匹配的CAL文件路径.'''
    yy = date.strftime('%Y')
    yymm = date.strftime('%Y%m')
    yymmdd = date.strftime('%Y-%m-%d')

    # 搜索对应的文件.
    CAL_dirpath = Path('/d4/wangj/dust_precipitation/data/CALIPSO')
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
    为一个降水个例寻找匹配的CALIPSO轨道.

    返回添加了CALIPSO文件路径的case记录(可能为空列表).
    '''
    rain_time = pd.to_datetime(case['rain_time'])
    rain_center = case['rain_center']
    matched_filepaths = []
    # 读取降水个例前后共三天的CALIPSO数据.
    dt = pd.Timedelta(days=1)
    for date in [rain_time - dt, rain_time, rain_time + dt]:
        for CAL_filepath in get_CAL_filepaths(date):
            f = Reader_for_CAL(str(CAL_filepath))
            track = f.read_lonlat()
            CAL_time = f.read_time().mean()
            f.close()

            # 若时间与空间都接近,那么认为匹配.
            if (
                is_space_near(track, rain_center) and
                is_time_near(CAL_time, rain_time)
            ):
                matched_filepaths.append(str(CAL_filepath))
            else:
                continue

    # 返回修改后的case.
    case_new = case.copy()
    case_new['CAL_filepaths'] = matched_filepaths

    return case_new

if __name__ == '__main__':
    # 读取降水个例.
    result_dirpath = Path(config['result_dirpath'])
    input_filepath = result_dirpath / 'rain_cases.json'
    with open(str(input_filepath), 'r') as f:
        cases = json.load(f)

    # 寻找能与CALIPSO文件匹配的降水个例.
    p = Pool(8)
    cases = p.map(process_one_case, cases)
    p.close()
    p.join()
    # 过滤掉没有匹配文件的个例.
    cases = [case for case in cases if case['CAL_filepaths']]

    # 打印找到的个例数.
    print(f'{len(cases)} matched cases found')

    # 把找到的结果写到json文件中.
    output_filepath = result_dirpath / 'matched_cases.json'
    with open(str(output_filepath), 'w') as f:
        json.dump(cases, f, indent=4)
