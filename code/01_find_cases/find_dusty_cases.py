#----------------------------------------------------------------------------
# 2021/05/08
# 从匹配了CALIPSO VFM轨道的降水事件中找出沙尘污染个例和清洁个例.
#
# 寻找的算法为:对每个降水事件匹配的VFM轨道来说,若降水中心上下一定纬度范围内的
# VFM数据中沙尘气溶胶的占比高于给定的阈值,则标记为high;若低于给定的另一个阈值,
# 则标记为low.如果一个降水事件对应的VFM中存在high,则记为污染个例;若VFM全为
# low,则记为清洁个例.
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

def dust_level(vfm):
    '''根据vfm中沙尘像元的占比,判断沙尘水平.'''
    d0, d1 = config['DUST_RATIOS']
    dust_ratio = np.count_nonzero(vfm == 8) / vfm.size * 100
    if dust_ratio <= d0:
        return 'low'
    elif dust_ratio >= d1:
        return 'high'
    else:
        return 'med'

def get_CAL_flag(track, center):
    '''
    以降水事件为中心,CAL_width为纬度范围,
    获取落入其中的CALIPSO数据的Boolean数组.
    '''
    CAL_width = config['CAL_width']
    lon, lat = track
    clon, clat = center
    dlat = CAL_width / 2
    flag = (lat >= (clat - dlat)) & (lat <= (clat + dlat))

    return flag

def process_one_case(case):
    '''读取一个降水事件对应的CALIPSO文件,并据此判断事件的污染程度.'''
    rain_center = case['rain_center']
    levels = []
    for CAL_filepath in case['CAL_filepaths']:
        f = reader_for_CAL(CAL_filepath)
        track = f.read_lonlat()
        vfm = f.read_vfm()
        f.close()

        scan_flag = get_CAL_flag(track, rain_center)
        vfm = vfm[scan_flag, :]
        levels.append(dust_level(vfm))

    # 若存在high level的VFM,那么判定为dusty case.
    # 若VFM都是low level,那么判定为clean case.
    if 'high' in levels:
        return 'dusty'
    elif levels.count('low') == len(levels):
        return 'clean'
    else:
        return None

if __name__ == '__main__':
    result_dirpath = Path(config['result_dirpath'])
    input_filepath = result_dirpath / 'rain_events_matched.json'
    with open(str(input_filepath), 'r') as f:
        cases = json.load(f)

    p = Pool(16)
    results = p.map(process_one_case, cases)
    p.close()
    p.join()
    # 寻找dusty和clean个例.
    dusty_cases = []
    clean_cases = []
    for case, result in zip(cases, results):
        if result == 'dusty':
            dusty_cases.append(case)
        elif result == 'clean':
            clean_cases.append(case)
        else:
            continue

    # 将结果输出到json文件中.
    records = {
        'dusty': {
            'number': len(dusty_cases),
            'cases': dusty_cases
        },
        'clean': {
            'number': len(clean_cases),
            'cases': clean_cases
        }
    }
    output_filepath = result_dirpath / 'found_cases.json'
    with open(str(output_filepath), 'w') as f:
        json.dump(records, f, indent=4)
