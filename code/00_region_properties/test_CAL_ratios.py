'''
2022-02-10
提取落入研究区域内的所有VFM剖面, 计算气溶胶像元占比和沙尘像元占比.
用于测试像元占比的数值分布.
'''
import json
from pathlib import Path
from multiprocessing import Pool
import sys
sys.path.append('../modules')

import numpy as np
import pandas as pd

import data_tools
import region_tools

# 读取配置文件, 将config作为全局变量.
with open('config.json') as f:
    config = json.load(f)

def calc_ratios_from_one_file(filepath_CAL):
    extents_DPR = config['extents_DPR']
    f = data_tools.ReaderCAL(str(filepath_CAL))
    lon, lat = f.read_lonlat()
    # 若没有轨道数据落入extents_DPR, 则结束读取.
    mask_scan = region_tools.region_mask(lon, lat, extents_DPR)
    if not mask_scan.any():
        f.close()
        return None
    # 继续读取feature type数据.
    hgt = f.read_hgt()
    ftype = f.read_ftype()
    ftype = data_tools.get_ftype_with_dust(ftype)
    f.close()
    # 截取数据.
    ftype = ftype[mask_scan, :]
    ftype = ftype[:, hgt <= 15]

    # 计算dust ratio和aerosol ratio.
    num_all = ftype.size
    num_cloud = np.count_nonzero(ftype == 2)
    num_dust = np.count_nonzero(ftype == 8)
    num_aerosol = np.count_nonzero(ftype == 3) + num_dust
    ratio_cloud = num_cloud / num_all * 100
    ratio_dust = num_dust / num_all * 100
    ratio_aerosol = num_aerosol / num_all * 100

    if ratio_cloud > 20:
        return ratio_dust, ratio_aerosol
    else:
        return None

ratios = []
def collect_results(result):
    global ratios
    if result is not None:
        ratios.append(result)

if __name__ == '__main__':
    # 设置时间.
    time_start = config['time_start']
    time_end = config['time_end']
    dates = pd.date_range(time_start, time_end, freq='D')
    # 只选取春季的数据.
    dates = dates[dates.month.isin([3, 4, 5])]

    p = Pool(10)
    for date in dates:
        for filepath_CAL in data_tools.get_CAL_filepaths_one_day(date):
            p.apply_async(
                calc_ratios_from_one_file,
                args=(filepath_CAL,), callback=collect_results
            )
    p.close()
    p.join()

    ratios = np.array(ratios).T
    ratio_dust, ratio_aerosol = ratios
    q = [0.25, 0.5, 0.75]
    print('Dust Ratio:', np.quantile(ratio_dust, q))
    print('Aerosol Ratio:', np.quantile(ratio_aerosol, q))
