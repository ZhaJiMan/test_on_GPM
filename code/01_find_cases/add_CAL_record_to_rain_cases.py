'''
2022-04-12
向每个个例写入匹配的CALIPSO VFM的信息.

流程:
- 读取降水个例前后三天的VFM数据, 截取地图范围内, 个例中心上下一定范围的
  纬度条带内的VFM剖面, 以剖面的平均经度减去个例中心的经度作为dx, 剖面的平均
  过境时间减去降水时间作为dt, 当dx和dt满足一定范围时记作与降水个例匹配.
- 计算截取的剖面中气溶胶像元和沙尘像元的占比, 作为记录保存到个例中.
- 当一个个例能匹配多条剖面时, 优先取dx的绝对值最小的剖面.
  当个例匹配不到剖面时, 将空记录(空字典)保存到个例中.

输入:
- cases_rain.json文件.
- data目录下的CALIPSO VFM文件.

输出:
- 写入了CALIPSO记录的个例json文件. 记录包含:
  - CAL文件路径
  - dust ratio
  - aerosol ratio

参数:
- extents_map: 地图范围.
- LAT_LENGTH: 截取VFM剖面时取降水中心上下共LAT_LENGTH度的纬向条带.
- DT0, DT1: 要求DT0<dt<DT1.
- DX0, DX1: 要求DX0<dx<DX1.

注意:
- 脚本使用了多进程.
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
# 其它全局参数.
LAT_LENGTH = 10
DT0, DT1 = -6, 3
DX0, DX1 = -6, 3

def add_record_one_case(case):
    # 读取参数.
    lonmin, lonmax, _, _ = config['extents_map']
    rain_time = pd.to_datetime(case['rain_time'])
    clon, clat = case['rain_center']

    # 设置截取CAL轨道数据的范围.
    half = LAT_LENGTH / 2
    latmin = clat - half
    latmax = clat + half
    extents_scan = [lonmin, lonmax, latmin, latmax]

    # 读取降水个例前后共三天的CAL数据.
    records = []
    day = pd.Timedelta(days=1)
    hour = pd.Timedelta(hours=1)
    for date in [rain_time - day, rain_time, rain_time + day]:
        for filepath_CAL in data_tools.get_CAL_filepaths_one_day(date):
            # 分阶段读取数据, 先读取经纬度和时间.
            f = data_tools.ReaderCAL(str(filepath_CAL))
            lon, lat= f.read_lonlat()
            time = f.read_time()

            # 若没有轨道数据落入extents_scan, 则结束读取.
            mask_scan = region_tools.region_mask(lon, lat, extents_scan)
            if not mask_scan.any():
                f.close()
                continue
            # 计算截取部分的轨道与降水中心的经度差和时间差.
            dx = lon[mask_scan].mean() - clon
            dt = time[mask_scan].mean() - rain_time
            dt = dt / hour  # 换算成小时.
            # 若dx和dt超出范围, 则结束读取.
            cond = (dx > DX0) and (dx < DX1) and (dt > DT0) and (dt < DT1)
            if not cond:
                f.close()
                continue

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
            num_dust = np.count_nonzero(ftype == 8)
            num_aerosol = np.count_nonzero(ftype == 3) + num_dust
            ratio_dust = num_dust / num_all * 100
            ratio_aerosol = num_aerosol / num_all * 100

            # 记录从CAL文件相关的信息.
            record = {
                'filepath_CAL': str(filepath_CAL),
                'dx': float(dx),
                'dt': float(dt),
                'ratio_dust': float(ratio_dust),
                'ratio_aerosol': float(ratio_aerosol)
            }
            records.append(record)

    # 优先选取dx绝对值最小的记录.
    if records:
        dx = np.array([record['dx'] for record in records])
        ind = np.abs(dx).argmin()
        record = records[ind]
    else:
        record = {}

    # 写入到个例中.
    case = case.copy()
    case['record_CAL'] = record

    return case

if __name__ == '__main__':
    # 读取降水个例.
    filepath_input = Path(config['dirpath_result'], 'cases_rain.json')
    with open(str(filepath_input)) as f:
        cases = json.load(f)

    # 为每个个例添加信息.
    p = Pool(10)
    cases = p.map(add_record_one_case, cases)
    p.close()
    p.join()

    # 打印匹配到记录的个例数.
    n = 0
    for case in cases:
        if case['record_CAL']:
            n += 1
    print(f'{n} rain cases matched')

    # 重新写成json文件.
    with open(str(filepath_input), 'w') as f:
        json.dump(cases, f, indent=4)
