#-----------------------------------------------------------------------------
# 2021/07/27
# 从GPM DPR文件中找出华东地区发生的降水个例.
#
# 时间范围,空间范围,以及降水像元阈值等参数都由配置文件config.json给出.
# 程序中只选取了春季(3~5月)的数据,根据需求不同可以在程序中修改.
#
# 具体流程为:
# - 依次读取每一天的DPR文件,判断是否有降水像元落入DPR_extent中.
# - 接着在比DPR_extent范围稍大的map_extent中利用改进后的连通域标记算法标记出
#   每个降水个例,若降水个例像元够多,海面上的部分够小,降水中心落在DPR_extent内,
#   则记为华东的一个降水个例.
# - 记录该个例的编号,对应的DPR文件,标记的下标,降水时间和中心位置.
#-----------------------------------------------------------------------------
import json
from pathlib import Path
from multiprocessing import Pool
import sys
sys.path.append('../modules')

import h5py
import numpy as np
import pandas as pd
import xarray as xr

from region_funcs import region_mask
from helper_funcs import recreate_dir
from label_funcs import two_pass
from data_reader import read_GPM_time

def get_DPR_filepaths(date):
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
    sub_dirpath = DPR_dirpath / yy / yymm
    for DPR_filepath in sub_dirpath.glob(f'*{yymmdd}-S*.HDF5'):
        yield DPR_filepath

# 读取配置文件,将config作为全局变量.
with open('config.json', 'r') as f:
    config = json.load(f)

def find_cases_from_one_file(DPR_filepath, output_dirpath):
    '''
    在一个DPR文件中寻找DPR_extent范围内的降水个例.

    若没找到,返回空列表;若找到至少一个个例,返回存有每个个例信息的列表,
    同时会把个例的下标Boolean数组以NC文件的形式保存到output_dirpath目录下.
    '''
    # 读取参数.
    DPR_extent = config['DPR_extent']
    map_extent = config['map_extent']
    RAIN_RADIUS = config['RAIN_RADIUS']
    RAIN_PIXEL_NUM = config['RAIN_PIXEL_NUM']
    OCEAN_RATIO = config['OCEAN_RATIO']
    orbit_number = DPR_filepath.stem.split('.')[-2]

    cases = []
    # 分阶段读取数据.先读取经纬度.
    f = h5py.File(str(DPR_filepath), 'r')
    Longitude = f['NS/Longitude'][:]
    Latitude = f['NS/Latitude'][:]
    # 若没有像元落入DPR_extent,则返回空列表.
    DPR_mask = region_mask(Longitude, Latitude, DPR_extent)
    if not DPR_mask.any():
        f.close()
        return cases
    # 即便有像元落入,若没有降水,也返回空列表.
    precipRateNearSurface = f['NS/SLV/precipRateNearSurface'][:]
    flagPrecip = precipRateNearSurface > 0  # 以地表降水大于0为准.
    if not flagPrecip[DPR_mask].any():
        f.close()
        return cases
    # 继续读取地表类型和扫描时间.
    landSurfaceType = f['NS/PRE/landSurfaceType'][:]
    scanTime = read_GPM_time(f, group='NS')
    f.close()

    # 让map_extent范围外的flagPrecip都为0.
    # map_extent比DPR_extent的范围稍大,以便于找出降水个例.
    map_mask = region_mask(Longitude, Latitude, map_extent)
    flagPrecip[~map_mask] = 0

    # 标记出每个降水个例,并进行过滤.
    labelled, nlabel = two_pass(flagPrecip, radius=RAIN_RADIUS)
    ocean_mask = landSurfaceType < 100
    num = 1     # 能通过过滤的个例数.
    for label in range(1, nlabel + 1):
        case_mask = labelled == label
        case_pixel_num = np.count_nonzero(case_mask)
        # 滤掉降水像元数少于RAIN_PIXEL_NUM的个例.
        if case_pixel_num < RAIN_PIXEL_NUM:
            continue
        # 滤掉洋面降水占比超过OCEAN_RATIO的个例.
        ocean_pixel_num = np.count_nonzero(case_mask & ocean_mask)
        if ocean_pixel_num / case_pixel_num * 100 >= OCEAN_RATIO:
            continue
        # 滤掉降水中心不在DPR_extent范围内的个例.
        clon = Longitude[case_mask].mean()
        clat = Latitude[case_mask].mean()
        if not bool(region_mask(clon, clat, DPR_extent)):
            continue

        # 结合轨道号给个例编号.
        case_number = f'{orbit_number}_{num:02}'
        # 用nc文件保存个例像元的Boolean数组.
        da = xr.DataArray(case_mask, dims=('nscan', 'nray'), name='case_mask')
        output_filepath = output_dirpath / (case_number + '.nc')
        da.to_netcdf(str(output_filepath))

        # 将其它参数写入到字典中.
        rain_time = scanTime[case_mask.nonzero()[0]].mean()  # 高级索引.
        case = {
            'case_number': case_number,
            'rain_time': rain_time.strftime('%Y-%m-%d %H:%M:%S'),
            'rain_center': [float(clon), float(clat)],  # 使用原生的float.
            'DPR_filepath': str(DPR_filepath),
            'mask_filepath': str(output_filepath),
        }
        cases.append(case)
        num += 1

    return cases

cases = []
def collect_results(result):
    '''收集apply_async的结果.'''
    global cases
    cases.extend(result)

if __name__ == '__main__':
    start_time = config['start_time']
    end_time = config['end_time']
    dates = pd.date_range(start_time, end_time, freq='D')
    # 只选取春季的数据
    dates = dates[dates.month.isin([3, 4, 5])]

    # 创建保存个例Boolean数组的目录.
    output_dirpath = Path(config['temp_dirpath']) / 'case_masks'
    recreate_dir(output_dirpath)

    # 从所有DPR文件中寻找降水个例.
    p = Pool(8)
    for date in dates:
        for DPR_filepath in get_DPR_filepaths(date):
            p.apply_async(
                func=find_cases_from_one_file,
                args=(DPR_filepath, output_dirpath),
                callback=collect_results
            )
    p.close()
    p.join()
    # 利用个例编号对打乱的结果排序.
    cases.sort(key=lambda case: case['case_number'])

    # 打印找到的个例数.
    print(f'{len(cases)} rain cases found')

    # 把找到的降水个例都写到json文件中.
    output_filepath = Path(config['result_dirpath']) / 'rain_cases.json'
    with open(str(output_filepath), 'w') as f:
        json.dump(cases, f, indent=4)
