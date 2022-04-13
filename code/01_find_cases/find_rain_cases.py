'''
2021-07-27
根据GPM DPR的观测找出华北地区发生的降水个例.

流程:
- 依次读取每一天的DPR文件, 判断是否有降水像元落入研究区域中.
- 接着在比研究区域稍大的区域中利用连通域标记算法标记出每个降水个例,
  若降水个例像元够多, 海面上的部分够小, 中心落在研究区域内,
  则记为华北的一个降水个例.
- 记录该个例的信息并保存.

输入:
- 服务器上的2ADPR文件.

输出:
- 记录所有个例信息的json文件, 个例信息包括:
  - 个例编号.
  - 降水时间和中心.
  - 对应文件的路径.
- 用于在DPR文件中索引降水点的掩膜数组.

参数:
- time_start: 选取DPR记录的起始时间.
- time_end: 选取DPR记录的结束时间.
- extents_DPR: 研究区域的范围
- extents_map: 地图范围, 比研究区域稍大.
- RAIN_RADIUS: 连通域算法里搜索相邻像素的半径.
- RAIN_PIXEL_NUM: 个例至少含有的降水像元数.
- OCEAN_RATIO: 个例在洋面上的像元数不能超过这个值.

注意:
- 时间范围为春季(3~5月), 硬编码在脚本中.
- 降水像元是根据近地表降水率是否大于0判断的.
- 脚本使用了多进程.
'''
import json
from pathlib import Path
from multiprocessing import Pool
import sys
sys.path.append('../modules')

import h5py
import numpy as np
import pandas as pd

import region_tools
import helper_tools
import data_tools
import labelling

# 读取配置文件, 将config作为全局变量.
with open('config.json') as f:
    config = json.load(f)
# 其它全局参数.
RAIN_RADIUS = 5
RAIN_PIXEL_NUM = 50
OCEAN_RATIO = 20

def find_cases_from_one_file(filepath_DPR, dirpath_output):
    '''
    在一个DPR文件中寻找研究区域内的降水个例.

    返回存有每个个例信息的列表, 未找到时返回空列表.
    同时会把个例的下标布尔数组以npy格式保存到dirpath_output目录下.
    '''
    extents_map = config['extents_map']
    extents_DPR = config['extents_DPR']
    orbit_number = filepath_DPR.stem.split('.')[-2]

    # 尝试寻找DPR文件对应的ENV和SLH文件.
    try:
        filepath_ENV = data_tools.to_ENV_filepath(filepath_DPR)
        filepath_SLH = data_tools.to_SLH_filepath(filepath_DPR)
    except FileNotFoundError:
        return []

    # 分阶段读取数据, 先读取经纬度.
    f = data_tools.ReaderDPR(str(filepath_DPR))
    Longitude, Latitude = f.read_lonlat()
    # 要求有像元落入研究区域.
    mask_DPR = region_tools.region_mask(Longitude, Latitude, extents_DPR)
    if not mask_DPR.any():
        f.close()
        return []
    # 要求落入的像元至少有近地表降水率大于0的.
    precipRateNearSurface = f.read_ds('SLV/precipRateNearSurface')
    mask_rain = precipRateNearSurface > 0
    if not mask_rain[mask_DPR].any():
        f.close()
        return []
    # 继续读取地表类型和扫描时间, 并关闭文件.
    landSurfaceType = f.read_ds('PRE/landSurfaceType', mask=False)
    scanTime = f.read_time()
    f.close()

    # 让extents_map范围外的mask_rain都为False.
    mask_map = region_tools.region_mask(Longitude, Latitude, extents_map)
    mask_rain[~mask_map] = False

    # 标记出extents_map里的每个降水个例, 过滤掉不符要求的个例.
    cases = []
    labelled, nlabel = labelling.two_pass(mask_rain, radius=RAIN_RADIUS)
    mask_ocean = landSurfaceType < 100
    num = 0     # 符合要求的个例数.
    for label in range(1, nlabel + 1):
        mask_case = labelled == label
        pixel_num_case = np.count_nonzero(mask_case)
        # 滤掉降水像元数少于RAIN_PIXEL_NUM的个例.
        if pixel_num_case < RAIN_PIXEL_NUM:
            continue
        # 滤掉洋面降水占比超过OCEAN_RATIO的个例.
        pixel_num_ocean = np.count_nonzero(mask_case & mask_ocean)
        ocean_ratio = pixel_num_ocean / pixel_num_case * 100
        if ocean_ratio >= OCEAN_RATIO:
            continue
        # 滤掉降水中心不在extents_DPR范围内的个例.
        clon = Longitude[mask_case].mean()
        clat = Latitude[mask_case].mean()
        if not region_tools.region_mask(clon, clat, extents_DPR):
            continue

        # 结合轨道号给个例编号.
        num += 1
        case_number = f'{orbit_number}_{num:02}'
        # 用npy文件保存个例像元的Boolean数组.
        filepath_output = dirpath_output / (case_number + '.npy')
        np.save(str(filepath_output), mask_case)

        # 将个例信息写入到字典中.
        rain_time = scanTime[mask_case.nonzero()[0]].mean()
        case = {
            'case_number': case_number,
            'rain_time': rain_time.strftime('%Y-%m-%d %H:%M:%S'),
            'rain_center': [float(clon), float(clat)],
            'filepath_DPR': str(filepath_DPR),
            'filepath_mask': str(filepath_output),
            'filepath_ENV': str(filepath_ENV),
            'filepath_SLH': str(filepath_SLH)
        }
        # 最后添加到个例列表中.
        cases.append(case)

    return cases

cases = []
def collect_results(result):
    '''收集apply_async的结果.'''
    global cases
    cases.extend(result)

if __name__ == '__main__':
    time_start = config['time_start']
    time_end = config['time_end']
    dates = pd.date_range(time_start, time_end, freq='D')
    # 只选取春季的数据.
    dates = dates[dates.month.isin([3, 4, 5])]

    # 重新创建保存个例Boolean数组的目录.
    dirpath_result = Path(config['dirpath_result'])
    dirpath_output = Path(config['dirpath_data'], 'DPR_case', 'mask')
    helper_tools.renew_dir(dirpath_output, parents=True)

    # 从所有DPR文件中寻找降水个例.
    p = Pool(10)
    for date in dates:
        for filepath_DPR in data_tools.get_DPR_filepaths_one_day(date):
            p.apply_async(
                find_cases_from_one_file,
                args=(filepath_DPR, dirpath_output),
                callback=collect_results
            )
    p.close()
    p.join()
    # 利用个例编号对打乱的结果排序.
    cases.sort(key=lambda case: case['case_number'])

    # 打印找到的个例数.
    print(f'{len(cases)} rain cases found')
    # 把找到的降水个例写入json文件中.
    filepath_output = dirpath_result / 'cases_rain.json'
    with open(str(filepath_output), 'w') as f:
        json.dump(cases, f, indent=4)
