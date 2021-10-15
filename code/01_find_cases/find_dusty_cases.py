#----------------------------------------------------------------------------
# 2021/05/08
# 依据CALIPSO VFM数据的沙尘像元占比,从匹配个例中找出沙尘污染个例和清洁个例.
#
# 算法为:
# - 对一个匹配个例的每个VFM数据,截取降水中心上下一定纬度范围内的数据,
#   计算这段数据内沙尘气溶胶像元的占比,将这些占比收集起来.
# - 若这些占比中存在高于给定阈值的值,则认为该个例为沙尘个例;
#   若这些占比都低于给定的另一个阈值,则认为该个例为清洁个例;
#   否则该个例为污染水平普通的个例,不纳入统计.
#----------------------------------------------------------------------------
import json
from pathlib import Path
from multiprocessing import Pool

import numpy as np

from data_reader import Reader_for_CAL

# 读取配置文件,将config作为全局变量.
with open('config.json', 'r') as f:
    config = json.load(f)

def get_CAL_mask(track, center):
    '''
    以降水个例为中心,CAL_width为纬度范围,
    获取落入其中的CALIPSO数据的Boolean数组.
    '''
    CAL_width = config['CAL_width']
    lon, lat = track
    clon, clat = center
    dlat = CAL_width / 2
    mask = (lat >= (clat - dlat)) & (lat <= (clat + dlat))

    return mask

def dust_ratio(vfm):
    '''计算VFM中沙尘像元的百分比占比.'''
    return np.count_nonzero(vfm == 8) / vfm.size * 100

def case_type(ratios):
    '''根据沙尘像元百分比决定个例的种类.'''
    ratios = np.asarray(ratios)
    d0, d1 = config['DUST_RATIOS']
    # 若存在大于阈值d1的,视为'dusty';
    # 若全部小于阈值d0,视为'clean';否则视为'medium'.
    if np.any(ratios > d1):
        return 'dusty'
    elif np.all(ratios < d0):
        return 'clean'
    else:
        return 'medium'

def check_one_case(case):
    '''根据降水个例匹配的CALIPSO文件决定其污染程度.返回污染类型.'''
    rain_center = case['rain_center']
    ratios = []
    # 收集每个对应的CALIPSO文件的污染程度.
    for CAL_filepath in case['CAL_filepaths']:
        f = Reader_for_CAL(CAL_filepath)
        track = f.read_lonlat()
        vfm = f.read_vfm()
        f.close()

        # 基本上find_matched_cases中已经保证了scan_mask有值.
        scan_mask = get_CAL_mask(track, rain_center)
        vfm = vfm[scan_mask, :]
        ratio = dust_ratio(vfm)
        ratios.append(ratio)

    return case_type(ratios)

if __name__ == '__main__':
    # 读取匹配的降水个例.
    result_dirpath = Path(config['result_dirpath'])
    input_filepath = result_dirpath / 'matched_cases.json'
    with open(str(input_filepath), 'r') as f:
        cases = json.load(f)

    # 判断每个个例的污染程度.
    p = Pool(8)
    types = p.map(check_one_case, cases)
    p.close()
    p.join()
    # 根据污染程度筛选出沙尘个例和清洁个例.
    dusty_cases = [case for t, case in zip(types, cases) if t == 'dusty']
    clean_cases = [case for t, case in zip(types, cases) if t == 'clean']

    # 打印找到的个例数.
    ndusty = len(dusty_cases)
    nclean = len(clean_cases)
    print(f'{ndusty} dusty cases found')
    print(f'{nclean} clean cases found')

    # 将结果输出到一个json文件中.
    records = {
        'dusty': {
            'num': ndusty,
            'cases': dusty_cases
        },
        'clean': {
            'num': nclean,
            'cases': clean_cases
        }
    }
    # 把找到的结果写到json文件中.
    output_filepath = result_dirpath / 'found_cases.json'
    with open(str(output_filepath), 'w') as f:
        json.dump(records, f, indent=4)
