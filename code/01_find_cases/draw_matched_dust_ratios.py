#----------------------------------------------------------------------------
# 2021/09/17
# 将匹配个例中的CALIPSO VFM数据的沙尘像元占比绘制成histogram图.
#
# 若一个个例匹配多个CALIPSO文件,则取最大的占比.
#----------------------------------------------------------------------------
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from data_reader import Reader_for_CAL
from find_dusty_cases import get_CAL_mask#, dust_ratio

# 读取配置文件,将config作为全局变量.
with open('config.json', 'r') as f:
    config = json.load(f)

def dust_ratio(vfm):
    '''计算VFM中沙尘像元的百分比占比.'''
    dust_num = np.count_nonzero(vfm == 8)
    aerosol_num = np.count_nonzero(vfm == 3) + dust_num
    if aerosol_num > 0:
        return dust_num / aerosol_num * 100
    else:
        return 0
    # return np.count_nonzero(vfm == 3) / vfm.size * 100

def calc_one_case(case):
    '''计算一个个例匹配的CALIPSO文件中最大的沙尘气溶胶占比.'''
    rain_center = case['rain_center']
    ratios = []
    for CAL_filepath in case['CAL_filepaths']:
        f = Reader_for_CAL(CAL_filepath)
        track = f.read_lonlat()
        vfm = f.read_vfm()
        f.close()

        scan_mask = get_CAL_mask(track, rain_center)
        vfm = vfm[scan_mask, :]
        ratio = dust_ratio(vfm)
        ratios.append(ratio)

    return max(ratios)

if __name__ == '__main__':
    # 读取匹配的降水个例.
    result_dirpath = Path(config['result_dirpath'])
    input_filepath = result_dirpath / 'matched_cases.json'
    with open(str(input_filepath), 'r') as f:
        cases = json.load(f)

    ncase = len(cases)
    ratios = list(map(calc_one_case, cases))

    fig, ax = plt.subplots()
    ax.hist(ratios)

    # 保存图片.
    output_filepath = Path(config['result_dirpath']) / 'dust_ratios.png'
    fig.savefig(str(output_filepath), dpi=300, bbox_inches='tight')
    plt.close(fig)
