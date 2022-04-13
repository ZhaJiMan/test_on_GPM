'''
2022-04-12
根据CAL记录和MERRA2记录从降水个例中筛选出污染个例和清洁个例.

流程:
- 根据降水个例中的CAL记录和MERRA2记录筛选出污染个例和清洁个例(具体判据建议
  直接阅读代码, 或参考论文).
- 根据后续的分析手动删除一些不合适的个例.
- 将污染个例和清洁个例的信息分别到json文件中.

输入:
- cases_rain.json

输出:
- cases_dusty.json
- cases_clean.json

注意:
- 筛选时使用的参数都硬编码在脚本里.
'''
import json
from pathlib import Path
import subprocess
import sys
sys.path.append('../modules')

import numpy as np

# 读取配置文件, 将config作为全局变量.
with open('config.json') as f:
    config = json.load(f)

def classify_case(case):
    '''根据个例含有的气溶胶信息进行分类, 返回个例类型.'''
    record_MERRA2 = case['record_MERRA2']
    record_CAL = case['record_CAL']

    # 用NaN表示缺测的信息.
    aod_dust = record_MERRA2.get('aod_dust', np.nan)
    aod_total = record_MERRA2.get('aod_total', np.nan)
    ratio_dust = record_CAL.get('ratio_dust', np.nan)
    ratio_aerosol = record_CAL.get('ratio_aerosol', np.nan)
    ratio_divide = ratio_dust / ratio_aerosol * 100

    cond_dusty1 = ratio_dust > 8 and ratio_divide > 50
    cond_dusty2 = aod_dust > 0.2
    cond_dusty = cond_dusty1 or cond_dusty2

    cond_clean1 = aod_total < 0.4 and aod_dust < 0.08
    cond_clean2 = True if np.isnan(ratio_dust) else ratio_dust < 4
    cond_clean = cond_clean1 and cond_clean2

    # cond_dusty优先级更高.
    if cond_dusty:
        return 'dusty'
    elif cond_clean:
        return 'clean'
    else:
        return None

def delete_cases(cases, case_numbers):
    '''根据个例编号手动删除一些个例, 并返回过滤后的个例列表.'''
    cases_new = []
    for case in cases:
        if not case['case_number'] in case_numbers:
            cases_new.append(case)

    return cases_new

if __name__ == '__main__':
    # 读取降水个例.
    dirpath_result = Path(config['dirpath_result'])
    with open(str(dirpath_result / 'cases_rain.json')) as f:
        cases_rain = json.load(f)

    # 收集污染和清洁个例.
    cases_dusty = []
    cases_clean = []
    for case in cases_rain:
        result = classify_case(case)
        if result == 'dusty':
            cases_dusty.append(case)
        elif result == 'clean':
            cases_clean.append(case)
        else:
            continue

    # 手动删除一些个例.
    case_numbers = ['017893_01', '023918_01', '024117_01']
    cases_dusty = delete_cases(cases_dusty, case_numbers)

    # 打印筛选出的个例.
    print('Dusty Cases:', len(cases_dusty))
    print('Clean Cases:', len(cases_clean))
    # 将两组个例写入json文件.
    with open(str(dirpath_result / 'cases_dusty.json'), 'w') as f:
        json.dump(cases_dusty, f, indent=4)
    with open(str(dirpath_result / 'cases_clean.json'), 'w') as f:
        json.dump(cases_clean, f, indent=4)

    # subprocess.run(['python', 'merge_and_add_ERA5_filepath_to_dusty_cases.py'])
    # subprocess.run(['python', 'draw_dusty_cases.py'])
