import json
from pathlib import Path
import subprocess
import sys
sys.path.append('../modules')

import numpy as np
import pandas as pd

# 读取配置文件, 将config作为全局变量.
with open('config.json') as f:
    config = json.load(f)

if __name__ == '__main__':
    # 读取降水个例.
    dirpath_result = Path(config['dirpath_result'])
    with open(str(dirpath_result / 'cases_rain.json')) as f:
        cases_rain = json.load(f)

    ncase = len(cases_rain)
    aod_combined = np.full(ncase, np.nan)
    ae_db = np.full(ncase, np.nan)
    aod_total = np.full(ncase, np.nan)
    aod_dust = np.full(ncase, np.nan)
    ratio_dust = np.full(ncase, np.nan)
    ratio_aerosol = np.full(ncase, np.nan)
    dt = np.full(ncase, np.nan)
    dx = np.full(ncase, np.nan)

    for i, case in enumerate(cases_rain):
        record_MYD = case['record_MYD']
        record_MERRA2 = case['record_MERRA2']
        record_CAL = case['record_CAL']

        if record_MYD:
            aod_combined[i] = record_MYD['aod_combined']
            ae_db[i] = record_MYD['ae_db']

        if record_MERRA2:
            aod_total[i] = record_MERRA2['aod_total']
            aod_dust[i] = record_MERRA2['aod_dust']

        if record_CAL:
            ratio_dust[i] = record_CAL['ratio_dust']
            ratio_aerosol[i] = record_CAL['ratio_aerosol']
            dt[i] = record_CAL['dt']
            dx[i] = record_CAL['dx']

    ratio_divide = ratio_dust / ratio_aerosol * 100
    cases_rain = np.array(cases_rain, dtype=object)

    # cond_dusty = (
    #     (ratio_dust > 4) & (ratio_divide > 50) &
    #     (aod_dust > 0.1) | (aod_dust > 0.2)
    # )
    # cond_clean = (aod_total < 0.4) & (aod_dust < 0.08)
    # cases_dusty = cases_rain[cond_dusty].tolist()
    # cases_clean = cases_rain[cond_clean].tolist()

    cond_close = (np.abs(dx) < 3) & (np.abs(dt) < 3)
    cond_upstream = (dx > -6) & (dx < 0) & (dt > -6) & (dt < 0)
    # cond_upstream = False
    cond_all = cond_close | cond_upstream

    cond_all = (dx > -6) & (dx < 3) & (dt > -6) & (dt < 3)

    # cond_all = aod_dust > 0.15
    cases_dusty = cases_rain[cond_all].tolist()

    print('Dusty Cases:', len(cases_dusty))
    # print('Clean Cases:', len(cases_clean))

    # # 将两组个例写入json文件.
    # with open(str(dirpath_result / 'cases_dusty.json'), 'w') as f:
    #     json.dump(cases_dusty, f, indent=4)
    # # with open(str(dirpath_result / 'cases_clean.json'), 'w') as f:
    # #     json.dump(cases_clean, f, indent=4)

    # subprocess.run(['python', 'merge_and_add_ERA5_filepath_to_dusty_cases.py'])
    # subprocess.run(['python', 'draw_dusty_cases.py'])

