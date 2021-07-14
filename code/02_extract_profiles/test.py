import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import xarray as xr

# 读取配置文件,将config作为全局变量.
with open('config.json', 'r') as f:
    config = json.load(f)

if __name__ == '__main__':
    # result_dirpath = Path(config['result_dirpath'])
    # with open(str(result_dirpath / 'found_cases.json'), 'r') as f:
    #     records = json.load(f)

    # dusty_cases = records['dusty']['cases']
    # case = dusty_cases[-1]
    # DPR_filepath = Path(case['DPR_filepath'])
    # ERA5_filepath = Path(case['ERA5_filepath'])

    ds = xr.load_dataset('test.nc')
