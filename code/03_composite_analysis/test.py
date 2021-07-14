import json
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import xarray as xr
from scipy.stats import mstats

import matplotlib as mpl
import matplotlib.pyplot as plt

# 读取配置文件,作为全局变量使用.
with open('config.json', 'r') as f:
    config = json.load(f)

if __name__ == '__main__':
    # 读取dusty cases和clean cases.
    input_dirpath = Path(config['input_dirpath'])
    with open(str(input_dirpath / 'found_cases.json'), 'r') as f:
        records = json.load(f)
    dusty_ds = xr.load_dataset(records['dusty']['profile_filepath'])
    # clean_ds = xr.load_dataset(records['clean']['profile_filepath'])

    ds = dusty_ds
    cape = ds.cape.data
