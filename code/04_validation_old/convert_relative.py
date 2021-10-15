import json
from pathlib import Path
import sys
sys.path.append('../modules')

import numpy as np
import xarray as xr

from profile_funcs import Binner, smooth_profiles

# 读取配置文件,作为全局变量使用.
with open('config.json', 'r') as f:
    config = json.load(f)

if __name__ == '__main__':
    # 读取dusty cases和clean cases.
    input_dirpath = Path(config['temp_dirpath']) / 'merged'
    ds = xr.load_dataset(str(input_dirpath / 'all_profile.nc'))

    precipRate = ds.precipRate.data.copy()
    zFactorCorrected = ds.zFactorCorrected.data.copy()
    Nw = ds.Nw.data.copy()
    Dm = ds.Dm.data.copy()

    npoint, nbin = precipRate.shape
    dh = 0.125
    bins = np.arange(nbin + 1) * dh

    heightZeroDeg = ds.heightZeroDeg.data
    binZeroDeg = np.digitize(heightZeroDeg, bins, right=True) - 1
    binZeroDeg[binZeroDeg == nbin] = -1

    # 设置需要的相对下标的范围.
    # 要求i0和i1在[-175, 175]范围内,且i1-i0<=176.
    i0 = -32
    i1 = 80
    dh = 0.125
    height_relative = np.arange(i0, i1 + 1) * dh
    nbin_relative = height_relative.shape[0]

    # 提前将每条廓线中相对下标范围以外的数据填充为缺测.
    fill_value = np.nan
    inds_relative = np.arange(nbin) - binZeroDeg[:, None]
    invalid = (inds_relative < i0) | (inds_relative > i1)
    precipRate[invalid] = fill_value
    zFactorCorrected[invalid] = fill_value
    Nw[invalid] = fill_value
    Dm[invalid] = fill_value

    # 将每条廓线中对应于相对下标的数据循环移位到最前面.
    shifts = -(binZeroDeg + i0)
    rows, cols = np.ogrid[:npoint, :nbin]
    shifts[shifts < 0] += nbin      # 保证循环步长都是正数.
    cols = cols - shifts[:, None]
    # 利用advanced indexing进行每一行的循环移位,再截取前nbin_relative个数据.
    precipRate_relative = precipRate[rows, cols][:, :nbin_relative]
    zFactorCorrected_relative = zFactorCorrected[rows, cols][:, :nbin_relative]
    Nw_relative = Nw[rows, cols][:, :nbin_relative]
    Dm_relative = Dm[rows, cols][:, :nbin_relative]

    ds.coords['height_r'] = ('height_r', height_relative)
    ds['precipRate_r'] = (('npoint', 'height_r'), precipRate_relative)
    ds['zFactorCorrected_r'] = (('npoint', 'height_r'), zFactorCorrected_relative)
    ds['Nw_r'] = (('npoint', 'height_r'), Nw_relative)
    ds['Dm_r'] = (('npoint', 'height_r'), Dm_relative)

    output_filepath = input_dirpath / 'all_profile_relative.nc'
    ds.to_netcdf(str(output_filepath))
