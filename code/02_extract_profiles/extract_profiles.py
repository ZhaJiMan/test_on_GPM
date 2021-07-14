#----------------------------------------------------------------------------
# 2021/07/08
# 提取每个个例的降水廓线数据,并分别将污染组与清洁组的所有个例的廓线数据
# 连接为一个文件.
#
# 数据的空间范围为DPR_extent内的数据,程序具体操作为:
# - 提取个例对应的DPR文件中的降水信息.
# - 用个例对应的ENV文件中的温度信息将降水廓线等数据转换到温度坐标上.
# - 将GMI_1B的亮温数据通过最邻近插值插到DPR坐标上.
# - 将ERA5 CAPE数据通过线性插值查到DPR坐标上.
# - 将上述的数据保存到NC文件中.
# - 将每组个例的NC文件连接为一整个文件.
#
# 注意:
# - 输出的廓线数据,高度和温度都随序号递增.
# - 雨型是整型数组,为了避免xarray读取时将其解读为浮点型,不设置缺测值.
#----------------------------------------------------------------------------
import json
from pathlib import Path
from multiprocessing import Pool

import sys
sys.path.append('../modules')
from region_funcs import get_extent_flag_both
from helper_funcs import recreate_dir
from convert_funcs import *

import h5py
import numpy as np
import pandas as pd
import xarray as xr

# 读取配置文件,将config作为全局变量.
with open('config.json', 'r') as f:
    config = json.load(f)

def set_output_filename(case):
    '''根据个例的降水时间设置nc文件的输出名.'''
    rain_time = pd.to_datetime(case['rain_time'])
    time_str = rain_time.strftime('%Y%m%d_%H%M')
    output_filename = time_str + '.nc'

    return output_filename

def extract_one_case(case, output_filepath):
    '''
    提取出一个个例中的降水廓线数据,保存为nc文件.
    01_find_cases中的算法已经保证了每个个例肯定有降水数据.
    '''
    DPR_extent = config['DPR_extent']
    # 读取DPR数据.
    with h5py.File(case['DPR_filepath'], 'r') as f:
        lon2D = f['NS/Longitude'][:]
        lat2D = f['NS/Latitude'][:]
        flagPrecip = f['NS/PRE/flagPrecip'][:]
        # 选取落入DPR_extent中且有降水的数据点.
        extent_flag = get_extent_flag_both(lon2D, lat2D, DPR_extent)
        rain_flag = flagPrecip > 0
        all_flag = extent_flag & rain_flag

        # 读取经纬度.
        lon1D = lon2D[all_flag]
        lat1D = lat2D[all_flag]
        # 读取高度量.
        elevation = f['NS/PRE/elevation'][:][all_flag]
        heightStormTop = f['NS/PRE/heightStormTop'][:][all_flag]
        heightZeroDeg = f['NS/VER/heightZeroDeg'][:][all_flag]
        binRealSurface = f['NS/PRE/binRealSurface'][:][all_flag]
        # 读取降水量.
        typePrecip = f['NS/CSF/typePrecip'][:][all_flag]
        precipRateNearSurface = f['NS/SLV/precipRateNearSurface'][:][all_flag]
        precipRate = f['NS/SLV/precipRate'][:][all_flag, :]
        zFactorCorrected = f['NS/SLV/zFactorCorrected'][:][all_flag, :]
        # 读取DSD量.
        Nw = f['NS/SLV/paramDSD'][:][all_flag, :, 0]
        Dm = f['NS/SLV/paramDSD'][:][all_flag, :, 1]

    # 读取ENV数据.
    with h5py.File(case['ENV_filepath'], 'r') as f:
        airTemperature = f['NS/VERENV/airTemperature'][:][all_flag, :]

    # 读取SLH数据.
    with h5py.File(case['SLH_filepath'], 'r') as f:
        SLH = f['Swath/latentHeating'][:][all_flag, :]

    # 读取VPH数据.
    with xr.open_dataset(case['VPH_filepath'], mask_and_scale=False) as ds:
        VPH = ds['latentHeating'].data[all_flag, :]

    # 读取GMI数据.
    with h5py.File(case['GMI_filepath'], 'r') as f:
        lon_GMI = f['S1/Longitude'][:]
        lat_GMI = f['S1/Latitude'][:]
        extent_flag = get_extent_flag_both(lon_GMI, lat_GMI, DPR_extent)
        lon_GMI = lon_GMI[extent_flag]
        lat_GMI = lat_GMI[extent_flag]

        # 读取亮温.
        Tb = f['S1/Tb'][:][extent_flag, :]

    # 读取ERA5数据.
    # 为了简化,选取降水时间前一个小时.
    rain_time = pd.to_datetime(case['rain_time'])
    with xr.open_dataset(case['ERA5_filepath']) as ds:
        cape = ds.cape.sel(time=rain_time.floor('H'))

    # 设置DPR使用的高度,使之与DPR变量相匹配.
    nbin = 176
    dh = 0.125  # 单位为km.
    height = (np.arange(nbin) + 0.5)[::-1] * dh
    # 设置SLH使用的高度.注意其是单调递增的.
    nlayer = 80
    height_LH = (np.arange(nlayer) + 0.5) * dh * 2

    # 设置DPR文件中常用的缺测值.
    fill_value = -9999.9
    # 将高度量的单位转为km.
    elevation[~np.isclose(elevation, fill_value)] /= 1000
    heightStormTop[~np.isclose(heightStormTop, fill_value)] /= 1000
    heightZeroDeg[~np.isclose(heightZeroDeg, fill_value)] /= 1000
    # 将温度的单位转为摄氏度,并且不用考虑缺测.
    airTemperature -= 273.15

    # 给出目标温度坐标.
    tmin = -60
    tmax = 20
    dt = 0.5
    temp = np.linspace(tmin, tmax, int((tmax - tmin) / dt) + 1)
    # 以地表与12km之间为高度范围.
    hmax = 12
    bottom_inds = binRealSurface - 1
    top_ind = np.nonzero(height <= hmax)[0][0]
    top_inds = np.full_like(bottom_inds, top_ind)

    # 转换廓线数据.
    converter = profile_converter(airTemperature, temp, bottom_inds, top_inds)
    # precipRate高空的缺测用0填充,地表以下用fill_value填充.
    precipRate_t = converter(precipRate, (0.0, fill_value))
    zFactorCorrected_t = converter(zFactorCorrected, fill_value)
    Nw_t = converter(Nw, fill_value)
    Dm_t = converter(Dm, fill_value)
    # VPH需要倒转高度维,并且高空缺测用0填充.
    VPH_t = converter(VPH[:, ::-1], (0.0, fill_value))
    # 由于SLH的垂直分辨率是DPR数据的一半,
    # 所以先将潜热数据沿高度复制两遍,再扩展到nbin大小,用0填充空位.
    SLH_new = np.hstack([
        SLH.repeat(2, axis=1),
        np.full((SLH.shape[0], nbin - nlayer * 2), 0.0, dtype=SLH.dtype)
    ])
    SLH_t = converter(SLH_new[:, ::-1], (0.0, fill_value))

    # 转换雨顶高度为雨顶温度.
    tempStormTop = convert_height(
        heightStormTop, height, airTemperature, fill_value
    )

    # 划分雨型.
    # -1表示缺测或没有降水.
    # 1表示stratiform rain.
    # 2表示deep convective rain.
    # 3表示shallow convective rain.
    # 4表示other rain.
    shallow_flag = \
        (heightStormTop > fill_value) & \
        (heightZeroDeg > fill_value) & \
        (heightStormTop < heightZeroDeg)
    rainType = np.where(typePrecip > 0, typePrecip // 10000000, -9999)
    rainType[rainType == 3] = 4
    rainType[(rainType == 2) & shallow_flag] = 3

    # 将GMI亮温匹配到DPR数据点上.
    Tb89V = Tb[:, 7]
    Tb89H = Tb[:, 8]
    PCT89 = 1.818 * Tb89V - 0.818 * Tb89H
    points_DPR = np.column_stack([lon1D, lat1D])
    points_GMI = np.column_stack([lon_GMI, lat_GMI])
    PCT89_DPR = match_to_DPR(points_DPR, points_GMI, PCT89)

    # 为了便于以后使用,将高度坐标上的廓线数据倒转,
    # 使高度随下标增大而增大.
    height = height[::-1]
    precipRate = precipRate[:, ::-1]
    zFactorCorrected = zFactorCorrected[:, ::-1]
    Nw = Nw[:, ::-1]
    Dm = Dm[:, ::-1]

    # 将ERA5的网格数据线性插值到lon1D和lat1D的点上.
    # 如果点超出了网格范围默认会报错,不过多进程的话不一定能看出来.
    cape_DPR = cape.interp(
        longitude=xr.DataArray(lon1D, dims='npoint'),
        latitude=xr.DataArray(lat1D, dims='npoint')
    ).data

    # 指定维度.
    dim_1D = ('npoint',)
    dim_Rr = ('npoint', 'height')
    dim_LH = ('npoint', 'height_LH')
    dim_te = ('npoint', 'temp')
    # 暂时不设属性.
    attrs = None
    # 用encoding指定数据类型和压缩参数.
    comp = {'zlib': True, 'complevel': 4}
    ec_flt = {'dtype': 'float32', '_FillValue': fill_value, **comp}
    ec_int = {'dtype': 'int32', '_FillValue': None, **comp}
    ec_coord = {'dtype': 'float32', '_FillValue': None, **comp}
    # 将数据存储到Dataset中.
    ds = xr.Dataset(
        data_vars={
            'lon': (dim_1D, lon1D, attrs, ec_flt),
            'lat': (dim_1D, lat1D, attrs, ec_flt),
            'elevation': (dim_1D, elevation, attrs, ec_flt),
            'heightStormTop': (dim_1D, heightStormTop, attrs, ec_flt),
            'tempStormTop': (dim_1D, tempStormTop, attrs, ec_flt),
            'heightZeroDeg': (dim_1D, heightZeroDeg, attrs, ec_flt),
            'rainType': (dim_1D, rainType, attrs, ec_int),
            'precipRateNearSurface': (dim_1D, precipRateNearSurface, attrs, ec_flt),
            'PCT89': (dim_1D, PCT89_DPR, attrs, ec_flt),
            'precipRate': (dim_Rr, precipRate, attrs, ec_flt),
            'precipRate_t': (dim_te, precipRate_t, attrs, ec_flt),
            'zFactorCorrected': (dim_Rr, zFactorCorrected, attrs, ec_flt),
            'zFactorCorrected_t': (dim_te, zFactorCorrected_t, attrs, ec_flt),
            'Nw': (dim_Rr, Nw, attrs, ec_flt),
            'Dm': (dim_Rr, Dm, attrs, ec_flt),
            'Nw_t': (dim_te, Nw_t, attrs, ec_flt),
            'Dm_t': (dim_te, Dm_t, attrs, ec_flt),
            'SLH': (dim_LH, SLH, attrs, ec_flt),
            'VPH': (dim_Rr, VPH, attrs, ec_flt),
            'SLH_t': (dim_te, SLH_t, attrs, ec_flt),
            'VPH_t': (dim_te, VPH_t, attrs, ec_flt),
            'cape': (dim_1D, cape_DPR, attrs, ec_flt)
        },
        coords={
            'height': (('height',), height, attrs, ec_coord),
            'height_LH': (('height_LH',), height_LH, attrs, ec_coord),
            'temp': (('temp',), temp, attrs, ec_coord)
        }
    )
    # 保存为nc文件.
    ds.to_netcdf(str(output_filepath))

def concat_cases(cases, output_filepath):
    '''将每个个例的nc文件沿npoint维连接起来,再保存为nc文件.'''
    ds_list = [xr.load_dataset(case['profile_filepath']) for case in cases]
    ds_all = xr.concat(ds_list, dim='npoint')
    # 重新制定encoding.
    comp = {'zlib': True, 'complevel': 4}
    ec_flt = {'dtype': 'float32', '_FillValue': -9999.9, **comp}
    ec_int = {'dtype': 'int32', '_FillValue': None, **comp}
    ec_coord = {'dtype': 'float32', '_FillValue': None, **comp}
    encoding = {}
    for var in ds_all.variables:
        if var in ['height', 'height_LH', 'temp']:
            encoding[var] = ec_coord
        elif var == 'rainType':
            encoding[var] = ec_int
        else:
            encoding[var] = ec_flt
    ds_all.to_netcdf(str(output_filepath), encoding=encoding)

if __name__ == '__main__':
    # 读取两组个例.
    input_dirpath = Path(config['input_dirpath'])
    with open(str(input_dirpath / 'found_cases.json'), 'r') as f:
        records = json.load(f)
    dusty_cases = records['dusty']['cases']
    clean_cases = records['clean']['cases']

    # 创建输出目录.
    output_dirpath = Path(config['data_dirpath']) / 'profile_files'
    recreate_dir(output_dirpath)
    # 再创建两个子目录.
    dusty_dirpath = output_dirpath / 'dusty_cases'
    clean_dirpath = output_dirpath / 'clean_cases'
    dusty_dirpath.mkdir()
    clean_dirpath.mkdir()

    # 提取出两组个例的廓线数据,分别保存到各自的目录下.
    # 同时将nc文件路径写入到case中.
    p = Pool(16)
    for case in dusty_cases:
        output_filepath = dusty_dirpath / set_output_filename(case)
        p.apply_async(extract_one_case, args=(case, output_filepath))
        case['profile_filepath'] = str(output_filepath)
    for case in clean_cases:
        output_filepath = clean_dirpath / set_output_filename(case)
        p.apply_async(extract_one_case, args=(case, output_filepath))
        case['profile_filepath'] = str(output_filepath)
    p.close()
    p.join()

    # 分别连接两组个例的nc文件,并把文件路径写入到record中.
    output_filepath1 = output_dirpath / 'dusty_profile.nc'
    output_filepath2 = output_dirpath / 'clean_profile.nc'
    concat_cases(dusty_cases, output_filepath1)
    concat_cases(clean_cases, output_filepath2)
    records['dusty']['profile_filepath'] = str(output_filepath1)
    records['clean']['profile_filepath'] = str(output_filepath2)

    # 重新保存修改后的json文件到02的目录下.
    result_dirpath = Path(config['result_dirpath'])
    with open(str(result_dirpath / 'found_cases.json'), 'w') as f:
        json.dump(records, f, indent=4)
