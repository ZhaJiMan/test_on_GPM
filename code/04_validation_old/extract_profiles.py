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
from convert_funcs import profile_converter, convert_height

# 读取配置文件,将config作为全局变量.
with open('config.json', 'r') as f:
    config = json.load(f)

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

def to_ENV_filepath(DPR_filepath):
    '''根据DPR文件的路径获取对应的ENV文件的路径.'''
    DPR_filename = DPR_filepath.name
    parts = DPR_filename.split('.')
    yy = parts[4][:4]

    ENV_dirpath = DPR_filepath.parents[4] / 'ENV' / 'V06'
    parts[0] = '2A-ENV'
    ENV_filename = '.'.join(parts)
    ENV_filepath = ENV_dirpath / yy / ENV_filename

    # 若文件不存在,报错并提示.
    assert ENV_filepath.exists(), str(ENV_filepath) + ' does not exist'

    return ENV_filepath

def extract_one_file(DPR_filepath, output_dirpath):
    '''
    提取出一个DPR文件中落入DPR_extent范围内的降水廓线数据,再保存为nc文件.

    要求地表降水大于0,发生在陆面.
    同时将高度坐标下的廓线匹配到温度坐标上.
    '''
    DPR_extent = config['DPR_extent']
    orbit_number = DPR_filepath.stem.split('.')[-2]
    ENV_filepath = to_ENV_filepath(DPR_filepath)

    # 先读取基本数据.
    f = h5py.File(str(DPR_filepath), 'r')
    Longitude = f['NS/Longitude'][:]
    Latitude = f['NS/Latitude'][:]
    precipRateNearSurface = f['NS/SLV/precipRateNearSurface'][:]
    landSurfaceType = f['NS/PRE/landSurfaceType'][:]

    # 若DPR_extent中没有陆面降水的像元,那么停止计算.
    DPR_mask = region_mask(Longitude, Latitude, DPR_extent)
    flagPrecip = precipRateNearSurface > 0
    land_mask = landSurfaceType > 99
    all_mask = DPR_mask & flagPrecip & land_mask
    if not all_mask.any():
        f.close()
        return None

    lon1D = Longitude[all_mask]
    lat1D = Latitude[all_mask]
    # 读取高度量.
    elevation = f['NS/PRE/elevation'][:][all_mask]
    heightStormTop = f['NS/PRE/heightStormTop'][:][all_mask]
    heightZeroDeg = f['NS/VER/heightZeroDeg'][:][all_mask]
    binRealSurface = f['NS/PRE/binRealSurface'][:][all_mask]
    # 读取降水量.
    typePrecip = f['NS/CSF/typePrecip'][:][all_mask]
    flagShallowRain = f['NS/CSF/flagShallowRain'][:][all_mask]
    precipRateNearSurface = f['NS/SLV/precipRateNearSurface'][:][all_mask]
    precipRate = f['NS/SLV/precipRate'][:][all_mask, :]
    zFactorCorrected = f['NS/SLV/zFactorCorrected'][:][all_mask, :]
    # 读取DSD量.
    paramDSD = f['NS/SLV/paramDSD'][:][all_mask, :, :]
    Nw = paramDSD[:, :, 0]
    Dm = paramDSD[:, :, 1]
    # 读取结束.
    f.close()

    # 读取ENV数据.
    with h5py.File(str(ENV_filepath), 'r') as f:
        airTemperature = f['NS/VERENV/airTemperature'][:][all_mask, :]

    # 设置DPR使用的高度,使之与DPR变量相匹配.
    nbin = 176
    dh = 0.125  # 单位为km.
    height = (np.arange(nbin) + 0.5)[::-1] * dh

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

    # 转换雨顶高度为雨顶温度.
    tempStormTop = convert_height(
        heightStormTop, height, airTemperature, fill_value
    )

    # 划分雨型.
    # -9999表示缺测或没有降水.
    # 1表示stratiform rain.
    # 2表示deep convective rain.
    # 3表示shallow convective rain.
    # 4表示other rain.
    rainType = np.where(typePrecip > 0, typePrecip // 10000000, -9999)
    rainType[rainType == 3] = 4
    rainType[flagShallowRain > 0] = 3
    rainType[(rainType == 2) & (flagShallowRain == 0)] = 2

    # 为了便于以后使用,将高度坐标上的廓线数据倒转,
    # 使高度随下标增大而增大.
    height = height[::-1]
    precipRate = precipRate[:, ::-1]
    zFactorCorrected = zFactorCorrected[:, ::-1]
    Nw = Nw[:, ::-1]
    Dm = Dm[:, ::-1]

    # 设置与一维数据等长的月份.
    mm = int(DPR_filepath.stem[27:29])
    month = np.full(len(lon1D), mm, dtype=np.int32)

    # 指定维度.
    dim_1D = ('npoint',)
    dim_Rr = ('npoint', 'height')
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
            'month': (dim_1D, month, attrs, ec_int),
            'elevation': (dim_1D, elevation, attrs, ec_flt),
            'heightStormTop': (dim_1D, heightStormTop, attrs, ec_flt),
            'tempStormTop': (dim_1D, tempStormTop, attrs, ec_flt),
            'heightZeroDeg': (dim_1D, heightZeroDeg, attrs, ec_flt),
            'rainType': (dim_1D, rainType, attrs, ec_int),
            'precipRateNearSurface': (dim_1D, precipRateNearSurface, attrs, ec_flt),
            'precipRate': (dim_Rr, precipRate, attrs, ec_flt),
            'precipRate_t': (dim_te, precipRate_t, attrs, ec_flt),
            'zFactorCorrected': (dim_Rr, zFactorCorrected, attrs, ec_flt),
            'zFactorCorrected_t': (dim_te, zFactorCorrected_t, attrs, ec_flt),
            'Nw': (dim_Rr, Nw, attrs, ec_flt),
            'Dm': (dim_Rr, Dm, attrs, ec_flt),
            'Nw_t': (dim_te, Nw_t, attrs, ec_flt),
            'Dm_t': (dim_te, Dm_t, attrs, ec_flt),
        },
        coords={
            'height': (('height',), height, attrs, ec_coord),
            'temp': (('temp',), temp, attrs, ec_coord)
        }
    )
    # 保存为nc文件.
    output_filepath = output_dirpath / (orbit_number + '.nc')
    ds.to_netcdf(str(output_filepath))

def concat_and_divide(input_dirpath, output_dirpath):
    '''
    将每个nc文件沿npoint维连接起来,再按月份和纬度划分为dusty和clean
    两组数据集,最后保存为nc文件.
    '''
    DIVIDE_LAT = config['DIVIDE_LAT']
    ds_list = list(map(xr.load_dataset, input_dirpath.glob('*')))
    ds_all = xr.concat(ds_list, dim='npoint')

    # 分出dusty数据集.
    cond = ds_all.month.isin([3, 4]) & (ds_all.lat > DIVIDE_LAT)
    dusty_ds = ds_all.isel(npoint=cond)
    # 分出clean数据集.
    cond = (ds_all.month == 5) & (ds_all.lat <= DIVIDE_LAT)
    clean_ds = ds_all.isel(npoint=cond)

    # 重新制定encoding.
    comp = {'zlib': True, 'complevel': 4}
    ec_flt = {'dtype': 'float32', '_FillValue': -9999.9, **comp}
    ec_int = {'dtype': 'int32', '_FillValue': None, **comp}
    ec_coord = {'dtype': 'float32', '_FillValue': None, **comp}
    encoding = {}
    for var in ds_all.variables:
        if var in ['height', 'temp']:
            encoding[var] = ec_coord
        elif var in ['month', 'rainType']:
            encoding[var] = ec_int
        else:
            encoding[var] = ec_flt

    all_filepath = output_dirpath / 'all_profile.nc'
    dusty_filepath = output_dirpath / 'dusty_profile.nc'
    clean_filepath = output_dirpath / 'clean_profile.nc'
    ds_all.to_netcdf(str(all_filepath), encoding=encoding)
    dusty_ds.to_netcdf(str(dusty_filepath), encoding=encoding)
    clean_ds.to_netcdf(str(clean_filepath), encoding=encoding)

if __name__ == '__main__':
    start_time = config['start_time']
    end_time = config['end_time']
    dates = pd.date_range(start_time, end_time, freq='D')
    # 只选取春季的数据
    dates = dates[dates.month.isin([3, 4, 5])]

    # 创建输出目录.
    single_dirpath = Path(config['temp_dirpath']) / 'single'
    merged_dirpath = Path(config['temp_dirpath']) / 'merged'
    recreate_dir(single_dirpath)
    recreate_dir(merged_dirpath)

    # 从所有DPR文件中提取出降水廓线数据.
    p = Pool(8)
    for date in dates:
        for DPR_filepath in get_DPR_filepaths(date):
            p.apply_async(
                func=extract_one_file,
                args=(DPR_filepath, single_dirpath)
            )
    p.close()
    p.join()

    # 合并每个文件的结果,并从中分出dusty和clean两组数据.
    concat_and_divide(single_dirpath, merged_dirpath)
