'''
2022-02-18
提取研究区域内的DPR降水数据.

流程:
- 依次读取每一天的DPR文件, 判断是否有降水像元落入研究区域中.
- 继续读取降水数据, ENV气温和SLH潜热数据.
- 将廓线数据从高度坐标转换到温度坐标上, 将雨顶高度转为雨顶温度.
- 调整雨型, 区分层云, 深对流和浅云降水.
- 将以上数据保存到文件中, 再将所有这些文件合并为单个文件.

输入:
- 服务器上的2A-DPR, 2A-ENV和2A-SLH文件.

输出:
- 含有降水和潜热数据的npz文件, 具体含有哪些变量请见代码.
  每个在研究区域观测到降水的DPR文件对应一个nc文件.
- 上一条的所有文件合并而成的单个文件.

参数:
- time_start: 选取DPR记录的起始时间.
- time_end: 选取DPR记录的结束时间.
- extents_DPR: 研究区域的范围.
'''
import json
from pathlib import Path
from multiprocessing import Pool
import sys
sys.path.append('../modules')

import h5py
import numpy as np
import pandas as pd
import xarray as xr

import helper_tools
import data_tools
import region_tools
import profile_tools

# 读取配置文件, 将config作为全局变量.
with open('config.json') as f:
    config = json.load(f)

def extract_data_from_one_file(filepath_DPR, dirpath_output):
    '''
    提取一个DPR文件落入研究区域内的降水数据, 进行处理并保存.

    输出文件以DPR文件的轨道号命名, 保存为npz格式.
    若研究区域内无像元落入, 则跳过该DPR文件.
    '''
    # 尝试寻找DPR文件对应的ENV和SLH文件, 若找不到则停止处理.
    try:
        filepath_ENV = data_tools.to_ENV_filepath(filepath_DPR)
        filepath_SLH = data_tools.to_SLH_filepath(filepath_DPR)
    except FileNotFoundError:
        return None

    # 分阶段读取数据, 先读取经纬度.
    f = data_tools.ReaderDPR(str(filepath_DPR))
    Longitude, Latitude = f.read_lonlat()
    # 要求有像元落入研究区域.
    extents_DPR = config['extents_DPR']
    mask_DPR = region_tools.region_mask(Longitude, Latitude, extents_DPR)
    if not mask_DPR.any():
        f.close()
        return None
    # 要求研究区域内的陆面有降水像元.
    precipRateNearSurface = f.read_ds('SLV/precipRateNearSurface')
    landSurfaceType = f.read_ds('PRE/landSurfaceType', mask=False)
    mask_rain = precipRateNearSurface > 0
    mask_land = landSurfaceType > 99
    mask_all = mask_DPR & mask_rain & mask_land
    if not mask_all.any():
        f.close()
        return None

    Longitude = Longitude[mask_all]
    Latitude = Latitude[mask_all]
    precipRateNearSurface = precipRateNearSurface[mask_all]
    # 读取高度量.
    elevation = f.read_ds('PRE/elevation')[mask_all]
    heightStormTop = f.read_ds('PRE/heightStormTop')[mask_all]
    heightZeroDeg = f.read_ds('VER/heightZeroDeg')[mask_all]
    # 读取降水量.
    rainType = f.read_rtype()[mask_all, 0]
    flagShallowRain = f.read_ds('CSF/flagShallowRain', mask=False)[mask_all]
    precipRate = f.read_ds('SLV/precipRate')[mask_all, :]
    zFactorCorrected = f.read_ds('SLV/zFactorCorrected')[mask_all, :]
    # 读取DSD量.
    paramDSD = f.read_ds('SLV/paramDSD')[mask_all, :, :]
    Nw = paramDSD[:, :, 0]
    Dm = paramDSD[:, :, 1]
    f.close()

    # 读取ENV数据.
    with h5py.File(str(filepath_ENV)) as f:
        airTemperature = f['NS/VERENV/airTemperature'][:][mask_all, :]

    # 读取SLH数据.
    with h5py.File(str(filepath_SLH)) as f:
        latentHeating = f['Swath/latentHeating'][:][mask_all, :]
    latentHeating[np.isclose(latentHeating, -9999.9)] = np.nan

    # 将高度量的单位转为km.
    elevation /= 1000
    heightStormTop /= 1000
    heightZeroDeg /= 1000
    # 将气温的单位转为摄氏度.
    airTemperature -= 273.15

    # 设置DPR使用的高度, 注意数值从大到小.
    nbin = 176
    dh = 0.125  # 单位为km.
    height_DPR = (np.arange(nbin) + 0.5)[::-1] * dh
    # 设置SLH使用的高度, 注意数值从小到大.
    nlayer = 80
    dh = 0.25  # 单位为km.
    height_SLH = (np.arange(nlayer) + 0.5) * dh

    # 将SLH数据的垂直分辨率线性插值到与DPR相同.
    # 保留NaN, 外插部分也用NaN填充.
    f = lambda profile: np.interp(
        x=height_DPR, xp=height_SLH, fp=profile,
        left=np.nan, right=np.nan
    )
    latentHeating = np.apply_along_axis(f, axis=-1, arr=latentHeating)

    # 给出温度坐标.
    tmin = -60
    tmax = 20
    dt = 0.5
    nt = int((tmax - tmin) / dt) + 1
    temp = np.linspace(tmin, tmax, nt)

    # 转换廓线数据.
    converter = profile_tools.ProfileConverter(airTemperature, height_DPR)
    precipRate_t = converter.convert3d(precipRate, temp)
    zFactorCorrected_t = converter.convert3d(zFactorCorrected, temp)
    Nw_t = converter.convert3d(Nw, temp)
    Dm_t = converter.convert3d(Dm, temp)
    latentHeating_t = converter.convert3d(latentHeating, temp)
    # 雨顶高度转为雨顶温度.
    tempStormTop = converter.convert2d(heightStormTop)

    # 划分雨型.
    descr = (
        '1 for stratiform\n'
        '2 for deep convective\n'
        '3 for shallow convective\n'
        '4 for other\n'
        '-9999 for missing\n'
        '-1111 for no rain'
    )
    rainType[rainType == 3] = 4
    rainType[flagShallowRain > 0] = 3
    rainType[(rainType == 2) & (flagShallowRain == 0)] = 2

    # 为了便于使用, 将高度坐标倒转.
    height_DPR = height_DPR[::-1]
    precipRate = precipRate[:, ::-1]
    zFactorCorrected = zFactorCorrected[:, ::-1]
    Nw = Nw[:, ::-1]
    Dm = Dm[:, ::-1]
    latentHeating = latentHeating[:, ::-1]

    # 设置月份.
    parts = filepath_DPR.stem.split('.')
    mm = int(parts[4][4:6])
    month = np.full(Longitude.shape[0], mm)

    # 保存为nc文件.
    dim1d = 'npoint'
    dim2d = ['npoint', 'height']
    dim2d_t = ['npoint', 'temp']
    eci = {'dtype': 'int32'}
    ecf = {'dtype': 'float32'}
    ds = xr.Dataset(
        data_vars={
            'lon': (dim1d, Longitude, {'units': 'degrees_east'}, ecf),
            'lat': (dim1d, Latitude, {'units': 'degrees_north'}, ecf),
            'month': (dim1d, month, {'units': 'months'}, eci),
            'elevation': (dim1d, elevation, {'units': 'km'}, ecf),
            'heightStormTop': (dim1d, heightStormTop, {'units': 'km'}, ecf),
            'tempStormTop': (dim1d, tempStormTop, {'units': 'celsius'}, ecf),
            'heightZeroDeg': (dim1d, heightZeroDeg, {'units': 'km'}, ecf),
            'rainType': (dim1d, rainType, {'units': 'none', 'description': descr}, eci),
            'precipRateNearSurface': (dim1d, precipRateNearSurface, {'units': 'mm/hr'}, ecf),
            'precipRate': (dim2d, precipRate, {'units': 'mm/hr'}, ecf),
            'zFactorCorrected': (dim2d, zFactorCorrected, {'units': 'dBZ'}, ecf),
            'Nw': (dim2d, Nw, {'units': '10log10(Nw)'}, ecf),
            'Dm': (dim2d, Dm, {'units': 'mm'}, ecf),
            'latentHeating': (dim2d, latentHeating, {'units': 'K/hr'}, ecf),
            'precipRate_t': (dim2d_t, precipRate_t, {'units': 'mm/hr'}, ecf),
            'zFactorCorrected_t': (dim2d_t, zFactorCorrected_t, {'units': 'dBZ'}, ecf),
            'Nw_t': (dim2d_t, Nw_t, {'units': '10log10(Nw)'}, ecf),
            'Dm_t': (dim2d_t, Dm_t, {'units': 'mm'}, ecf),
            'latentHeating_t': (dim2d_t, latentHeating_t, {'units': 'K/hr'}, ecf)
        },
        coords={
            'height': ('height', height_DPR, {'units': 'km'}),
            'temp': ('temp', temp, {'units': 'celsius'})
        }
    )
    orbit_number = parts[-2]
    filepath_output = dirpath_output / (orbit_number + '.nc')
    ds.to_netcdf(str(filepath_output))

def merge_files(dirpath_input, filepath_output):
    '''合并extract_data_from_one_file函数生成的所有文件.'''
    datasets = []
    for filepath_input in sorted(dirpath_input.iterdir()):
        ds = xr.load_dataset(str(filepath_input))
        datasets.append(ds)
    merged = xr.concat(datasets, dim='npoint')
    merged.to_netcdf(str(filepath_output))

if __name__ == '__main__':
    time_start = config['time_start']
    time_end = config['time_end']
    dates = pd.date_range(time_start, time_end, freq='D')
    # 只选取春季的数据.
    dates = dates[dates.month.isin([3, 4, 5])]

    # 创建输出目录.
    dirpath_output = Path(config['dirpath_data'], 'DPR_region')
    helper_tools.renew_dir(dirpath_output)
    dirpath_single = dirpath_output / 'single'
    helper_tools.new_dir(dirpath_single)

    # 从所有DPR文件中提取出降水数据.
    p = Pool(10)
    for date in dates:
        for filepath_DPR in data_tools.get_DPR_filepaths_one_day(date):
            p.apply_async(
                extract_data_from_one_file,
                args=(filepath_DPR, dirpath_single)
            )
    p.close()
    p.join()

    # 合并前面产生的所有文件.
    merge_files(dirpath_single, dirpath_output / 'data_DPR.nc')
