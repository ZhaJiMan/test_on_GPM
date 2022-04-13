'''
2022-04-12
提取和处理每个污染和清洁个例对应的降水数据.

流程:
- 根据掩膜文件读取DPR, ENV, CSH, SLH和VPH数据.
- 通过线性插值使CSH和SLH的垂直分辨率与DPR数据一致.
- 将三维廓线数据从高度坐标转换到温度坐标.
- 读取GMI数据计算PCT89, 通过最邻近插值匹配到DPR像元上.
- 读取ERA5数据计算TCWV和MFD, 再线性插值到DPR像元上.
- 将所有数据保存为nc文件, 并将文件路径写入个例记录中.
- 为方便进行合成分析, 分别将所有污染个例和清洁个例的文件沿样本点维度连接起来,
- 构成两个单独的文件.

输入:
- cases_dusty.json
- cases_clean.json

输出:
- 个例数据文件.
- cases_dusty.json
- cases_clean.json

参数:
- extents_DPR: 研究范围.

注意:
- 脚本使用了多进程.
- 除了整型的雨型, 其它数据的缺测均用NaN表示.
- 高度坐标廓线的nbin维已经经过颠倒处理.
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
from scipy.ndimage import gaussian_filter

import helper_tools
import data_tools
import region_tools
import profile_tools

# 读取配置文件, 将config作为全局变量.
with open('config.json') as f:
    config = json.load(f)

def extract_data_from_one_case(case, dirpath_output):
    '''提取和处理个例的降水数据并保存为nc文件, 返回写入了路径的个例记录.'''
    # 读取DPR数据.
    mask_case = np.load(case['filepath_mask'])
    with data_tools.ReaderDPR(case['filepath_DPR']) as f:
        # 读取经纬度.
        lon_DPR, lat_DPR = f.read_lonlat()
        lon_DPR = lon_DPR[mask_case]
        lat_DPR = lat_DPR[mask_case]
        # 读取高度量.
        elevation = f.read_ds('PRE/elevation')[mask_case]
        heightStormTop = f.read_ds('PRE/heightStormTop')[mask_case]
        heightZeroDeg = f.read_ds('VER/heightZeroDeg')[mask_case]
        # 读取降水量.
        rainType = f.read_rtype()[mask_case, 0]
        flagShallowRain = f.read_ds('CSF/flagShallowRain', mask=False)[mask_case]
        precipRateNearSurface = f.read_ds('SLV/precipRateNearSurface')[mask_case]
        precipRate = f.read_ds('SLV/precipRate')[mask_case, :]
        zFactorCorrected = f.read_ds('SLV/zFactorCorrected')[mask_case, :]
        # 读取DSD量.
        paramDSD = f.read_ds('SLV/paramDSD')[mask_case, :, :]
        Nw = paramDSD[:, :, 0]
        Dm = paramDSD[:, :, 1]

    # 读取ENV数据.
    with h5py.File(case['filepath_ENV']) as f:
        airTemperature = f['NS/VERENV/airTemperature'][:][mask_case, :]

    # 将高度量的单位转为km.
    elevation /= 1000
    heightStormTop /= 1000
    heightZeroDeg /= 1000
    # 将气温的单位转为摄氏度.
    airTemperature -= 273.15

    # 读取CSH和SLH数据.
    with h5py.File(case['filepath_CSH']) as f:
        csh = f['Swath/latentHeating'][:][mask_case, :]
    with h5py.File(case['filepath_SLH']) as f:
        slh = f['Swath/latentHeating'][:][mask_case, :]
    # 需要手动设置缺测.
    fill_value = -9999.9
    csh[np.isclose(csh, fill_value)] = np.nan
    slh[np.isclose(slh, fill_value)] = np.nan

    # 读取VPH数据. 注意高度维递增.
    with xr.open_dataset(case['filepath_VPH']) as ds:
        vph = ds['latentHeating'].values[mask_case, :]

    # 设置DPR使用的高度, 注意数值从大到小.
    nbin = 176
    dh = 0.125  # 单位为km.
    height_DPR = (np.arange(nbin) + 0.5)[::-1] * dh
    # 设置CSH和SLH使用的高度, 注意数值从小到大.
    nlayer = 80
    dh = 0.25  # 单位为km.
    height_LH = (np.arange(nlayer) + 0.5) * dh

    # 将CSH和SLH数据的垂直分辨率线性插值到与DPR相同.
    # 保留NaN, 外插部分也用NaN填充.
    f = lambda profile: np.interp(
        x=height_DPR, xp=height_LH, fp=profile,
        left=np.nan, right=np.nan
    )
    csh = np.apply_along_axis(f, axis=-1, arr=csh)
    slh = np.apply_along_axis(f, axis=-1, arr=slh)

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
    csh_t = converter.convert3d(csh, temp)
    slh_t = converter.convert3d(slh, temp)
    vph_t = converter.convert3d(vph[:, ::-1], temp)
    # 雨顶高度转为雨顶温度.
    tempStormTop = converter.convert2d(heightStormTop)

    # 为了便于使用, 将DPR数据的高度坐标倒转.
    height_DPR = height_DPR[::-1]
    precipRate = precipRate[:, ::-1]
    zFactorCorrected = zFactorCorrected[:, ::-1]
    airTemperature = airTemperature[:, ::-1]
    Nw = Nw[:, ::-1]
    Dm = Dm[:, ::-1]
    csh = csh[:, ::-1]
    slh = slh[:, ::-1]

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

    # 读取GMI数据.
    with h5py.File(case['filepath_GMI']) as f:
        lon_GMI = f['S1/Longitude'][:]
        lat_GMI = f['S1/Latitude'][:]
        Tb = f['S1/Tb'][:]
    # 用extents_map截取数据.
    extents_map = config['extents_map']
    mask_map = region_tools.region_mask(lon_GMI, lat_GMI, extents_map)
    lon_GMI = lon_GMI[mask_map]
    lat_GMI = lat_GMI[mask_map]
    Tb = Tb[mask_map, :]
    Tb89V = Tb[:, 7]
    Tb89H = Tb[:, 8]

    # 将GMI亮温匹配到DPR像元上.
    pct89 = 1.818 * Tb89V - 0.818 * Tb89H
    pct89 = profile_tools.match_GMI_to_DPR(
        lon_GMI, lat_GMI, lon_DPR, lat_DPR, pct89
    )

    # 读取ERA5数据.
    ds = xr.load_dataset(case['filepath_ERA5'])
    ds['w'] = ds.w.sel(level=500)
    ds['level'] = ds.level * 100  # 气压单位转为Pa.

    # 计算积分量.
    g = 9.8
    a = 6371E3
    tcwv = ds.q.integrate('level') / g
    uq = ds.u * ds.q
    vq = ds.v * ds.q
    coslat = np.cos(np.deg2rad(ds.latitude))
    duqdlon = np.rad2deg(uq.differentiate('longitude'))
    dvqdlat = np.rad2deg(vq.differentiate('latitude'))
    div = (duqdlon / coslat + dvqdlat) / a
    mfd = div.integrate('level') / g
    # 进行平滑
    tcwv.values = gaussian_filter(tcwv, sigma=1)
    mfd.values = gaussian_filter(mfd, sigma=1)
    ds = ds.assign(tcwv=tcwv, mfd=mfd).drop_vars(['u', 'v', 'q'])

    # 通过线性插值得到DPR像元位置ERA5的变量.
    ds = ds.interp(
        longitude=xr.DataArray(lon_DPR, dims='npoint'),
        latitude=xr.DataArray(lat_DPR, dims='npoint')
    )
    w = ds.w.values
    tcwv = ds.tcwv.values
    mfd = ds.mfd.values
    cape = ds.cape.values

    # 设置月份.
    rain_time = pd.to_datetime(case['rain_time'])
    month = np.full(lon_DPR.shape[0], rain_time.month)

    # 保存为nc文件.
    dim1d = 'npoint'
    dim2d = ['npoint', 'height']
    dim2d_t = ['npoint', 'temp']
    eci = {'dtype': 'int32'}
    ecf = {'dtype': 'float32'}
    ds = xr.Dataset(
        data_vars={
            'lon': (dim1d, lon_DPR, {'units': 'degrees_east'}, ecf),
            'lat': (dim1d, lat_DPR, {'units': 'degrees_north'}, ecf),
            'month': (dim1d, month, {'units': 'months'}, eci),
            'elevation': (dim1d, elevation, {'units': 'km'}, ecf),
            'heightStormTop': (dim1d, heightStormTop, {'units': 'km'}, ecf),
            'tempStormTop': (dim1d, tempStormTop, {'units': 'celsius'}, ecf),
            'heightZeroDeg': (dim1d, heightZeroDeg, {'units': 'km'}, ecf),
            'rainType': (dim1d, rainType, {'units': 'none', 'description': descr}, eci),
            'precipRateNearSurface': (dim1d, precipRateNearSurface, {'units': 'mm/hr'}, ecf),
            'precipRate': (dim2d, precipRate, {'units': 'mm/hr'}, ecf),
            'zFactorCorrected': (dim2d, zFactorCorrected, {'units': 'dBZ'}, ecf),
            'airTemperature': (dim2d, airTemperature, {'units': '℃'}, ecf),
            'Nw': (dim2d, Nw, {'units': '10log10(Nw)'}, ecf),
            'Dm': (dim2d, Dm, {'units': 'mm'}, ecf),
            'csh': (dim2d, csh, {'units': 'K/hr'}, ecf),
            'slh': (dim2d, slh, {'units': 'K/hr'}, ecf),
            'vph': (dim2d, vph, {'units': 'K/hr'}, ecf),
            'precipRate_t': (dim2d_t, precipRate_t, {'units': 'mm/hr'}, ecf),
            'zFactorCorrected_t': (dim2d_t, zFactorCorrected_t, {'units': 'dBZ'}, ecf),
            'Nw_t': (dim2d_t, Nw_t, {'units': '10log10(Nw)'}, ecf),
            'Dm_t': (dim2d_t, Dm_t, {'units': 'mm'}, ecf),
            'csh_t': (dim2d_t, csh_t, {'units': 'K/hr'}, ecf),
            'slh_t': (dim2d_t, slh_t, {'units': 'K/hr'}, ecf),
            'vph_t': (dim2d_t, vph_t, {'units': 'K/hr'}, ecf),
            'pct89': (dim1d, pct89, {'units': 'K'}, ecf),
            'w': (dim1d, w, {'units': 'Pa/s'}, ecf),
            'tcwv': (dim1d, tcwv, {'units': 'kg/m2'}, ecf),
            'mfd': (dim1d, mfd, {'units': 'kg/m2/s'}, ecf),
            'cape': (dim1d, cape, {'units': 'J/kg'}, ecf)
        },
        coords={
            'height': ('height', height_DPR, {'units': 'km'}),
            'temp': ('temp', temp, {'units': 'celsius'})
        },
        attrs={
            'rain_time': case['rain_time'],
            'rain_center': case['rain_center'],
            'source': case['filepath_DPR'].split('/')[-1]
        }
    )
    filename_output = case['case_number'] + '.nc'
    filepath_output = dirpath_output / filename_output
    ds.to_netcdf(str(filepath_output))

    # 将路径写入个例.
    case = case.copy()
    case['filepath_profile'] = str(filepath_output)

    return case

def merge_files(dirpath_input, filepath_output):
    '''合并extract_data_from_one_case函数生成的所有文件.'''
    datasets = []
    for filepath_input in sorted(dirpath_input.iterdir()):
        ds = xr.load_dataset(str(filepath_input))
        datasets.append(ds)
    merged = xr.concat(datasets, dim='npoint', combine_attrs='drop')
    merged.to_netcdf(str(filepath_output))

if __name__ == '__main__':
    # 读取两组个例.
    dirpath_input = Path(config['dirpath_input'])
    filepath_dusty = dirpath_input / 'cases_dusty.json'
    filepath_clean = dirpath_input / 'cases_clean.json'
    with open(str(filepath_dusty)) as f:
        cases_dusty = json.load(f)
    with open(str(filepath_clean)) as f:
        cases_clean = json.load(f)

    # 创建输出目录.
    dirpath_output = Path(config['dirpath_data'], 'DPR_case')
    dirpath_dusty = dirpath_output / 'dusty'
    dirpath_clean = dirpath_output / 'clean'
    dirpath_merged = dirpath_output / 'merged'
    helper_tools.renew_dir(dirpath_dusty)
    helper_tools.renew_dir(dirpath_clean)
    helper_tools.renew_dir(dirpath_merged)

    # 提取每个个例的廓线数据, 并记录文件路径.
    p = Pool(10)
    cases_dusty = p.starmap(
        extract_data_from_one_case,
        [(case, dirpath_dusty) for case in cases_dusty]
    )
    cases_clean = p.starmap(
        extract_data_from_one_case,
        [(case, dirpath_clean) for case in cases_clean]
    )
    p.close()
    p.join()

    # 分别合并污染组和清洁组的廓线数据.
    filepath_merged1 = dirpath_merged / 'data_dusty.nc'
    filepath_merged2 = dirpath_merged / 'data_clean.nc'
    merge_files(dirpath_dusty, filepath_merged1)
    merge_files(dirpath_clean, filepath_merged2)

    # 重新写成json文件.
    with open(str(filepath_dusty), 'w') as f:
        json.dump(cases_dusty, f, indent=4)
    with open(str(filepath_clean), 'w') as f:
        json.dump(cases_clean, f, indent=4)
