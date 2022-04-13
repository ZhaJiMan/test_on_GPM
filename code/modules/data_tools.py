from pathlib import Path

import h5py
from pyhdf.SD import SD, SDC
import numpy as np
import pandas as pd

class ReaderDPR:
    '''读取2ADPR文件的类.'''
    def __init__(self, filepath, mode='NS'):
        '''打开文件.'''
        self.f = h5py.File(filepath)
        self.mode = Path(mode)

    def close(self):
        '''关闭文件.'''
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def read_ds(self, dsname, mask=True):
        '''读取dataset, 可决定是否设置缺测.'''
        ds = self.f[str(self.mode / dsname)]
        data = ds[:]

        if mask:
            fill_value = ds.attrs['_FillValue']
            data[np.isclose(data, fill_value)] = np.nan

        return data

    def read_lonlat(self):
        '''读取经纬度.'''
        lon = self.f[str(self.mode / 'Longitude')][:]
        lat = self.f[str(self.mode / 'Latitude')][:]

        return lon, lat

    def read_time(self):
        '''读取沿scan方向的时间, 返回DatetimeIndex.'''
        dirpath = self.mode / 'ScanTime'
        year = self.f[str(dirpath / 'Year')][:]
        month = self.f[str(dirpath / 'Month')][:]
        day = self.f[str(dirpath / 'DayOfMonth')][:]
        hour = self.f[str(dirpath / 'Hour')][:]
        minute = self.f[str(dirpath / 'Minute')][:]
        second = self.f[str(dirpath / 'Second')][:]
        millisecond = self.f[str(dirpath / 'MilliSecond')][:]
        df = pd.DataFrame({
            'year': year, 'month': month, 'day': day,
            'hour': hour, 'minute': minute, 'second': second,
            'ms': millisecond
        })

        # 将Series结果转为Index, 与其它函数保持一致.
        return pd.DatetimeIndex(pd.to_datetime(df))

    def read_rtype(self):
        '''
        读取雨型并解译其位信息.

        返回数组形状为(nscan, nray, 8), 最后一维对应于从左向右的位数.
        例如rtype[:, :, 0]表示main rain type.
        '''
        typePrecip = self.read_ds('CSF/typePrecip', mask=False)
        typePrecip = typePrecip[:, :, None]
        divisors = 10**np.arange(8)[::-1]
        divisors = divisors.astype(typePrecip.dtype)
        rtype = np.where(
            typePrecip > 0,
            typePrecip // divisors % 10,
            typePrecip
        )  # 保留no rain与missing.

        return rtype

class ReaderMYD:
    '''读取MYD04_L2文件的类.'''
    def __init__(self, filepath):
        '''打开文件.'''
        self.sd = SD(filepath, SDC.READ)

    def close(self):
        '''关闭文件'''
        self.sd.end()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def search_sds(self, keyword):
        '''搜索SDS名, 无视大小写.'''
        sdsnames = sorted(self.sd.datasets().keys())
        for sdsname in sdsnames:
            if keyword.lower() in sdsname.lower():
                print(sdsname)

    def read_sds(self, sdsname, scale=True):
        '''读取SDS, 可决定是否缩放并设置缺测.'''
        sds = self.sd.select(sdsname)
        data = sds[:]

        # 使用valid_range要比_FillValue更靠谱.
        if scale:
            attrs = sds.attributes()
            vmin, vmax = attrs['valid_range']
            scale_factor = attrs['scale_factor']
            add_offset = attrs['add_offset']
            data = np.where(
                (data >= vmin) & (data <= vmax),
                (data - add_offset) * scale_factor,
                np.nan
            )

        return data

    def read_lonlat(self):
        '''读取经纬度, 可能含缺测.'''
        lon = self.read_sds('Longitude')
        lat = self.read_sds('Latitude')

        return lon, lat

    def read_time(self):
        '''读取沿swath方向的TAI时间, 返回DatetimeIndex, 可能含缺测.'''
        second = self.read_sds('Scan_Start_Time')[:, 0]
        time = pd.to_datetime(second, unit='s', origin='1993-01-01')

        return time

class ReaderCAL:
    '''读取CAL_LID_L2_VFM文件的类.'''
    def __init__(self, filepath):
        '''打开文件.'''
        self.sd = SD(filepath, SDC.READ)

    def close(self):
        '''关闭文件.'''
        self.sd.end()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def read_ftype(self):
        '''
        读取特征分类并解译其位信息.

        返回数组形状为(nrec, 545, 7), 第一维是记录数, 第二维是高度维,
        第三维对应于特征分类从右到左的位数.
        例如fcf[:, :, 0]表示feature type.
        '''
        data = self.sd.select('Feature_Classification_Flags')[:]
        # 不同高度层中都只选取第一个profile来代表5km范围内的值.
        fcf1 = data[:, 1165:1455]
        fcf2 = data[:, 165:365]
        fcf3 = data[:, 0:55]
        # 倒转高度维,使序号增加时对应的高度也增加.
        fcf = np.hstack([fcf3, fcf2, fcf1])[:, ::-1]

        # 利用位运算进行解译.
        shifts = [0, 3, 5, 7, 9, 12, 13]
        bits = [7, 3, 3, 3, 7, 1, 7]
        ftype = (fcf[:, :, None] >> shifts) & bits

        return ftype

    def read_lonlat(self):
        '''读取经纬度.'''
        lon = self.sd.select('Longitude')[:, 0]
        lat = self.sd.select('Latitude')[:, 0]

        return lon, lat

    def read_hgt(self):
        '''读取FCF对应的垂直高度, 单位为km.'''
        hgt1 = (np.arange(290) + 0.5) * 0.03 - 0.5
        hgt2 = (np.arange(200) + 0.5) * 0.06 + 8.2
        hgt3 = (np.arange(55) + 0.5) * 0.18 + 20.2
        hgt = np.concatenate([hgt1, hgt2, hgt3])

        return hgt

    def read_time(self):
        '''读取TAI时间, 返回DatetimeIndex.'''
        second = self.sd.select('Profile_Time')[:, 0]
        time = pd.to_datetime(second, unit='s', origin='1993-01-01')

        return time

def get_ftype_with_dust(ftype, polluted=True):
    '''读取ftype中的feature_type信息, 并将沙尘像元赋值为8.'''
    type_main = ftype[..., 0]
    type_sub = ftype[..., 4]
    mask_main = type_main == 3
    # 是否计入polluted dust.
    if polluted:
        mask_sub = (type_sub == 2) | (type_sub == 5)
    else:
        mask_sub = (type_sub == 2)
    mask_dust = mask_main & mask_sub
    ftype = np.where(mask_dust, 8, type_main)

    return ftype

def check_existence(filepath):
    '''若文件不存在, 报错并提示.'''
    if not filepath.exists():
        raise FileNotFoundError(str(filepath))

def get_DPR_filepaths_one_day(date):
    '''返回某一天内所有的2ADPR文件的路径.'''
    yyyy = date.strftime('%Y')
    yyyymm = date.strftime('%Y%m')
    yyyymmdd = date.strftime('%Y%m%d')

    # DPR数据在服务器上是分两个位置存储的, 需要分开处理.
    if date < pd.to_datetime('2019-05-30'):
        dirpath = Path('/data00/0/GPM/DPR/V06')
    else:
        dirpath = Path('/data04/0/gpm/DPR/V06')

    sub_dirpath = dirpath / yyyy / yyyymm
    for filepath in sub_dirpath.glob(
        f'2A.GPM.DPR.V8-20180723.{yyyymmdd}-*.V06A.HDF5'
    ):
        yield filepath

def get_CAL_filepaths_one_day(date):
    '''返回某一天内所有的CAL_LID_L2_VFM文件的路径.'''
    yyyy = date.strftime('%Y')
    yyyymm = date.strftime('%Y%m')
    yyyymmdd = date.strftime('%Y-%m-%d')

    dirpath = Path('/d4/wangj/dust_precipitation/data/CALIPSO')
    sub_dirpath = dirpath / yyyy / yyyymm
    for filepath in sub_dirpath.glob(
        f'CAL_LID_L2_VFM-Standard-V4-20.{yyyymmdd}T*_Subset.hdf'
    ):
        yield filepath

def get_MYD_filepaths_one_day(date):
    '''返回某一天内所有的MYD04_L2文件的路径.'''
    yyyy = date.strftime('%Y')
    jjj = date.strftime('%j')
    yyyyjjj = date.strftime('%Y%j')

    dirpath = Path('/data00/0/MODIS/MYD04_L2/061')
    sub_dirpath = dirpath / yyyy / jjj
    for filepath in sub_dirpath.glob(f'MYD04_L2.A{yyyyjjj}.*.061.*.hdf'):
        yield filepath

def get_ERA5_filepath_one_day(date, varname):
    '''返回某一天指定变量的ERA5文件路径.'''
    yyyy = date.strftime('%Y')
    yyyymmdd = date.strftime('%Y%m%d')

    dirpath = Path('/data04/0/ERA5_NANJING') / varname
    sub_dirpath = dirpath / yyyy
    filename = f'era5.{varname}.{yyyymmdd}.nc'
    filepath = sub_dirpath / filename
    check_existence(filepath)

    return filepath

def get_MERRA2_filepath_one_day(date):
    '''返回某一天的MERRA2文件路径.'''
    yyyy = date.strftime('%Y')
    yyyymmdd = date.strftime('%Y%m%d')

    dirpath = Path('/data04/0/Backup/zhuhx/Merra2_2014_2020_spring')
    sub_dirpath = dirpath / yyyy
    filename = f'MERRA2_400.tavg1_2d_aer_Nx.{yyyymmdd}.nc4'
    filepath = sub_dirpath / filename
    check_existence(filepath)

    return filepath

def to_ENV_filepath(filepath_DPR):
    '''根据DPR文件的路径获取对应的ENV文件的路径.'''
    filename_DPR = filepath_DPR.name
    parts = filename_DPR.split('.')
    yyyy = parts[4][:4]

    dirpath_GPM = filepath_DPR.parents[4]
    dirpath_ENV = dirpath_GPM / 'ENV' / 'V06'
    parts[0] = '2A-ENV'
    filename_ENV = '.'.join(parts)
    filepath_ENV = dirpath_ENV / yyyy / filename_ENV
    check_existence(filepath_ENV)

    return filepath_ENV

def to_SLH_filepath(filepath_DPR):
    '''根据DPR文件的路径获取对应的SLH文件的路径.'''
    filename_DPR = filepath_DPR.name
    parts = filename_DPR.split('.')
    yyyy = parts[4][:4]

    # 两个服务器上的SLH文件的版本存在差异, 需要分开处理.
    dirpath_GPM = filepath_DPR.parents[4]
    if dirpath_GPM == Path('/data00/0/GPM'):
        parts[-2] = 'V06A'
    elif dirpath_GPM  == Path('/data04/0/gpm'):
        parts[-2] = 'V06B'

    dirpath_SLH = dirpath_GPM / 'SLH' / 'V06'
    parts[3] = 'GPM-SLH'
    filename_SLH = '.'.join(parts)
    filepath_SLH = dirpath_SLH / yyyy / filename_SLH
    check_existence(filepath_SLH)

    return filepath_SLH

if __name__ == '__main__':
    # 测试DPR文件.
    filepath_DPR = '/data00/0/GPM/DPR/V06/2017/201705/2A.GPM.DPR.V8-20180723.20170504-S153418-E170652.018078.V06A.HDF5'
    with ReaderDPR(filepath_DPR) as f:
        lon, lat = f.read_lonlat()
        time = f.read_time()
        Rr3D = f.read_ds('SLV/precipRate')
        rtype = f.read_rtype()[:, :, 0]
    start_time, end_time = time[[0, -1]].strftime('%Y-%m-%d %H:%M')
    print('For 2ADPR file:')
    print('lon range:', lon.min(), lon.max())
    print('lat range:', lat.min(), lat.max())
    print('time range:', start_time, 'to', end_time)
    print('rain range:', np.nanmin(Rr3D), np.nanmax(Rr3D))

    # 测试MYD04_L2文件.
    filepath_MYD = '/data00/0/MODIS/MYD04_L2/061/2017/124/MYD04_L2.A2017124.0510.061.2018032091945.hdf'
    with ReaderMYD(filepath_MYD) as f:
        lon, lat = f.read_lonlat()
        time = f.read_time()
        aod = f.read_sds('Deep_Blue_Aerosol_Optical_Depth_550_Land')
    start_time, end_time = time.dropna()[[0, -1]].strftime('%Y-%m-%d %H:%M')
    print('For MYD04_L2 file:')
    print('lon range:', np.nanmin(lon), np.nanmax(lon))
    print('lat range:', np.nanmin(lat), np.nanmax(lat))
    print('time range:', start_time, 'to', end_time)
    print('aod range:', np.nanmin(aod), np.nanmax(aod), '\n')

    # 测试CALIPSO
    filepath_CAL = '/d4/wangj/dust_precipitation/data/CALIPSO/2017/201705/CAL_LID_L2_VFM-Standard-V4-20.2017-05-04T18-42-46ZN_Subset.hdf'
    with ReaderCAL(filepath_CAL) as f:
        lon, lat = f.read_lonlat()
        hgt = f.read_hgt()
        time = f.read_time()
        ftype = f.read_ftype()[:, :, 0]
    start_time, end_time = time[[0, -1]].strftime('%Y-%m-%d %H:%M')
    print('For CAL_LID_L2_VFM file:')
    print('lon range:', lon.min(), lon.max())
    print('lat range:', lat.min(), lat.max())
    print('height range:', hgt.min(), hgt.max())
    print('time range:', start_time, 'to', end_time)
    print('ftype range:', ftype.min(), ftype.max(), '\n')

    # 测试搜索文件.
    date = pd.to_datetime('2017-05-03')
    print(len(list(get_DPR_filepaths_one_day(date))))
    print(len(list(get_CAL_filepaths_one_day(date))))
    print(len(list(get_MYD_filepaths_one_day(date))))
    print(get_ERA5_filepath_one_day(date, 'geopotential'))
    print(get_ERA5_filepath_one_day(date, 'relative_humidity'))

    # 测试DPR文件路径转换.
    filepath_DPR = Path(filepath_DPR)
    print(to_ENV_filepath(filepath_DPR))
    print(to_SLH_filepath(filepath_DPR))
