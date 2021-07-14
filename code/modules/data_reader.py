#------------------------------------------------------------------------------
# 2021/04/26
# 存放一些用于读取数据的函数和类.
#------------------------------------------------------------------------------
import h5py
from pyhdf.SD import SD, SDC

import numpy as np
import pandas as pd

def read_GPM_time(f, group):
    '''读取GMI或DPR文件的时间并返回DatetimeIndex.'''
    year = f[group + '/ScanTime/Year'][:]
    month = f[group + '/ScanTime/Month'][:]
    day = f[group + '/ScanTime/DayOfMonth'][:]
    hour = f[group + '/ScanTime/Hour'][:]
    minute = f[group + '/ScanTime/Minute'][:]
    second = f[group + '/ScanTime/Second'][:]
    millisecond = f[group + '/ScanTime/MilliSecond'][:]
    df = pd.DataFrame({
        'year': year, 'month': month, 'day': day,
        'hour': hour, 'minute': minute, 'second': second,
        'ms': millisecond
    })

    # 将Series结果转为Index,与其它函数保持一致.
    return pd.DatetimeIndex(pd.to_datetime(df))

def read_GMI_time(f):
    '''读取GMI文件的时间并返回DatetimeIndex.'''
    return read_GPM_time(f, 'S1')

def read_DPR_time(f):
    '''读取DPR文件的时间并返回DatetimeIndex.'''
    return read_GPM_time(f, 'NS')

class reader_for_MYD:
    '''读取MYD04_L2文件数据的类.'''
    def __init__(self, filepath):
        '''读取文件.'''
        self.sd = SD(filepath, SDC.READ)

    def read_lonlat(self):
        '''读取经纬度数据.直接返回numpy数组,应该没有缺测值.'''
        lon = self.sd.select('Longitude')[:]
        lat = self.sd.select('Latitude')[:]

        return lon, lat

    def read_sds(self, sdsname):
        '''读取scientific dataset变量,同时进行scale和mask操作.'''
        sds = self.sd.select(sdsname)
        data = sds[:]

        attrs = sds.attributes()
        scale_factor = attrs['scale_factor']
        add_offset = attrs['add_offset']
        fill_value = attrs['_FillValue']

        data = np.ma.masked_values(data, fill_value)
        data = (data - add_offset) * scale_factor

        return data

    def read_time(self):
        '''读取沿swath的时间,返回DatetimeIndex.'''
        second = self.sd.select('Scan_Start_Time')[:, 0]
        # 准确来说是TAI时间,跟UTC时间有数十秒的误差.
        time = pd.to_datetime(second, unit='s', origin='1993-01-01')

        return time

    def close(self):
        '''关闭文件.'''
        self.sd.end()

def unpack(vfm):
    '''从vfm数组中提取出类型数据.额外设置沙尘对应8.'''
    feature_type = vfm & 7
    sub_type = (vfm >> 9) & 7
    flag_dust = (feature_type == 3) & (sub_type == 2)
    # 让1~7对应于feature_type,8对应于dust.
    new_type = np.where(flag_dust, 8, feature_type)

    return new_type

class reader_for_CAL:
    '''读取CAL_LID_L2_VFM文件的类.'''
    def __init__(self, filepath):
        '''读取文件.'''
        self.sd = SD(filepath, SDC.READ)

    def read_lonlat(self):
        '''读取经纬度.'''
        lon = self.sd.select('Longitude')[:, 0]
        lat = self.sd.select('Latitude')[:, 0]

        return lon, lat

    def read_time(self):
        '''读取时间,返回DatetimeIndex.'''
        second = self.sd.select('Profile_Time')[:, 0]
        # 准确来说是TAI时间,跟UTC时间有数十秒的误差.
        time = pd.to_datetime(second, unit='s', origin='1993-01-01')

        return time

    def read_vfm(self):
        '''读取vertical feature mask.'''
        data = self.sd.select('Feature_Classification_Flags')[:]
        # 不同高度层中都只选取第一个profile来代表5km范围内的值.
        vfm1 = data[:, 1165:1455]
        vfm2 = data[:, 165:365]
        vfm3 = data[:, 0:55]
        # 倒转高度维,使序号增加时对应的高度也增加.
        vfm = np.hstack([vfm3, vfm2, vfm1])[:, ::-1]

        return unpack(vfm)

    def read_hgt(self):
        '''读取VFM对应的垂直高度,单位为km.'''
        hgt1 = (np.arange(290) + 0.5) * 0.03 - 0.5
        hgt2 = (np.arange(200) + 0.5) * 0.06 + 8.2
        hgt3 = (np.arange(55) + 0.5) * 0.18 + 20.2
        hgt = np.concatenate([hgt1, hgt2, hgt3])

        return hgt

    def close(self):
        '''关闭文件.'''
        self.sd.end()
