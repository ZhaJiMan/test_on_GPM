import numpy as np

def get_extent_flag_both(lon, lat, extent):
    '''
    获取同时满足落入到给定方框内条件的经纬度数据的Boolean数组.

    Parameters
    ----------
    lon : ndarray
        经度坐标的数组.可以是一维或者二维的,要求形状与lat一致.

    lat : ndarray
        纬度坐标的数组.可以是一维或者二维的.要求形状与lon一致.

    extent : 4-tuple of float
        方框的经纬度范围[lonmin, lonmax, latmin, latmax].

    Returns
    -------
    flag : Boolean ndarray
        同时落入方框内的经纬度数据的Boolean数组.形状与lon和lat一致.
    '''
    lonmin, lonmax, latmin, latmax = extent
    flag = \
        (lon >= lonmin) & (lon <= lonmax) & \
        (lat >= latmin) & (lat <= latmax)

    return flag

def get_extent_flag_either(lon, lat, extent):
    '''
    分别获取满足落入到给定方框内条件的经纬度数据的两个Boolean数组.

    Parameters
    ----------
    lon : ndarray
        经度坐标的数组.要求维度为一维.

    lat : ndarray
        纬度坐标的数组.要求维度为一维,长度不需要和lon一致.

    extent : 4-tuple of float
        方框的经纬度范围[lonmin, lonmax, latmin, latmax].

    Returns
    -------
    lon_flag : Boolean ndarray
        满足落入到方框内的lon数组的Boolean数组.形状与lon一致.

    lat_flag : Boolean ndarray
        满足落入到方框内的lat数组的Boolean数组.形状与lat一致.
    '''
    lonmin, lonmax, latmin, latmax = extent
    lon_flag = (lon >= lonmin) & (lon <= lonmax)
    lat_flag = (lat >= latmin) & (lat <= latmax)

    return lon_flag, lat_flag
