import numpy as np
from scipy import stats

def region_mask(lon, lat, extents, AND=True):
    '''
    返回用于索引给定经纬度方框内数据的Boolean数组.

    Parameters
    ----------
    lon : ndarray
        经度数组. 若AND=True则要求形状与lat一致.

    lat : ndarray
        纬度数组. 若AND=True则要求形状与lon一致.

    extents : 4-tuple of float
        经纬度方框的范围[lonmin, lonmax, latmin, latmax].

    AND : bool
        决定经度和维度的Boolean数组是否进行求与计算.

    Returns
    -------
    mask : ndarray or tuple of ndarray
        用于索引数据的Boolean数组.
    '''
    lonmin, lonmax, latmin, latmax = extents
    mask_lon = (lon >= lonmin) & (lon <= lonmax)
    mask_lat = (lat >= latmin) & (lat <= latmax)
    if AND:
        return mask_lon & mask_lat
    else:
        return mask_lon, mask_lat

def grid_data(lon, lat, data, bins_lon, bins_lat):
    '''
    利用平均的方式对散点变量进行二维格点化.

    变量中的缺测均用NaN表示.

    Parameters
    ----------
    lon : (npt,) array_like
        一维经度数组.

    lat : (npt,) array_like
        一维纬度数组.

    data : (npt,) array_like or (nvar,) list of array_like
        一维变量数组. 如果是多个变量组成的列表, 那么分别对每个变量进行计算.

    bins_lon : (nlon + 1,) array_like
        用于划分经度的bins.

    bins_lat : (nlat + 1,) array_like
        用于划分纬度的bins.

    Returns
    -------
    glon : (nlon,) ndarray
        格点中心的经度.

    glat : (nlat,) ndarray
        格点中心的纬度.

    gridded : (nlat, nlon) or (nvar, nlat, nlon) ndarray
        每个格点的平均值. 为了便于画图, 纬度维在前.
    '''
    # 会因为格点内全为NaN而警告.
    gridded, bins_lat, bins_lon, _ = stats.binned_statistic_2d(
        lat, lon, data, bins=[bins_lat, bins_lon],
        statistic=np.nanmean
    )
    glon = (bins_lon[1:] + bins_lon[:-1]) / 2
    glat = (bins_lat[1:] + bins_lat[:-1]) / 2

    return glon, glat, gridded

if __name__ == '__main__':
    xbins = ybins = [0, 5, 10]
    x = y = [1, 2, 6, 7]
    data = [1, 2, np.nan, np.nan]
    xc, yc, avg = grid_data(x, y, [data]*2, xbins, ybins)
