#----------------------------------------------------------------------------
# 2021/07/08
# 用于截取区域内数据的函数.
#----------------------------------------------------------------------------
import numpy as np

def region_mask(lon, lat, extent):
    '''
    获取落入给定区域方框内的经纬度数组对应的Boolean数组.

    函数内只是应用简单的比较运算.若经度有跳变,需要提前对数据进行预处理.

    Parameters
    ----------
    lon2D : ndarray
        一维或二维的经度数组.要求形状与纬度数组相匹配.

    lat2D : ndarray
        一维或二维的纬度数组.要求形状与经度数组相匹配.

    extent : 4-tuple of float
        区域方框的经纬度范围[lonmin, lonmax, latmin, latmax].

    Returns
    -------
    mask : ndarray
        落入区域方框内的经纬度数组对应的Boolean数组.
    '''
    lonmin, lonmax, latmin, latmax = extent
    mask = (
        (lon >= lonmin) & (lon <= lonmax) &
        (lat >= latmin) & (lat <= latmax)
    )

    return mask

# 测试.
if __name__ == '__main__':
    extent = [100, 150, 0, 60]
    lon, lat = 120, 70
    mask = region_mask(lon, lat, extent)
