import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

def add_Chinese_provinces(ax, **kwargs):
    '''
    给一个GeoAxes添加上中国省界的地图.

    Parameters
    ----------
    ax : GeoAxes
        要被画上地图的Axes,投影不限.

    **kwargs
        绘制feature时的Matplotlib关键词参数,例如linewidth,facecolor,alpha等.

    Returns
    -------
    None
    '''
    proj = ccrs.PlateCarree()
    shp_file_path = 'D:/maps/shps/bou2_4p.shp'
    reader = Reader(shp_file_path)
    provinces = cfeature.ShapelyFeature(reader.geometries(), proj)
    ax.add_feature(provinces, **kwargs)

    return None

def set_map_ticks(
        ax, extent=None,
        dx=10, dy=10, nx=0, ny=0,
        labelsize='medium'):
    '''
    设置一个PlateCarree投影的GeoAxes的显示范围与ticks.

    Parameters
    ----------
    ax : GeoAxes
        要被画上地图的Axes,投影必须是PlateCarree.

    extent : 4-tuple of float, default: None
        地图的经纬度范围[lonmin, lonmax, latmin, latmax].默认会画出全球范围.

    dx : float, default: 10
        经度的major ticks从-180度开始算起,间距由dx指定.默认值为10度.

    dy : float, default: 10
        纬度的major ticks从-90度开始算起,间距由dy指定.默认值为10度.

    nx : float, default: 0
        经度的相邻两个major ticks间的minor ticks的数目.默认没有minor ticks.

    ny : float, default: 0
        纬度的相邻两个major ticks间的minor ticks的数目.默认没有minor ticks.

    labelsize : str or float, default: 'x-small'
        tick label的大小.默认为'x-small'.

    Returns
    -------
    None
    '''
    proj = ccrs.PlateCarree()

    # 设置x轴.
    # 不直接使用np.arange(-180, 180 + dx, dx),防止舍入误差.
    major_xticks = np.arange(360 // dx + 1) * dx - 180.0
    ax.set_xticks(major_xticks, crs=proj)
    if nx != 0:
        ddx = dx / (nx + 1)
        minor_xticks = np.arange(360 // ddx + 1) * ddx - 180.0
        ax.set_xticks(minor_xticks, minor=True, crs=proj)

    # 设置y轴.
    major_yticks = np.arange(180 // dy + 1) * dy - 90
    ax.set_yticks(major_yticks, crs=proj)
    if ny != 0:
        ddy = dy / (ny + 1)
        minor_yticks = np.arange(180 // ddy + 1) * ddy - 90
        ax.set_yticks(minor_yticks, minor=True, crs=proj)

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params('both', labelsize=labelsize)

    # 最后限定范围,以免和ticks冲突
    if extent is None:
        ax.set_global()
    else:
        ax.set_extent(extent, crs=proj)

    return None

def draw_box_on_map(ax, extent, **plot_kw):
    '''
    在一个PlateCarree投影的GeoAxes上画出一个方框.

    Parameters
    ----------
    ax : GeoAxes
        被绘制的GeoAxes.要求投影必须为PlateCarree.

    extent : 4-tuple of float
        方框的经纬度范围[lonmin, lonmax, latmin, latmax].

    **plot_kw
        利用plot方法画方框时的参数.

    Returns
    -------
    None
    '''
    lonmin, lonmax, latmin, latmax = extent
    x = [lonmin, lonmax, lonmax, lonmin, lonmin]
    y = [latmin, latmin, latmax, latmax, latmin]
    data_proj = ccrs.PlateCarree()
    ax.plot(x, y, transform=data_proj, **plot_kw)

    return None

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
