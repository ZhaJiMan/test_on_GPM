import sys

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
    if sys.platform == 'win32':
        shp_filepath = 'D:/maps/shps/bou2_4p.shp'
    elif sys.platform == 'linux':
        shp_filepath = '/home/wangj/shps/bou2_4p.shp'
    reader = Reader(shp_filepath)
    provinces = cfeature.ShapelyFeature(reader.geometries(), proj)
    ax.add_feature(provinces, **kwargs)

    return None

def set_map_ticks(ax, dx=60, dy=30, nx=0, ny=0, labelsize='medium'):
    '''
    为PlateCarree投影的地图设置tick和tick label.
    需要注意,set_extent应该在该函数之后使用.

    Parameters
    ----------
    ax : GeoAxes
        需要被设置的地图,要求投影必须为PlateCarree.

    dx : float, default: 60
        经度的major ticks的间距,从-180度开始算起.默认值为10.

    dy : float, default: 30
        纬度的major ticks,从-90度开始算起,间距由dy指定.默认值为10.

    nx : float, default: 0
        经度的minor ticks的个数.默认值为0.

    ny : float, default: 0
        纬度的minor ticks的个数.默认值为0.

    labelsize : str or float, default: 'medium'
        tick label的大小.默认为'medium'.

    Returns
    -------
    None
    '''
    if not isinstance(ax.projection, ccrs.PlateCarree):
        raise ValueError('Projection of ax should be PlateCarree!')
    proj = ccrs.PlateCarree()   # 给ticks设置专用的crs.

    # 设置x轴.
    major_xticks = np.arange(-180, 180 + 0.9 * dx, dx)
    ax.set_xticks(major_xticks, crs=proj)
    if nx > 0:
        ddx = dx / (nx + 1)
        minor_xticks = np.arange(-180, 180 + 0.9 * ddx, ddx)
        ax.set_xticks(minor_xticks, minor=True, crs=proj)

    # 设置y轴.
    major_yticks = np.arange(-90, 90 + 0.9 * dy, dy)
    ax.set_yticks(major_yticks, crs=proj)
    if ny > 0:
        ddy = dy / (ny + 1)
        minor_yticks = np.arange(-90, 90 + 0.9 * ddy, ddy)
        ax.set_yticks(minor_yticks, minor=True, crs=proj)

    # 为tick label增添度数标识.
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=labelsize)

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
    proj = ccrs.PlateCarree()
    ax.plot(x, y, transform=proj, **plot_kw)

    return None

# 测试.
if __name__ == '__main__':
    proj = ccrs.PlateCarree()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=proj)
    ax.coastlines()
    set_map_ticks(ax)
    plt.show()
