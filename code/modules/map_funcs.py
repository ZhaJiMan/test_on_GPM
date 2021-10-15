#----------------------------------------------------------------------------
# 2021/10/15
# 绘制地图用的函数.
#----------------------------------------------------------------------------
import sys

import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

def add_Chinese_provinces(ax, **feature_kw):
    '''
    给一个GeoAxes添加上中国省界的shapefile.

    Parameters
    ----------
    ax : GeoAxes
        要添加shapefile的地图.

    **feature_kw
        绘制feature时的Matplotlib关键词参数,例如linewidth,facecolor,alpha等.
    '''
    if sys.platform == 'win32':
        shp_filepath = 'D:/maps/shps/bou2_4p.shp'
    elif sys.platform == 'linux':
        shp_filepath = '/home/wangj/shps/bou2_4p.shp'
    proj = ccrs.PlateCarree()
    reader = Reader(shp_filepath)
    geometries = reader.geometries()
    provinces = cfeature.ShapelyFeature(geometries, proj)
    ax.add_feature(provinces, **feature_kw)

def set_map_extent_and_ticks(
    ax, extent, xticks, yticks, nx=0, ny=0,
    xformatter=LongitudeFormatter(),
    yformatter=LatitudeFormatter()
):
    '''
    为矩形投影的地图设置extent和ticks.

    Parameters
    ----------
    ax : GeoAxes
        需要被设置的地图.支持_RectangularProjection和Mercator投影.

    extent : 4-tuple of float
        地图的经纬度范围[lonmin, lonmax, latmin, latmax].
        若值为None,则给出全球范围.

    xticks : list of float
        经度major ticks的位置.

    yticks : list of float
        纬度major ticks的位置.

    nx : int
        经度的两个major ticks之间minor ticks的个数.默认没有minor ticks.
        当经度不是等距分布时,请不要进行设置.

    ny : int
        纬度的两个major ticks之间minor ticks的个数.默认没有minor ticks.
        当纬度不是等距分布时,请不要进行设置.

    xformatter : LongitudeFormatter
        经度的major ticks的formatter.默认使用无参数的LongitudeFormatter.

    yformatter : LatitudeFormatter
        纬度的major ticks的formatter.默认使用无参数的LatitudeFormatter.
    '''
    # 设置ticks.
    proj = ccrs.PlateCarree()
    ax.set_xticks(xticks, crs=proj)
    ax.set_yticks(yticks, crs=proj)
    if nx > 0:
        xlocator = mpl.ticker.AutoMinorLocator(nx + 1)
        ax.xaxis.set_minor_locator(xlocator)
    if ny > 0:
        ylocator = mpl.ticker.AutoMinorLocator(ny + 1)
        ax.yaxis.set_minor_locator(ylocator)

    # 添加经纬度标识.
    ax.xaxis.set_major_formatter(xformatter)
    ax.yaxis.set_major_formatter(yformatter)

    # 最后设置extent,防止ticks超出extent的范围.
    if extent is None:
        ax.set_global()
    else:
        ax.set_extent(extent, crs=proj)

def add_box_on_map(ax, extent, **plot_kw):
    '''
    在矩形投影的GeoAxes上画出一个空心的方框.

    Parameters
    ----------
    ax : GeoAxes
        被绘制的GeoAxes.

    extent : 4-tuple of float
        方框的经纬度范围[lonmin, lonmax, latmin, latmax].

    **plot_kw
        利用plot方法画方框时的参数,例如linewidth,color等.
    '''
    lonmin, lonmax, latmin, latmax = extent
    x = [lonmin, lonmax, lonmax, lonmin, lonmin]
    y = [latmin, latmin, latmax, latmax, latmin]
    ax.plot(x, y, transform=ccrs.PlateCarree(), **plot_kw)
