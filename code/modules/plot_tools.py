import sys

import numpy as np
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

def add_Chinese_provinces(ax, **feature_kw):
    '''
    在地图上画出中国省界的shapefile.

    Parameters
    ----------
    ax : GeoAxes
        目标地图.

    **feature_kw
        调用add_feature时的关键字参数.
        例如linewidth, edgecolor和facecolor等.
    '''
    if sys.platform == 'win32':
        shp_filepath = 'D:/maps/shps/bou2_4p.shp'
    elif sys.platform == 'linux':
        shp_filepath = '/home/wangj/shps/bou2_4p.shp'
    reader = shpreader.Reader(shp_filepath)
    geometries = reader.geometries()
    provinces = cfeature.ShapelyFeature(geometries, ccrs.PlateCarree())
    ax.add_feature(provinces, **feature_kw)

def set_map_extent_and_ticks(
    ax, extents, xticks, yticks, nx=0, ny=0,
    xformatter=None, yformatter=None
):
    '''
    设置矩形投影的地图的经纬度范围和刻度.

    Parameters
    ----------
    ax : GeoAxes
        目标地图. 支持_RectangularProjection和Mercator投影.

    extents : 4-tuple of float or None
        经纬度范围[lonmin, lonmax, latmin, latmax]. 值为None表示全球.

    xticks : array_like
        经度主刻度的坐标.

    yticks : array_like
        纬度主刻度的坐标.

    nx : int, optional
        经度主刻度之间次刻度的个数. 默认没有次刻度.
        当经度不是等距分布时, 请不要进行设置.

    ny : int, optional
        纬度主刻度之间次刻度的个数. 默认没有次刻度.
        当纬度不是等距分布时, 请不要进行设置.

    xformatter : Formatter, optional
        经度主刻度的Formatter. 默认使用无参数的LongitudeFormatter.

    yformatter : Formatter, optional
        纬度主刻度的Formatter. 默认使用无参数的LatitudeFormatter.
    '''
    # 设置主刻度.
    proj = ccrs.PlateCarree()
    ax.set_xticks(xticks, crs=proj)
    ax.set_yticks(yticks, crs=proj)
    # 设置次刻度.
    xlocator = mticker.AutoMinorLocator(nx + 1)
    ylocator = mticker.AutoMinorLocator(ny + 1)
    ax.xaxis.set_minor_locator(xlocator)
    ax.yaxis.set_minor_locator(ylocator)

    # 设置Formatter.
    if xformatter is None:
        xformatter = LongitudeFormatter()
    if yformatter is None:
        yformatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(xformatter)
    ax.yaxis.set_major_formatter(yformatter)

    # 在最后调用set_extent, 防止刻度拓宽显示范围.
    if extents is None:
        ax.set_global()
    else:
        ax.set_extent(extents, crs=proj)

def add_box_on_map(ax, extents, **rect_kw):
    '''
    在地图上画出一个方框.

    Parameters
    ----------
    ax : GeoAxes
        目标地图. 最好为矩形投影, 否则效果可能很糟.

    extents : 4-tuple of float
        方框的经纬度范围[lonmin, lonmax, latmin, latmax].

    **rect_kw
        创建Rectangle时的关键字参数.
        例如linewidth, edgecolor和facecolor等.
    '''
    lonmin, lonmax, latmin, latmax = extents
    rect = mpatches.Rectangle(
        (lonmin, latmin), lonmax - lonmin, latmax - latmin,
        transform=ccrs.PlateCarree(), **rect_kw
    )
    ax.add_patch(rect)

def add_equal_axes(ax, loc, pad, width):
    '''
    在原有的Axes旁新添一个等高或等宽的Axes并返回该对象.

    Parameters
    ----------
    ax : Axes or array_like of Axes
        原有的Axes, 也可以为一组Axes构成的数组.

    loc : {'left', 'right', 'bottom', 'top'}
        新Axes相对于旧Axes的位置.

    pad : float
        新Axes与旧Axes的间距.

    width: float
        当loc='left'或'right'时, width表示新Axes的宽度.
        当loc='bottom'或'top'时, width表示新Axes的高度.

    Returns
    -------
    ax_new : Axes
        新Axes对象.
    '''
    # 无论ax是单个还是一组Axes, 获取ax的大小位置.
    axes = np.atleast_1d(ax).ravel()
    bbox = mtransforms.Bbox.union([ax.get_position() for ax in axes])

    # 决定新Axes的大小位置.
    if loc == 'left':
        x0_new = bbox.x0 - pad - width
        x1_new = x0_new + width
        y0_new = bbox.y0
        y1_new = bbox.y1
    elif loc == 'right':
        x0_new = bbox.x1 + pad
        x1_new = x0_new + width
        y0_new = bbox.y0
        y1_new = bbox.y1
    elif loc == 'bottom':
        x0_new = bbox.x0
        x1_new = bbox.x1
        y0_new = bbox.y0 - pad - width
        y1_new = y0_new + width
    elif loc == 'top':
        x0_new = bbox.x0
        x1_new = bbox.x1
        y0_new = bbox.y1 + pad
        y1_new = y0_new + width

    # 创建新Axes.
    fig = axes[0].get_figure()
    bbox_new = mtransforms.Bbox.from_extents(x0_new, y0_new, x1_new, y1_new)
    ax_new = fig.add_axes(bbox_new)

    return ax_new

def get_slice_xticks(
    lon, lat, ntick, decimals=2,
    lon_formatter=None, lat_formatter=None
):
    '''
    用经纬度的点数表示切片数据的横坐标, 在横坐标上取ntick个等距的刻度,
    利用线性插值计算每个刻度标签的经纬度值, 并返回这些量.

    Parameters
    ----------
    lon : (npt,) array_like
        切片数据的经度.

    lat : (npt,) array_like
        切片数据的纬度.

    ntick : int
        刻度的数量.

    decimals : int, optional
        刻度标签里数值的小数位数. 默认保留两位小数.

    lon_formatter : Formatter, optional
        刻度标签里经度的Formatter. 默认使用无参数的LongitudeFormatter.

    lat_formatter : Formatter, optional
        刻度标签里纬度的Formatter. 默认使用无参数的LatitudeFormatter.

    Returns
    -------
    x : (npt,) ndarray
        切片数据的横坐标. 数值等于np.arange(npt).

    xticks : (ntick,) ndarray
        刻度的位置.

    xticklabels : (ntick,) list of str
        刻度标签. 用刻度处的经纬度值表示.
    '''
    # 通过线性插值计算刻度的经纬度值.
    npt = len(lon)
    x = np.arange(npt)
    xticks = np.linspace(0, npt - 1, ntick)
    lon_ticks = np.interp(xticks, x, lon).round(decimals)
    lat_ticks = np.interp(xticks, x, lat).round(decimals)

    # 获取字符串形式的刻度标签.
    xticklabels = []
    if lon_formatter is None:
        lon_formatter = LongitudeFormatter()
    if lat_formatter is None:
        lat_formatter = LatitudeFormatter()
    for i in range(ntick):
        lon_label = lon_formatter(lon_ticks[i])
        lat_label = lat_formatter(lat_ticks[i])
        xticklabels.append(lon_label + '\n' + lat_label)

    return x, xticks, xticklabels

def make_qualitative_cmap(colors):
    '''
    创建一组定性的colormap和norm, 同时返回对应刻度的位置.

    Parameters
    ----------
    colors : (N,) list or (N, 3) or (N, 4) array_like
        colormap所含的颜色. 可以为含有颜色的列表或RGB(A)数组.

    Returns
    -------
    cmap : ListedColormap
        创建的colormap.

    norm : Normalize
        创建的norm. N个颜色对应于0~N-1范围的数据.

    ticks : (N,) ndarray
        colorbar刻度的坐标.
    '''
    N = len(colors)
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.Normalize(vmin=-0.5, vmax=N-0.5)
    ticks = np.arange(N)

    return cmap, norm, ticks

def get_rain_cmap():
    '''创建用于降水的colormap和norm.'''
    bounds = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0]
    nbin = len(bounds) - 1
    norm = mcolors.BoundaryNorm(bounds, nbin)
    cmap = cm.get_cmap('jet', nbin)
    cmap.set_under(color='lavender', alpha=0.2)

    return cmap, norm

def get_aod_cmap():
    '''返回适用于AOD的cmap.'''
    rgb = np.loadtxt(
        '/d4/wangj/dust_precipitation/code/modules/NEO_modis_aer_od.csv',
        delimiter=','
    ) / 256
    cmap = mcolors.ListedColormap(rgb)

    return cmap

def letter_axes(axes, x, y, **text_kw):
    '''
    给一组Axes按顺序标注字母.

    Parameters
    ----------
    axes : array_like of Axes
        目标Axes的数组.

    x : float or array_like
        字母的横坐标, 基于Axes单位.
        可以为标量或数组, 数组形状需与axes相同.

    y : float or array_like
        字母的纵坐标. 基于Axes单位.
        可以为标量或数组, 数组形状需与axes相同.

    y : float or array_like可以为标量或数组, 数组形状需与axes相同.

    **text_kw
        调用text时的关键字参数.
        例如fontsize, fontfamily和color等.
    '''
    axes, x, y = np.broadcast_arrays(np.atleast_1d(axes), x, y)
    for i, (ax, xi, yi) in enumerate(zip(axes.flat, x.flat, y.flat)):
        letter = chr(ord('`') + i + 1)
        ax.text(
            xi, yi, f'({letter})', ha='center', va='center',
            transform=ax.transAxes, **text_kw
        )

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    extents_map = [70, 140, 10, 60]
    extents_box = [110, 124, 32, 45]

    proj = ccrs.PlateCarree()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=proj)

    add_Chinese_provinces(ax, lw=0.3, ec='k', fc='none')
    set_map_extent_and_ticks(
        ax, extents=extents_map,
        xticks=np.arange(-180, 190, 20),
        yticks=np.arange(-90, 100, 20),
        nx=1, ny=1
    )
    ax.tick_params(labelsize='small')
    add_box_on_map(ax, extents_box, lw=1, ec='C3', fc='none')

    fig.savefig('test.png', dpi=300)
    plt.close(fig)
