#----------------------------------------------------------------------------
# 2021/07/08
# 将DPR廓线数据从高度坐标转换到温度坐标上的函数,和匹配GMI数据的函数.
#----------------------------------------------------------------------------
import numpy as np
from scipy.interpolate import interp1d, NearestNDInterpolator

def profile_converter(airTemperature, temp, bottom_inds, top_inds):
    '''
    利用最邻近插值将廓线数据从高度坐标转换到温度坐标上.

    仅选取一定高度范围内的廓线数据,以尽量避免气温非单调带来的误差.
    使用最邻近插值能够保留廓线数据中的缺测信息.
    因为转换要进行很多次,所以用闭包简化函数.

    Parameters
    ----------
    airTemperature : (npoint, nheight) ndarray
        ENV文件中的气温.

    temp : (ntemp,) ndarray
        目标温度坐标.要求温度随序号增大而增大.

    bottom_inds : (npoint,) ndarray
        给定的高度范围下界的下标.

    top_inds : (npoint,) ndarray
        给定的高度范围上界的下标.
        要求top_inds的每个元素小于对应的bottom_inds的元素.

    Returns
    -------
    converter : function
        进行转换的函数.
    '''
    def converter(var, fill_value):
        '''
        对廓线数据进行转换.

        Parameters
        ----------
        var : (npoint, nheight) ndarray
            需要被转换的廓线数组.第一维表示廓线数目,第二维为高度维.
            要求第二维遵循DPR原始数据,高度随序号增大而减小.

        fill_value: float or 2-tuple of float
            当temp超出给定高度范围内airTemperature廓线的值域时所使用的缺测值.
            若为单个浮点数,那么超出的范围都由这个数进行填充.
            若为含两个浮点数的元组,那么temp<t_min的部分(即高空)由第一个数填充,
            而temp>t_max的部分(即低空)由第二个数进行填充.
            fill_value的选取要根据var的缺测种类来决定.

        Returns
        -------
        var_new : (npoint, ntemp) ndarray
            转换后的var数组.第二维与temp相匹配.
        '''
        npoint, nheight = var.shape
        ntemp = len(temp)
        var_new = np.zeros((npoint, ntemp), dtype=var.dtype)

        # 对bottom_ind到top_ind之间的数据进行最邻近插值.
        for i in range(npoint):
            slicer = slice(top_inds[i], bottom_inds[i] + 1)
            f = interp1d(
                airTemperature[i, slicer], var[i, slicer],
                kind='nearest', bounds_error=False, fill_value=fill_value
            )
            var_new[i, :] = f(temp)

        return var_new

    return converter

def convert_height(var, height, airTemperature, fill_value):
    '''
    利用线性插值将高度相关的变量var插值为温度变量.

    Parameters
    ----------
    var : (npoint,) ndarray
        高度相关变量的一维数组,例如雨顶高度和零度层高度.

    height : (nheight,) ndarray
        高度的一维数组.要求单调递减,且单位与var一致.

    airTemperature : (npoint, nheight) ndarray
        ENV文件中气温的二维数组.要求第二维与height相匹配.

    fill_value : float
        var的缺测值,同时也会设为var_new的缺测值.

    Returns
    -------
    var_new : (npoint,) ndarray
        转换后的变量.与var的大小和类型相同.
    '''
    npoint, nheight = airTemperature.shape
    var_new = np.full_like(var, fill_value)
    for i in range(npoint):
        if np.isclose(var[i], fill_value):
            continue
        else:
            # np.interp函数要求xp单调递增.
            var_new[i] = np.interp(
                var[i], height[::-1], airTemperature[i, ::-1],
                left=fill_value, right=fill_value
            )

    return var_new

def match_to_DPR(points_DPR, points_GMI, var_GMI):
    '''
    利用最邻近插值将GMI的变量匹配到DPR的数据点上.

    Parameters
    ----------
    points_DPR : (npoint, 2) ndarray
        想要匹配到的DPR的数据点的位置.两列分别表示经度和纬度.

    points_GMI : (npoint_GMI, 2) ndarray
        GMI变量的数据点的位置.两列分别表示经度和纬度.

    var_GMI : (npoint_GMI) ndarray
        GMI变量,维度与points_GMI匹配.

    Returns
    -------
    var_DPR : (npoint) ndarray
        在DPR数据点位置上的变量值.
    '''
    npoint = points_DPR.shape[0]
    f = NearestNDInterpolator(points_GMI, var_GMI)
    var_DPR = f(points_DPR)

    return var_DPR
