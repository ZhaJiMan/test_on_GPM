#----------------------------------------------------------------------------
# 2021/10/15
# 处理降水廓线数据的函数.
#----------------------------------------------------------------------------
import numpy as np
import xarray as xr
from scipy.stats import mstats
from scipy.ndimage import gaussian_filter1d

def calc_cfad(arr, y, xbins, ybins, norm=None):
    '''
    计算出DPR廓线数据的CFAD分布.

    np.histogram2d的默认行为是区间左闭右开,但最后一个全闭.
    当y是等差数组且ybins与之匹配时可能会产生错误的效果.
    这里通过延长bins,使函数统一服从左闭右开的规则.

    Parameters
    ----------
    arr : ndarray
        二维的廓线数组,第一维是廓线数目,第二维是垂直方向.
        arr中可以含有NaN.

    y : ndarray
        与arr的垂直方向相对应的一维坐标数组.

    xbins : ndarray, shape (nx + 1,)
        用于划分arr数值的一维数组.要求单调递增.

    ybins : ndarray, shape (ny + 1,)
        用于划分垂直方向范围的一维数组.要求单调递增.
        注意ybins不能随便设置,需要与y相匹配.

    norm : str
        设置归一化的方式.默认不做归一化.
        若为'sum',使用计数的总和做归一化.
        若为'max',使用计数的最大值做归一化.

    Returns
    -------
    H : ndarray, shape (ny, nx)
        廓线数据经过分bin后得到的CFAD分布计数.
    '''
    # 提前给bins延长相等的一格.
    xbins = np.append(xbins, xbins[-1])
    ybins = np.append(ybins, ybins[-1])
    npoint = arr.shape[0]
    H = np.histogram2d(
        arr.ravel(), np.tile(y, npoint), [xbins, ybins]
    )[0].astype(int)    # 默认为浮点型.
    # 最后去掉延长的部分,并转置成适合画图的形状.
    H = H[:-1, :-1].T

    # 进行归一化操作.
    # 当H全为0时,后两种可能导致NaN和Inf的结果.
    if norm is None:
        return H
    elif norm == 'sum':
        return H / H.sum()
    elif norm == 'max':
        return H / H.max()
    else:
        raise ValueError('unsupported normalization')

    return H

def is_monotonic(x):
    '''判断序列x是否严格单调.'''
    if len(x) > 2:
        d = np.diff(x)
        if np.all(d > 0) or np.all(d < 0):
            return True
        else:
            return False
    else:
        return False

class Binner:
    '''一个简单的数组分bin实现.'''
    def __init__(self, x, y, bins, axis=0):
        '''
        在某个维度上对多维数组进行按bin分组.

        无论bins是单调递增还是单调递减,区间都服从低值边开,高值边闭的规则.
        允许每个bin都是空的.

        Parameters
        ----------
        x : ndarray or MaskedArray, shape (nx,)
            被分bin的一维数组.可以含有NaN或缺测.

        y : ndarray, shape(..., nx, ...)
            被分bin的多维数组,第axis维对应于x.数组是否含有缺测不影响分组.

        bins : array_like, len (nbin + 1,)
            bin的边缘值.要求严格单调.

        axis : int
            指定在哪一维上分bin.默认值为0.

        Attributes
        ----------
        nbin : int
            分组的个数.

        counts : ndarray, shape (nbin,)
            每个bin中有几个x的数据点.

        labels : list of str, len (nbin,)
            每个bin对应的标签.格式为'bins[i] ~ bins[i + 1]'.

        data : list, len (nbin,)
            存储分到每个bin中的数组的列表.空bin对应None.
        '''
        # 要求bins严格单调.
        nbin = len(bins) - 1
        if not is_monotonic(bins):
            raise ValueError('bins must be monotonic')

        # 将x处理为MaskedArray.
        if not isinstance(x, np.ma.MaskedArray):
            x = np.ma.masked_invalid(x)

        # 求出x的非缺测值落入哪个bin中,并统计每个bin中的点数.
        digits = np.digitize(x, bins, right=True)
        digits[x.mask] = 0
        counts = np.bincount(digits, minlength=nbin+1)[1:nbin+1]

        # 设置分组标签.
        labels = []
        for i in range(nbin):
            labels.append(f'{bins[i]} ~ {bins[i + 1]}')

        # 根据落入bin中的情况截取y.
        data = []
        for i in range(nbin):
            # 使用nonzero的结果进行索引,能维持切片后的数组的形状.
            indices = np.nonzero(digits == i + 1)[0]
            if indices.size == 0:
                data.append(None)
            else:
                data.append(y.take(indices, axis))

        self.nbin = nbin
        self.counts = counts
        self.labels = labels
        self.data = data

    def show(self):
        '''简单打印分组结果.'''
        for label, count in zip(self.labels, self.counts):
            print(label, ':', count)

    def apply(self, func, **kwargs):
        '''
        对每个bin中的数组应用函数func,再将结果汇总到列表中.

        Parameters
        ----------
        func : callable
            可以以func(a, **kwargs)形式调用的函数.

        **kwargs : dict
            func的额外关键字.

        Returns
        -------
        result : list, len (nbin,)
            汇总后的结果,空bin的结果为None.
        '''
        f = lambda x: func(x, **kwargs) if x is not None else None
        result = [f(a) for a in self.data]

        return result

def smooth_profiles(arr, **kwargs):
    '''
    高斯平滑廓线数组.

    会跳过缺测值,只平滑有效部分.若廓线全部缺测,则相当于不进行操作.

    Parameters
    ----------
    arr : ndarray or MaskedArray, shape (..., nbin)
        廓线数组.要求最后一维是高度(温度)维.
        缺测以NaN或maksed的形式出现.

    kwargs : any
        传给gaussian_filter1d函数的关键字.

    Returns
    -------
    arr_new : MaskedArray, shape (..., nbin)
        平滑后的廓线数组,与arr的形状相同.
    '''
    def smooth_func(x):
        '''平滑一条廓线中有效值的部分.'''
        y = x.copy()
        y[~y.mask] = gaussian_filter1d(y[~y.mask], **kwargs)

        return y

    # 转换为MaskedArray.
    arr_new = np.ma.masked_invalid(arr)
    arr_new = np.ma.apply_along_axis(smooth_func, -1, arr_new)

    return arr_new

# 测试.
if __name__ == '__main__':
    filepath = '../../data/composite_temp/profile_files/merged/dusty_profile.nc'
    ds = xr.load_dataset(filepath)
    x = ds.precipRateNearSurface.to_masked_array()
    y = ds.precipRate.to_masked_array()
    Rr1D = ds.precipRateNearSurface.to_masked_array()
    Rr2D = ds.precipRate.to_masked_array()
    sh = ds.heightStormTop.to_masked_array()

    bins = np.linspace(0, 10, 11)
    # 下面avgs和sems的结果都是列表.
    # 有需要的话自己根据元素的数据类型和形状来combine结果.
    b1 = Binner(Rr1D, sh, bins)
    avgs1 = b1.apply(np.ma.mean)
    sems1 = b1.apply(mstats.sem)
    b2 = Binner(Rr1D, Rr2D, bins, axis=0)
    avgs2 = b2.apply(np.ma.mean, axis=0)
    sems2 = b2.apply(mstats.sem, axis=0)
