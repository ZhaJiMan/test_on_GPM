import numpy as np
import xarray as xr
from scipy.stats import mstats

def calc_cfad(arr, y, xbins, ybins, norm=None):
    '''
    计算出廓线数据的CFAD分布.

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

    norm : str
        设置归一化的方式.默认不做归一化.
        若为'sum',使用计数的总和做归一化.
        若为'max',使用计数的最大值做归一化.

    Returns
    -------
    H : ndarray, shape (ny, nx)
        廓线数据经过分bin后得到的CFAD数值.
    '''
    npoint = arr.shape[0]
    H = np.histogram2d(
        arr.flatten(), np.tile(y, npoint), [xbins, ybins]
    )[0].T

    # 若H的值全为0,或不指定norm,那么不进行归一化操作.
    s = H.sum()
    if norm is None or np.isclose(s, 0):
        return H
    elif norm == 'sum':
        return H / s
    elif norm == 'max':
        return H / H.max()

def is_increasing(x):
    '''判断序列x是否单调递增.'''
    if len(x) >= 2 and np.all(np.diff(x) > 0):
        return True
    else:
        return False

def is_decreasing(x):
    '''判断序列x是否单调递减.'''
    if len(x) >= 2 and np.all(np.diff(x) < 0):
        return True
    else:
        return False

class Binner:
    '''一个简单的分bin实现.'''

    def __init__(self, x, y, bins, axis=0):
        '''
        在某个维度上对多维数组进行按bin分组.

        每个bin的范围仿照pd.cut,当bins单调递增时,为左开右闭区间.
        而当bins单调递减时,若再从小到大排序,依旧服从左开右闭区间.
        不能出现所有bin都空的情况,以保证reduce时有结果.

        Parameters
        ----------
        x : ndarray or MaskedArray, shape (nx,)
            被分bin的一维数组.可以含有NaN或缺测.

        y : ndarray or MaskedArray, shape(..., nx, ...)
            被分bin的多维数组,第axis维对应于x.可以含有NaN或缺测.

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
        if not (is_increasing(bins) or is_decreasing(bins)):
            raise ValueError('bins must increase monotonically')

        # 将x和y处理为MaskedArray.
        if not isinstance(x, np.ma.MaskedArray):
            x = np.ma.masked_invalid(x)
        if not isinstance(y, np.ma.MaskedArray):
            y = np.ma.masked_invalid(y)

        # 求出x的非缺测值落入哪个bin中,并统计每个bin中的点数.
        digits = np.digitize(x, bins, right=True)
        digits[x.mask] = 0
        counts = np.bincount(digits, minlength=nbin+1)[1:nbin+1]
        # 若每个bin都是空的,那么报错.
        if np.all(counts == 0):
            raise ValueError('No value of x falls into bins')

        # 设置分组标签.
        labels = []
        for i in range(nbin):
            labels.append(f'{bins[i]} ~ {bins[i + 1]}')

        # 根据落入bin中的情况截取y.
        data = []
        for i in range(nbin):
            indices = np.nonzero(digits == i + 1)[0]
            if indices.size == 0:
                data.append(None)
            else:
                # take方法能保留截取出的数组的维度.
                data.append(y.take(indices, axis))

        self.nbin = nbin
        self.counts = counts
        self.labels = labels
        self.data = data
        self.axis = axis

    def show(self):
        '''简单打印分组结果.'''
        for label, count in zip(self.labels, self.counts):
            print(label, ': ', count)

    def map(self, func, **kwargs):
        '''
        对每个bin中的数组应用函数func,再将结果汇总到列表中.

        Parameters
        ----------
        func : callable
            可以以func(arr, **kwargs)形式调用的函数.
            要求专门应用于MaskedArray.

        **kwargs : dict
            func的额外关键字.要求不再出现axis.

        Returns
        -------
        result : MaskedArray, shape (nbin, ...)
            汇总后的结果,数据类型与func返回的结果相同.第一维表示不同分组.
        '''
        f = lambda x : func(x, **kwargs)
        result = [f(a) if a is not None else None for a in self.data]

        return result


# 测试.
if __name__ == '__main__':
    filepath = '../../data/profile_files/dusty_profile.nc'
    ds = xr.load_dataset(filepath)
    x = ds.precipRateNearSurface.to_masked_array()
    y = ds.precipRate.to_masked_array()
    Rr1D = ds.precipRateNearSurface.to_masked_array()
    Rr2D = ds.precipRate.to_masked_array()
    sh = ds.heightStormTop.to_masked_array()

    bins = np.linspace(0, 10, 11)[::-1]
    b1 = Binner(Rr1D, sh, bins)
    avgs1 = b1.reduce(np.ma.mean)
    sems1 = b1.reduce(mstats.sem)

    b2 = Binner(Rr1D, Rr2D, bins, axis=0)
    b2.show()
    avgs2 = b2.reduce(np.ma.mean)
    sems2 = b2.reduce(mstats.sem)
    qs2 = b2.reduce(mstats.mquantiles, alphap=1, betap=1)
