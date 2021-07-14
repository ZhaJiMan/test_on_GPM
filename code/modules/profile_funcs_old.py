import numpy as np
import xarray as xr
from scipy.stats import ttest_ind

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
    '''判断序列是否单调递增.'''
    if len(x) >= 2 and np.all(np.diff(x) >= 0):
        return True
    else:
        return False

def is_decreasing(x):
    '''判断序列是否单调递减.'''
    if len(x) >= 2 and np.all(np.diff(x) <= 0):
        return True
    else:
        return False

class Binner:
    '''
    将多维数组按bin分组,并进行简单统计处理.

    每个bin的范围仿照pd.cut,为左开右闭区间.

    Parameters
    ----------
    x : ndarray or MaskedArray, shape (nx,)
        被分bin的一维数组.可以含有NaN.

    y : ndarray or MaskedArray, shape(..., nx, ...)
        被分bin的多维数组,第axis维对应于x.可以含有NaN.

    bins : array_like, len (nbin + 1,)
        bin的边缘值.要求单调递增或递减.

    axis : int
        指定在哪一维上分bin.默认值为0.

    Attributes
    ----------
    nbin : int
        数据被分成了几个bin.

    counts : ndarray, shape (nbin,)
        每个bin中有几个x的数据点.

    data : list of MaskedArray, len (nbin,)
        存储分到每个bin中的数组的列表.
        若某个bin中没有数据落入,那么data对应的数组为空数组.
    '''
    def __init__(self, x, y, bins, axis=0):
        # 要求bins必须单调.
        nbin = len(bins) - 1
        if not(is_increasing(bins) or is_decreasing(bins)):
            raise ValueError('bins must be monotonic')

        # 预先将x和y处理为MaskedArray.
        if not isinstance(x, np.ma.MaskedArray):
            x = np.ma.masked_invalid(x)
        if not isinstance(y, np.ma.MaskedArray):
            y = np.ma.masked_invalid(y)

        # 求出x的非缺测值落入哪个bin中,并统计每个bin中的点数.
        digits = np.digitize(x, bins, right=True)
        digits[x.mask] = 0
        counts = np.bincount(digits, minlength=nbin+1)[1:]
        # 若每个bin中都没有数据,那么报错.
        if np.all(counts == 0):
            raise ValueError('None of the data falls within bins')

        # 对y进行分bin.
        data = []
        for i in range(nbin):
            # 用indices截取的数组能保留维度.可能出现空数组.
            indices = np.nonzero(digits == i + 1)[0]
            data.append(y.take(indices, axis))

        # 之后用于统计的数组形状.
        reduced_shape = list(y.shape)
        reduced_shape.pop(axis)
        reduced_shape.insert(0, nbin)

        self.nbin = nbin
        self.axis = axis
        self.data = data
        self.counts = counts
        self.reduced_shape = reduced_shape

    def reduce(self, func, **kwargs):
        '''
        对每个bin中的数组在axis维度上应用函数func,将结果聚合为数组.

        Parameters
        ----------
        func : callable
            可以以func(arr, axis=self.axis, **kwargs)形式调用的函数.
            能压缩每个bin中的数组在self.axis方向上的维度.

        **kwargs : dict
            func的额外参数.要求不能有axis.

        Returns
        -------
        reduced : MaskedArray, shape (nbin, ...)
            每个bin中的数组在reduce后聚合而成的数组.
            空bin对应的部分会设为masked.
        '''
        result = np.ma.masked_all(self.reduced_shape)
        for i, arr in enumerate(self.data):
            if arr.size == 0:
                continue
            else:
                result[i, ...] = func(arr, axis=self.axis, **kwargs)

        return result

    def mean(self):
        '''对每个bin中的数组在axis维度上求平均.'''
        return self.reduce(np.ma.mean)

    def sem(self):
        '''对每个bin中的数组在axis维度上求标准误.'''
        # ddof=1,且用np.ma.sqrt避免样本数小于1的情况.
        func = lambda x: x.std(self.axis) / np.ma.sqrt(x.count(self.axis) - 1)
        return self.reduce(func)

    def sem_old(self):
        '''对每个bin中的数据在axis维度上求标准误,并聚合为MaskedArray.'''
        result = np.ma.masked_all(self.reduced_shape)
        for i in range(self.nbin):
            # 若第i个bin中是空数组,那么跳过.
            arr = self.data[i]
            if arr.size == 0:
                continue
            else:
                # ddof=1时常有除零的问题,所以这里设ddof=0.
                sigma = arr.std(self.axis)
                n = arr.count(self.axis)
                result[i, ...] = sigma / np.sqrt(n)

        return result

def binned_tval(x_list, y_list):
    '''
    对两组已经分好bin的数据计算t value.

    要求x_list与y_list每个bin中含有的数据是一维数组.可以是含NaN,
    也可以是MaskedArray.

    若某个bin中两组数据不满足做t检验的要求,那么对应的t value缺测.
    返回的是t value的绝对值.
    '''
    def can_be_test(x, y):
        '''
        判断两个一维数组x和y是否可以用来做t检验.

        要求x与y都是不含NaN的ndarray.
        判断标准为x与y的长度都至少大于2,且二者的方差不全为0.
        '''
        nx = x.size
        ny = y.size
        if not (nx >= 2 and ny >= 2):
            return False

        xvar = x.var()
        yvar = y.var()
        if xvar > 0 or yvar > 0:
            return True
        else:
            return False

    def get_valid(x):
        '''获取一维数组x中的有效值.'''
        if isinstance(x, np.ma.MaskedArray):
            return x.compressed()
        else:
            return x[~np.isnan(x)]

    nbin = len(x_list)
    tvals = np.ma.masked_all(nbin)
    for i in range(nbin):
        x = get_valid(x_list[i])
        y = get_valid(y_list[i])
        if can_be_test(x, y):
            tvals[i] = np.abs(ttest_ind(x, y)[0])
        else:
            continue

    return tvals

# 测试用的数据.
if __name__ == '__main__':
    filepath = '../../data/profile_files/dusty_profile.nc'
    ds = xr.load_dataset(filepath)
    ds = ds.isel(npoint=(ds.rainType == 1))
    Rr = ds.precipRate.data
    surfRr = ds.precipRateNearSurface.data
    sh = ds.tempStormTop.data

    bins = np.linspace(0.1, 30, 10)
    b1 = Binner(surfRr, sh, bins)
    b2 = Binner(surfRr, Rr, bins)

    Rr1D = ds.precipRateNearSurface
    Rr2D = ds.precipRate
    bins = np.linspace(200, 300, 11)
