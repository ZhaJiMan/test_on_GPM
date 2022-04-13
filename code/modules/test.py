import warnings

import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt

def nansem(a, axis=None, ddof=0):
    '''计算忽略NaN的标准误.'''
    valid = ~np.isnan(a)
    n = np.count_nonzero(valid, axis)
    n = np.where(n > 0, n, np.nan)
    var = np.nanvar(a, axis=axis, ddof=ddof)
    sem = np.sqrt(var / n)

    return sem

class ProfileBinner:
    '''对廓线进行分组计算的类.'''
    def __init__(self, x, data, bins):
        '''
        根据x的值划分每组含有的data廓线.

        分组区间服从低值边开, 高值边闭的规则.
        不允许所有分组都是空的.

        Parameters
        ----------
        x : ndarray
            分组时所依据的数组. 缺测用NaN表示.

        data : (..., nz) ndarray
            被分组的廓线数组. 要求最后一维是高度维, 前面的维度与x相匹配.
            缺测用NaN表示.

        bins : (nbin,) array_like
            划分每个bin的边缘值. 要求单调递增或递减.

        Attributes
        ----------
        nbin : int
            组数.

        counts : (nbin,) ndarray
            每组含有的x的数据点数.

        labels : (nbin,) list of str
            每组对应的标签. 格式为'bins[i] ~ bins[i + 1]'.

        groups : (nbin,) list of ndarray
            存储每组对应的廓线数组的列表. 空bin用None填充.
        '''
        # 检查x和data的形状.
        if x.shape != data.shape[:-1]:
            raise ValueError('x的形状与data不匹配')

        # 得出x的每个元素在第几个bin中, 并统计每个bin中的点数.
        nbin = len(bins) - 1
        digits = np.digitize(x, bins, right=True)
        counts = np.bincount(digits, minlength=nbin+2)[1:nbin+1]
        if np.all(counts == 0):
            raise ValueError('x没有元素落入bins中')

        # 设置每组的标签.
        labels = []
        for i in range(nbin):
            label = f'{bins[i]} ~ {bins[i + 1]}'
            labels.append(label)

        # 索引每组含有的data. 空bin用None填充.
        groups = []
        for i in range(nbin):
            mask = digits == i + 1
            group = data[mask, :] if mask.any() else None
            groups.append(group)

        self.nbin = nbin
        self.counts = counts
        self.labels = labels
        self.groups = groups

    def show(self):
        '''简单打印分组结果.'''
        for label, count in zip(self.labels, self.counts):
            print(label, ':', count)

    def apply(self, func, **kwargs):
        '''
        对每组含有的廓线数组进行数值计算, 并将结果合并为单个数组.

        Parameters
        ----------
        func : callable
            以func(arr, **kwargs)形式调用的函数.
            要求其对不同形状的输入有相同形状的输出.

        **kwargs : dict
            func的关键字.

        Returns
        -------
        applied : (nbin, ...) ndarray
            合并后的结果. 缺测用NaN表示.
        '''
        applied = []
        for group in self.groups:
            arr = None if group is None else func(group, **kwargs)
            applied.append(arr)

        return self.combine(applied)

    def combine(self, applied):
        '''将apply方法的结果合并为单个数组.'''
        # 获取第一个非None的元素, 根据它的形状决定combined的形状.
        first = next(arr for arr in applied if arr is not None)
        if np.isscalar(first):
            shape = (self.nbin,)
        else:
            shape = (self.nbin,) + first.shape

        # 将applied的元素填入combined中.
        combined = np.full(shape, np.nan)
        for i, arr in enumerate(applied):
            if arr is not None:
                combined[i] = arr

        return combined

    def mean(self):
        '''求每组的平均廓线.'''
        return self.apply(np.nanmean, axis=0)

    def std(self):
        '''求每组的标准差廓线.'''
        return self.apply(np.nanstd, axis=0)

    def sem(self):
        '''求每组的标准误廓线.'''
        return self.apply(nansem, axis=0)

    def quantile(self, q):
        '''求每组的分位数廓线.'''
        return self.apply(np.nanquantile, axis=0, q=q)

with np.load('000150.npz') as f:
    temp = f['temp']
    precipRateNearSurface = f['precipRateNearSurface']
    precipRate_t = f['precipRate_t']

x = precipRateNearSurface[[0]]
data = precipRate_t[[0], :]
# bins = [0, 1, 2, 5, 10]
bins = [0, 0.5, 1]
binner = ProfileBinner(x, data, bins)
mean = binner.mean()
std = binner.std()
sem = binner.sem()
quantile = binner.quantile(0.5)

plt.figure(figsize=(3, 6))
plt.plot(mean[0, :], temp)
plt.plot(mean[0, :] - std[1, :], temp)
plt.plot(mean[0, :] + std[1, :], temp)
plt.xlim(0, None)
plt.ylim(20, -60)
plt.show()