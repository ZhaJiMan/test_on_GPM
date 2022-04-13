import numpy as np
from scipy import stats, interpolate, ndimage

class ProfileConverter:
    '''将高度坐标上的DPR降水数据转换到温度坐标上的类.'''
    def __init__(self, airTemperature, height, hmax=12):
        '''
        创建一个转换器.

        会选取hmax高度以下的气温廓线进行线性拟合, 从而得到随高度单调递减的
        气温廓线. hmax建议根据研究区域的对流层高度进行选择.

        Parameters
        ----------
        airTemperature : (..., nbin) ndarray
            ENV文件的气温廓线数据. 要求最后一维是高度维.

        height : (nbin,) ndarray
            DPR和ENV文件高度维的坐标. 要求单调递减.

        hmax : float, optional
            参与线性拟合的气温数据的最大高度. 默认为12km.
        '''
        # 拟合出单调递增的airTemperature.
        airTemperature_fitted = np.zeros_like(airTemperature)
        for index in np.ndindex(airTemperature.shape[:-1]):
            profile = airTemperature[index]
            mask = height <= hmax
            result = stats.linregress(height[mask], profile[mask])
            profile_fitted = result.slope * height + result.intercept
            airTemperature_fitted[index] = profile_fitted

        self.height = height
        self.airTemperature = airTemperature
        self.airTemperature_fitted = airTemperature_fitted

    def convert3d(self, data, temp):
        '''
        将廓线数据从高度坐标转换到温度坐标上.

        转换通过线性插值实现, 数据缺测和外插部分均用NaN表示.

        Parameters
        ----------
        data : (..., nbin) ndarray
            廓线数组. 要求形状与airTemperature相同.
            可选变量例如precipRate, zFactorCorrected, Nw, Dm等.

        temp : (nt,) ndarray
            给出的温度坐标. 要求单位与airTemperature相同,
            数值范围最好与hmax相配合.

        Returns
        -------
        data_interpolated : (..., nt) ndarray
            温度坐标上的data.
        '''
        if data.shape != self.airTemperature.shape:
            raise ValueError('data的形状与airTemperature不同.')
        # 线性插值出温度坐标上的data.
        shape_new = data.shape[:-1] + temp.shape
        data_interpolated = np.zeros(shape_new, data.dtype)
        for index in np.ndindex(shape_new[:-1]):
            profile_data = data[index]
            profile_temp = self.airTemperature_fitted[index]
            data_interpolated[index] = np.interp(
                temp, profile_temp, profile_data,
                left=np.nan, right=np.nan
            )

        return data_interpolated

    def convert2d(self, data):
        '''
        将高度变量转换为温度变量.

        转换通过线性插值实现, 数据缺测和外插部分均用NaN表示.

        Parameters
        ----------
        data : ndarray
            高度变量数组. 要求形状与airTemperature前几维匹配,
            单位与height相同. 可选变量例如heightStormTop, heightZeroDeg等.

        Returns
        -------
        data_interpolated : ndarray
            data高度对应的温度值.
        '''
        if data.shape != self.airTemperature.shape[:-1]:
            raise ValueError('data的形状与airTemperature不匹配.')
        # 线性插值出data对应的温度值.
        data_interpolated = np.zeros_like(data)
        for index, x in np.ndenumerate(data):
            profile = self.airTemperature[index]
            # np.interp函数要求xp单调递增.
            data_interpolated[index] = np.interp(
                x, self.height[::-1], profile[::-1],
                left=np.nan, right=np.nan
            )

        return data_interpolated

def match_GMI_to_DPR(lon_GMI, lat_GMI, lon_DPR, lat_DPR, data_GMI):
    '''利用最邻近插值将GMI数据匹配到DPR的像元上.'''
    if not (lon_GMI.shape == lat_GMI.shape == data_GMI.shape):
        raise ValueError('GMI数据的形状不匹配')
    if lon_DPR.shape != lat_DPR.shape:
        raise ValueError('DPR数据的形状不匹配')
    points_GMI = np.column_stack([lon_GMI.ravel(), lat_GMI.ravel()])
    points_DPR = np.column_stack([lon_DPR.ravel(), lat_DPR.ravel()])
    f = interpolate.NearestNDInterpolator(points_GMI, data_GMI)
    data_DPR = f(points_DPR).reshape(lon_DPR.shape)

    return data_DPR

def nansem(a, axis=0, ddof=1):
    '''计算忽略NaN的标准误.'''
    return stats.mstats.sem(
        np.ma.masked_invalid(a), axis, ddof
    ).filled(np.nan)

class ProfileBinner:
    '''对DPR廓线数据进行分组计算的类.'''
    def __init__(self, x, data, bins):
        '''
        根据x的值将data划分到每一组中.

        分组区间服从低值边开, 高值边闭的规则. 不允许分组全空.
        数据缺测用NaN表示.

        Parameters
        ----------
        x : ndarray
            分组时所依据的数组.
            可选变量例如precipRateNearSurface, heightStormTop等.

        data : (..., nh) ndarray or (nvar,) list of ndarray
            被分组的廓线数组. 要求前几维与x匹配, 最后一维是垂直坐标维.
            可选变量例如precipRate, zFactorCorrected等.
            若是形状相同的廓线数组组成的列表, 那么分别对每个数组进行计算.

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

        groups : (nbin,) or (nvar, nbin) ndarray
            存储每组对应的廓线数组的数组(类型为object). 空bin用None填充.
        '''
        if not isinstance(data, list):
            data = [data]
        data = np.asarray(data)
        if x.shape != data.shape[1:-1]:
            raise ValueError('x的形状与data不匹配')

        # 得出x的每个元素在第几个bin中, 并统计每个bin中的点数.
        nbin = len(bins) - 1
        digits = np.digitize(x, bins, right=True)
        counts = np.bincount(digits.ravel(), minlength=nbin+2)[1:nbin+1]
        if np.all(counts == 0):
            raise ValueError('x没有元素落入bins中')

        # 设置每组的标签.
        labels = []
        for i in range(nbin):
            label = f'{bins[i]} ~ {bins[i + 1]}'
            labels.append(label)

        # 索引每组含有的data. 空bin用None填充.
        nvar = data.shape[0]
        groups = np.full((nvar, nbin), None)
        for j in range(nbin):
            mask = digits == j + 1
            if mask.any():
                for i in range(nvar):
                    groups[i, j] = data[i, mask, :]
        groups = np.squeeze(groups)

        self.counts = counts
        self.labels = labels
        self.groups = groups

    def __repr__(self):
        '''简单显示分组结果.'''
        lines = ['Bin | Number']
        for label, count in zip(self.labels, self.counts):
            line = f'{label} : {count}'
            lines.append(line)

        return '\n'.join(lines)

    def apply(self, func, **kwargs):
        '''
        将函数应用于每个分组, 并将计算结果合并成单个数组.

        Parameters
        ----------
        func : callable
            以func(arr, **kwargs)形式调用的函数.
            要求返回数值结果, 且对不同形状的输入有相同形状的输出.

        **kwargs : dict
            func的关键字.

        Returns
        -------
        applied : (nbin, ...) or (nvar, nbin, ...) ndarray
            合并后的结果.
        '''
        applied = np.full(self.groups.shape, None)
        for index, group in np.ndenumerate(self.groups):
            if group is not None:
                applied[index] = func(group, **kwargs)

        return self.combine(applied)

    @staticmethod
    def combine(applied):
        '''将apply方法的结果合并为单个数组.'''
        # 获取第一个非None的元素, 根据它的形状决定combined的形状.
        first = next(arr for arr in applied.flat if arr is not None)
        shape = applied.shape + np.asarray(first).shape

        # 将applied的元素填入combined中.
        combined = np.full(shape, np.nan)
        for index, arr in np.ndenumerate(applied):
            if arr is not None:
                combined[index] = arr

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
        '''求每组的分位数廓线. q可以为数组.'''
        return self.apply(np.nanquantile, axis=0, q=q)

def normalize_hist(h, norm=None):
    '''
    对histogram的结果进行归一化.

    Parameters
    ----------
    h : ndarray
        np.histogram等函数的结果. 要求数值非负.

    norm : str
        归一化的方式. 默认不进行归一化, 直接返回h.
        若为'sum', h除以其总和.
        若为'max', h除以其最大值.

    Returns
    -------
    normalized : ndarray
        归一化后的h. 其数据类型与norm有关.
    '''
    if np.any(h < 0):
        raise ValueError('数组必须非负')

    # 当h全0时可能产生NaN或Inf.
    if norm is None:
        return h
    elif norm == 'sum':
        return h / h.sum()
    elif norm == 'max':
        return h / h.max()

def hist1d(x, bins, sigma=None, norm=None):
    '''
    计算一维histogram.

    Parameters
    ----------
    x : array_like
        输入的数据, 会被展平成一维数组. 可含NaN.

    bins : (nbin + 1,) array_like
        划分x数值的边缘值. 要求单调递增.

    sigma : float
        进行高斯平滑的参数. 值越大结果越平滑, 默认不进行平滑.

    norm : str
        进行归一化的参数. 见normalize_hist函数的说明.

    Returns
    -------
    h : (nbin,) ndarray
        一维histogram.
    '''
    h = np.histogram(x, bins)[0]
    if sigma is not None:
        h = ndimage.gaussian_filter1d(h, sigma)
    h = normalize_hist(h, norm)

    return h

def hist2d(x, y, xbins, ybins, sigma=None, norm=None):
    '''
    计算二维histogram.

    Parameters
    ----------
    x : (npt,) array_like
        一维横坐标数据. 要求长度与y相同, 可含NaN.

    y : (npt,) array_like
        一维纵坐标数据. 要求长度与x相同, 可含NaN.

    xbins : (nx + 1), array_like
        划分x数值的边缘值. 要求单调递增.

    ybins : (ny + 1), array_like
        划分y数值的边缘值. 要求单调递增.

    sigma : float
        进行高斯平滑的参数. 值越大结果越平滑, 默认不进行平滑.

    norm : str
        进行归一化的参数. 见normalize_hist函数的说明.

    Returns
    -------
    H : (ny, nx) ndarray
        二维histogram.
    '''
    H = np.histogram2d(x, y, [xbins, ybins])[0]
    H = H.astype(int).T  # np.histogram2d的结果为浮点型.
    if sigma is not None:
        H = ndimage.gaussian_filter(H, sigma)
    H = normalize_hist(H, norm)

    return H

def cfad(data, y, xbins, ybins, sigma=None, norm=None):
    '''
    计算DPR廓线数据的CFAD分布.

    Parameters
    ----------
    data : (..., nh) ndarray
        廓线数组. 要求最后一维是垂直坐标维, 可含NaN.

    y : (nh,) ndarray
        data对应的垂直坐标.

    xbins : (nx + 1,) array_like
        划分data数值的边缘值. 要求单调递增.

    ybins : (ny + 1,) array_like
        划分垂直坐标的边缘值. 要求单调递增.
        注意ybins不能随便设置, 需要与y相配合.

    sigma : float
        进行高斯平滑的参数. 值越大结果越平滑, 默认不进行平滑.

    norm : str
        进行归一化的参数. 见normalize_hist函数的说明.

    Returns
    -------
    H : (ny, nx) ndarray
        CFAD分布.
    '''
    # 通过延长xbins和ybins, 使np.histogram2d的区间严格满足左闭右开的规则.
    # 避免对于某些y和ybins, 最后一个bin内的数据点异常增多.
    xbins = np.append(xbins, xbins[-1])
    ybins = np.append(ybins, ybins[-1])
    npt = np.prod(data.shape[:-1])
    H = np.histogram2d(data.ravel(), np.tile(y, npt), [xbins, ybins])[0]
    # 去掉延长的部分, 并转置成适合画图的形状.
    H = H.astype(int)[:-1, :-1].T

    if sigma is not None:
        H = ndimage.gaussian_filter(H, sigma)
    H = normalize_hist(H, norm)

    return H

def smooth_profiles(data, sigma):
    '''
    高斯平滑廓线数组.

    会跳过缺测值只平滑有效部分.
    若廓线全部缺测, 则相当于不进行操作.

    Parameters
    ----------
    data : (..., nh) ndarray
        廓线数组. 要求最后一维是垂直坐标维度, 缺测用NaN表示.

    sigma : float
        进行高斯平滑的参数. 值越大结果越平滑, 默认不进行平滑.

    Returns
    -------
    smoothed : (..., nh) ndarray
        平滑后的data.
    '''
    def f(profile):
        smoothed = profile.copy()
        mask = ~np.isnan(profile)
        smoothed[mask] = ndimage.gaussian_filter1d(
            profile[mask], sigma=sigma
        )

        return smoothed

    return np.apply_along_axis(f, axis=-1, arr=data)

def ttest_profiles(data1, data2, alpha):
    '''
    对两组廓线进行平均廓线是否相等的双尾t检验.

    数据缺测均用NaN表示.

    Parameters
    ----------
    data1 : (npt1, nh) ndarray
        廓线数组. 要求最后一维是垂直坐标维度.

    data2 : (npt2, nh) ndarray
        廓线数组. 要求最后一维是垂直坐标维度.

    alpha : float
        显著性水平. 例如0.05或0.01.

    Returns
    -------
    mask : (nh,) ndarray
        掩膜廓线. 真值部分表示通过了检验.
    '''
    if data1.ndim != 2 or data2.ndim != 2:
        raise ValueError('data必须是二维数组')
    pvalue = stats.ttest_ind(
        data1, data2, axis=0,
        equal_var=False, nan_policy='omit'
    ).pvalue.filled(np.nan)
    mask = pvalue < alpha

    return mask

if __name__ == '__main__':
    from pathlib import Path

    import h5py
    import matplotlib.pyplot as plt

    import data_tools
    import region_tools

    filepath_DPR = Path('/data00/0/GPM/DPR/V06/2017/201705/2A.GPM.DPR.V8-20180723.20170503-S070958-E084232.018057.V06A.HDF5')
    filepath_ENV = data_tools.to_ENV_filepath(filepath_DPR)

    extents = [108, 118, 30, 40]
    with data_tools.ReaderDPR(str(filepath_DPR)) as f:
        # 选取区域内近地表降水率大于0的像元.
        Longitude, Latitude = f.read_lonlat()
        precipRateNearSurface = f.read_ds('SLV/precipRateNearSurface')
        mask_extent = region_tools.region_mask(Longitude, Latitude, extents)
        mask_rain = precipRateNearSurface > 0
        mask_all = mask_extent & mask_rain

        precipRateNearSurface = precipRateNearSurface[mask_all]
        # 读取高度量.
        heightStormTop = f.read_ds('PRE/heightStormTop')[mask_all]
        heightZeroDeg = f.read_ds('VER/heightZeroDeg')[mask_all]
        # 读取降水量.
        precipRate = f.read_ds('SLV/precipRate')[mask_all, :]
        zFactorCorrected = f.read_ds('SLV/zFactorCorrected')[mask_all, :]

    # 读取ENV数据.
    with h5py.File(str(filepath_ENV), 'r') as f:
        airTemperature = f['NS/VERENV/airTemperature'][:][mask_all, :]

    # 转换单位.
    heightStormTop /= 1000
    heightZeroDeg /= 1000
    airTemperature -= 273.15

    # 设置DPR使用的高度, 注意数值从大到小.
    nbin = 176
    dh = 0.125  # 单位为km.
    height = (np.arange(nbin) + 0.5)[::-1] * dh

    # 给出温度坐标.
    tmin = -60
    tmax = 20
    dt = 0.5
    nt = int((tmax - tmin) / dt) + 1
    temp = np.linspace(tmin, tmax, nt)

    # # 测试转换坐标.
    # converter = ProfileConverter(airTemperature, height)
    # precipRate_t = converter.convert3d(precipRate, temp)
    # zFactorCorrected_t = converter.convert3d(zFactorCorrected, temp)
    # tempStormTop = converter.convert2d(heightStormTop)
    # tempZeroDeg = converter.convert2d(heightZeroDeg)

    # 测试分组.
    bins = [0, 1, 5, 10]
    binner = ProfileBinner(
        data=[precipRate, zFactorCorrected],
        bins=bins
    )
    means = binner.mean()
    sems = binner.sem()
    quantiles = binner.quantile([0.25, 0.5, 0.75])

    # # 测试CFAD.
    # dx = 1
    # dy = 0.25
    # xmin, xmax = 10, 60
    # ymin, ymax = 0, 15
    # xbins = np.linspace(xmin, xmax, int((xmax - xmin) / dx) + 1)
    # ybins = np.linspace(ymin, ymax, int((ymax - ymin) / dy) + 1)
    # H = cfad(zFactorCorrected, height, xbins, ybins)
    
