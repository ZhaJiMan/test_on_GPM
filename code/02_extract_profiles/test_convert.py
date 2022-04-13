'''
2022-04-12
展示高度坐标转换为温度坐标的效果.
组图形状为(1, 3), 第一张展示气温廓线随高度线性递减的趋势, 第二张和第三张展示
降水率廓线转换前后的形状.

注意:
- 为了能让脚本独立运行, 具体文件和索引都硬编码在脚本里.
'''
import json
from pathlib import Path
import sys
sys.path.append('../modules')

import h5py
import numpy as np
import xarray as xr
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import helper_tools
import data_tools
import profile_tools
import plot_tools

# 读取配置文件, 将config作为全局变量.
with open('config.json') as f:
    config = json.load(f)

if __name__ == '__main__':
    # 从个例掩膜中得到的层云降水的索引.
    index = (
        np.array([
            2900, 2900, 2901, 2901, 2902, 2902, 2903, 2903, 2904, 2904, 2904,
            2905, 2905, 2905, 2905, 2905, 2905, 2905, 2906, 2906, 2906, 2906,
            2906, 2907, 2907, 2907, 2907, 2908, 2908, 2909, 2909, 2909, 2910,
            2910, 2910, 2911, 2911, 2911, 2911, 2911, 2912, 2912, 2912, 2912,
            2912, 2913, 2913, 2913, 2913, 2913, 2913, 2914, 2914, 2914, 2914,
            2914, 2915, 2915, 2915, 2915, 2916, 2916, 2917, 2917, 2917, 2918,
            2918, 2918, 2918, 2919, 2919, 2919, 2919, 2919, 2919, 2920, 2920,
            2920, 2920, 2920, 2921, 2921, 2923, 2923, 2923, 2923, 2924, 2924,
            2924, 2924, 2924, 2925, 2925, 2925, 2926, 2926, 2926, 2927, 2927,
            2927, 2927, 2928, 2928, 2928, 2929, 2930, 2933, 2934, 2934, 2935,
            2936, 2936, 2937, 2937, 2938, 2938
        ]),
        np.array([
            5, 6, 5, 6, 5, 6, 4, 5, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 4, 5,
            6, 1, 3, 4, 7, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2, 3,
            4, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 0, 1, 2, 3, 1, 3, 0, 1, 2, 1,
            2, 3, 5, 1, 2, 3, 4, 5, 6, 1, 3, 4, 5, 6, 3, 5, 1, 2, 4, 5, 1, 2,
            4, 5, 6, 2, 3, 5, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 4, 4, 3, 3, 4, 4,
            3, 4, 3, 4, 4, 5
        ])
    )

    # 读取DPR和ENV数据.
    filepath_DPR = Path('/data00/0/GPM/DPR/V06/2014/201405/2A.GPM.DPR.V8-20180723.20140518-S113517-E130750.001242.V06A.HDF5')
    filepath_ENV = Path('/data00/0/GPM/ENV/V06/2014/2A-ENV.GPM.DPR.V8-20180723.20140518-S113517-E130750.001242.V06A.HDF5')
    with data_tools.ReaderDPR(str(filepath_DPR)) as f:
        heightZeroDeg = f.read_ds('VER/heightZeroDeg')[index]
        precipRate = f.read_ds('SLV/precipRate')[index]
    heightZeroDeg /= 1000
    with h5py.File(str(filepath_ENV)) as f:
        airTemperature = f['NS/VERENV/airTemperature'][:][index]
    airTemperature -= 273.15

    # 给出DPR的高度坐标.
    nbin = 176
    dh = 0.125  # 单位为km.
    height = (np.arange(nbin) + 0.5)[::-1] * dh
    # 给出目标温度坐标.
    tmin = -60
    tmax = 20
    dt = 0.5
    nt = int((tmax - tmin) / dt) + 1
    temp = np.linspace(tmin, tmax, nt)

    # 转换廓线数据.
    converter = profile_tools.ProfileConverter(airTemperature, height)
    precipRate_t = converter.convert3d(precipRate, temp)

    # 计算平均廓线.
    heightZeroDeg = np.nanmean(heightZeroDeg)
    airTemperature = airTemperature.mean(axis=0)
    airTemperature_fitted = converter.airTemperature_fitted.mean(axis=0)
    precipRate = np.nanmean(precipRate, axis=0)
    precipRate_t = np.nanmean(precipRate_t, axis=0)
    # 平滑廓线.
    precipRate = profile_tools.smooth_profiles(precipRate, sigma=1)
    precipRate_t = profile_tools.smooth_profiles(precipRate_t, sigma=1)

    # 近地表的降水廓线设为缺测.
    precipRate[height < 1.5] = np.nan
    precipRate_t[temp > 10] = np.nan

    # 组图形状为(1, 3).
    # 第一张画气温平均廓线及其线性拟合的结果.
    # 第二张和第三张分别画高度坐标和温度坐标下的降水率平均廓线.
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.4)

    # 画出气温廓线.
    axes[0].plot(airTemperature, height, lw=2, c='C3', label='2ADPRENV')
    axes[0].plot(
        airTemperature_fitted, height,
        lw=1, c='magenta', ls='--', label='fitted'
    )
    axes[0].legend(fontsize='small', loc='upper right')
    # 画出降水率廓线.
    axes[1].plot(precipRate, height, lw=2, c='C0')
    axes[2].plot(precipRate_t, temp, lw=2, c='C0')
    # 添加零度层辅助线.
    for ax in axes[:2]:
        ax.axhline(heightZeroDeg, lw=1, color='k', ls='--')
    axes[2].axhline(0, lw=1, color='k', ls='--')

    # 设置坐标轴.
    axes[0].set_xlim(-60, 20)
    axes[0].set_xlabel('Temperature (℃)', fontsize='medium')
    for ax in axes[:2]:
        ax.set_ylim(0, 12)
        ax.set_ylabel('Height', fontsize='medium')
        ax.yaxis.set_major_locator(mticker.MultipleLocator(2))
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    for ax in axes[1:]:
        ax.set_xlim(0, None)
        ax.set_xlabel('Rain Rate (mm/h)', fontsize='medium')
    axes[2].set_ylim(20, -60)
    axes[2].set_ylabel('Temperature (℃)', fontsize='medium')
    axes[2].yaxis.set_major_locator(mticker.MultipleLocator(20))
    axes[2].yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    for ax in axes:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.tick_params(labelsize='small')

    # 为子图标出字母标识.
    plot_tools.letter_axes(axes, 0.07, 0.96, fontsize='medium')

    # 保存图片.
    dirpath_output = Path(config['dirpath_result'])
    filepath_output = dirpath_output / 'test_convert.png'
    fig.savefig(str(filepath_output), dpi=300, bbox_inches='tight')
    plt.close()
