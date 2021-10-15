#----------------------------------------------------------------------------
# 2021/07/08
# 画出温度坐标下所有的平均廓线.
#
# 廓线变量包括:
# - precipRate
# - Nw
# - Dm
# - SLH
# - VPH
#
# 组图形状为(2, nvar),行表示雨型,列表示廓线变量.
# 没有进行分组,意在展示廓线的总体平均情况.
#----------------------------------------------------------------------------
import json
from pathlib import Path
from collections import namedtuple
import sys
sys.path.append('../modules')

import numpy as np
import xarray as xr
from scipy.stats import mstats
import matplotlib as mpl
import matplotlib.pyplot as plt

from profile_funcs import smooth_profiles
from helper_funcs import letter_subplots

# 读取配置文件,作为全局变量使用.
with open('config.json', 'r') as f:
    config = json.load(f)

# 全局的平滑参数.
SMOOTH = True
SIGMA = 2

if __name__ == '__main__':
    # 读取dusty cases和clean cases.
    input_dirpath = Path(config['input_dirpath'])
    with open(str(input_dirpath / 'found_cases.json'), 'r') as f:
        records = json.load(f)
    dusty_ds = xr.open_dataset(records['dusty']['profile_filepath'])
    clean_ds = xr.open_dataset(records['clean']['profile_filepath'])

    # 若输出目录不存在,那么新建.
    output_dirpath = Path(config['result_dirpath']) / 'Rr_profiles'
    if not output_dirpath.exists():
        output_dirpath.mkdir()

    output_filepath = output_dirpath / 'Rr_mean_profiles.png'
    varnames = ['precipRate_t', 'Nw_t', 'Dm_t', 'SLH_t', 'VPH_t']
    xlabels = [
        'Rain Rate (mm/h)',
        'Nw (dBNw)',
        'Dm (mm)',
        'Latent Heat (K/h)',
        'Latent Heat (K/h)'
    ]
    nvar = len(varnames)

    temp = dusty_ds.temp.data
    npoints = np.zeros((2, 2), dtype=int)
    mean_profiles = np.ma.masked_all((2, 2, 5, len(temp)))
    sem_profiles = np.ma.masked_all((2, 2, 5, len(temp)))
    ds_list = [dusty_ds, clean_ds]
    # 计算多个变量的平均廓线和标准误差廓线.
    for j, ds in enumerate(ds_list):
        for i in range(2):
            rainType = ds.rainType.data
            cond = rainType == i + 1
            npoint = np.count_nonzero(cond)
            # 跳过廓线条数为0的情况.
            if npoint == 0:
                continue
            else:
                npoints[i, j] = npoint
            for k, varname in enumerate(varnames):
                var = ds[varname].isel(npoint=cond).to_masked_array()
                mean_profiles[i, j, k, :] = var.mean(axis=0)
                sem_profiles[i, j, k, :] = mstats.sem(var, axis=0)

    # 仅选取10℃高度以上的数据.
    mean_profiles = mean_profiles[..., temp <= 10]
    sem_profiles = sem_profiles[..., temp <= 10]
    temp = temp[temp <= 10]

    # 进行平滑.
    if SMOOTH:
        mean_profiles = smooth_profiles(mean_profiles, sigma=SIGMA)
        sem_profiles = smooth_profiles(sem_profiles, sigma=SIGMA)

    # 画图用的参数.
    groups = ['Dusty', 'Clean']
    colors = ['C1', 'C0']
    Rstrs = ['Stratiform', 'Convective']
    # 组图形状为(2, nvar).
    # 行表示两种雨型,列表示nvar个廓线变量.每张子图中画出两组的廓线.
    fig, axes = plt.subplots(2, nvar, figsize=(2 * nvar, 6))
    fig.subplots_adjust(wspace=0.3, hspace=0.25)

    # 在每张子图上画出平均廓线和标准误差廓线.
    for i in range(2):
        for j in range(nvar):
            ax = axes[i, j]
            for k in range(2):
                color = colors[k]
                mean_profile = mean_profiles[i, k, j, :]
                sem_profile = sem_profiles[i, k, j, :]

                ax.plot(mean_profile, temp, lw=1, c=color,)
                ax.fill_betweenx(
                    temp,
                    mean_profile - 1.96 * sem_profile,
                    mean_profile + 1.96 * sem_profile,
                    color=color, alpha=0.4
                )

    # 每张子图左上角标出雨型,右上角标出变量名.
    for i ,Rstr in enumerate(Rstrs):
        for ax in axes[i, :]:
            ax.set_title(Rstr, loc='left', fontsize='x-small')
    for j, varname in enumerate(varnames):
        for ax in axes[:, j]:
            ax.set_title(varname.strip('_t'), loc='right', fontsize='x-small')

    # 设置ticks.先设置公共的部分,再设置每个变量对应的部分.
    for ax in axes.flat:
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
        ax.set_ylim(20, -60)
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(20))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(10))
        ax.tick_params(labelsize='x-small')
    for ax in axes[:, 0]:
        ax.set_xlim(0, 1.2 * ax.get_xlim()[1])
    for ax in axes[:, 1]:
        ax.set_xlim(20, 40)
    for ax in axes[:, 2]:
        ax.set_xlim(None, 1.2 * ax.get_xlim()[1])
    for ax in axes[:, 3:].flat:
        xmax = np.max(np.abs(ax.get_xlim()))
        ax.set_xlim(-0.8 * xmax, 1.4 * xmax)

    # 在组图边缘添加label.
    for ax, xlabel in zip(axes[1, :], xlabels):
        ax.set_xlabel(xlabel, fontsize='x-small')
    for ax in axes[:, 0]:
        ax.set_ylabel('Temperature (℃)', fontsize='x-small')

    # 因为最后两列是潜热,所以额外加上x=0处的辅助线.
    for ax in axes[:, 3:].flat:
        ax.axvline(0, color='k', ls='--', lw=0.6)

    # 因为每一列的廓线数目信息是相同的,并且在子图中标出legend会很挤.
    # 所以把legend放在最右边的位置.
    for i in range(2):
        lines = []
        for k in range(2):
            line = mpl.lines.Line2D(
                [], [], c=colors[k], lw=2,
                label=f'{groups[k]}\n({npoints[i, k]})'
            )
            lines.append(line)
        axes[i, -1].legend(
            handles=lines, loc='center left', bbox_to_anchor=(1.05, 0.5),
            handlelength=1, fontsize='x-small'
        )

    # 为子图标出字母标识.
    letter_subplots(axes, (0.08, 0.96), 'x-small')

    # 保存图片.
    fig.savefig(str(output_filepath), dpi=300, bbox_inches='tight')
    plt.close(fig)
