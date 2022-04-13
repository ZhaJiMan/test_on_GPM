'''
2022-04-12
对污染组和清洁组的样本进行简单统计, 并汇总为CSV表格.
表格分为两种:
- 统计所有污染个例或所有清洁个例不同雨型的廓线样本数及其占比.
- 统计合并后的污染组和清洁组三种雨型(层云, 对流和浅云)的廓线样本数, 以及平均
  地表降水率.
'''
import json
from pathlib import Path
import sys
sys.path.append('../modules')

import numpy as np
import pandas as pd
import xarray as xr

import helper_tools

# 读取配置文件, 将config作为全局变量.
with open('config.json') as f:
    config = json.load(f)

def make_table_for_all_cases(cases, filepath_output):
    '''统计一组个例不同雨型的廓线数目及其占比, 保存为CSV文件.'''
    # 统计每个个例的廓线数目.
    dict_row = {}
    for case in cases:
        rain_time = case['rain_time']
        with xr.open_dataset(case['filepath_profile']) as ds:
            rainType = ds.rainType.values
        dict_row[rain_time] = {
            'stratiform': np.count_nonzero(rainType == 1),
            'convective': np.count_nonzero(rainType == 2),
            'shallow': np.count_nonzero(rainType == 3),
        }
    df_num = pd.DataFrame.from_dict(dict_row, orient='index')
    # 向最后一行添加总和.
    df_num.loc['total', :] = df_num.sum()
    df_num = df_num.astype(int)  # 有时会自动变成浮点型.

    # 计算廓线数的百分比, 并将其与廓线数合并为字符串.
    df_ratio = df_num.div(df_num.sum(axis=1), axis=0) * 100
    formatter = lambda x: f'{x:.1f}'
    df_str = df_num.astype(str) + ' (' + df_ratio.applymap(formatter) + '%)'

    # 保存文件.
    df_str.to_csv(str(filepath_output))

def make_table_merged(ds_dusty, ds_clean, filepath_output):
    '''统计合并后两组的廓线数目和平均降水率, 保存为CSV文件.'''
    # 设置标签.
    Rtypes = ['Stratiform', 'Convective', 'Shallow']
    names = ['Number', 'Rain']
    groups = ['Dusty', 'Clean']

    df_num = pd.DataFrame(
        np.zeros((2, 3), dtype=int),
        index=groups, columns=Rtypes
    )
    df_rain = pd.DataFrame(
        np.zeros((2, 3), dtype=float),
        index=groups, columns=Rtypes
    )

    # 统计两种分组三种雨型的廓线数和平均降水速率.
    for i, ds in enumerate([ds_dusty, ds_clean]):
        rainType = ds.rainType.values
        precipRateSurface = ds.precipRateNearSurface.values
        for j in range(3):
            mask = rainType == j + 1
            df_num.iloc[i, j] = np.count_nonzero(mask)
            df_rain.iloc[i, j] = precipRateSurface[mask].mean()

    # 计算廓线数的百分比, 并将其与廓线数合并为字符串.
    df_ratio = df_num.div(df_num.sum(axis=1), axis=0) * 100
    formatter = lambda x: f'{x:.1f}'
    df_str = df_num.astype(str) + ' (' + df_ratio.applymap(formatter) + '%)'
    # 降水速率保留两位小数.
    df_rain = df_rain.round(decimals=2)

    # 使用多重索引.
    multi_index = pd.MultiIndex.from_product([Rtypes, names])
    df_all = pd.DataFrame(
        np.zeros((2, 6), dtype=float),
        index=groups, columns=multi_index
    )
    # 为了避免columns无法对齐, 用values进行赋值.
    df_all.loc[:, (slice(None), 'Number')] = df_str.values
    df_all.loc[:, (slice(None), 'Rain')] = df_rain.values

    # 保存文件.
    df_all.to_csv(str(filepath_output))

if __name__ == '__main__':
    # 读取两组个例的记录和数据集.
    dirpath_input = Path(config['dirpath_input'])
    dirpath_merged = Path(config['dirpath_data'], 'DPR_case', 'merged')
    with open(str(dirpath_input / 'cases_dusty.json')) as f:
        cases_dusty = json.load(f)
    with open(str(dirpath_input / 'cases_clean.json')) as f:
        cases_clean = json.load(f)
    ds_dusty = xr.load_dataset(str(dirpath_merged / 'data_dusty.nc'))
    ds_clean = xr.load_dataset(str(dirpath_merged / 'data_clean.nc'))

    # 创建输出目录.
    dirpath_output = Path(config['dirpath_result'], 'tables')
    helper_tools.renew_dir(dirpath_output)

    # 创建表格.
    make_table_for_all_cases(cases_dusty, dirpath_output / 'table_dusty.csv')
    make_table_for_all_cases(cases_clean, dirpath_output / 'table_clean.csv')
    make_table_merged(ds_dusty, ds_clean, dirpath_output / 'table_merged.csv')
