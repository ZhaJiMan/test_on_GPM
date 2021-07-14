#----------------------------------------------------------------------------
# 2021/05/08
# 对污染组和清洁组每个个例的廓线数据进行简单统计,并汇总为CSV表格.
#
# 统计的变量有
# - 平均地表高度.
# - 平均零度层高度.
# - 平均雨顶高度.
# - 平均地表降水率.
# - 总的廓线数目和四种雨型的廓线数目和占比.
#----------------------------------------------------------------------------
import json
from pathlib import Path
import sys
sys.path.append('../modules')
from helper_funcs import recreate_dir

import numpy as np
import pandas as pd
import xarray as xr

# 读取配置文件,作为全局变量.
with open('config.json', 'r') as f:
    config = json.load(f)

def create_table_cases(cases, output_filepath):
    '''统计一系列个例的的平均值与雨型组成,并保存为CSV文件.'''
    row_dict1 = {}
    row_dict2 = {}
    # 读取数据.
    for case in cases:
        time = case['rain_time']
        ds = xr.load_dataset(case['profile_filepath'])
        rainType = ds.rainType.data
        # DataArray计算平均值时会自动忽略NaN.
        # 但是计算出的标量依旧是DataArray对象,需要手动转换.
        row_dict1[time] = {
            'elv (km)': float(ds.elevation.mean()),
            'h0 (km)': float(ds.heightZeroDeg.mean()),
            'sh (km)': float(ds.heightStormTop.mean()),
            'NSRR (mm/h)': float(ds.precipRateNearSurface.mean())
        }
        row_dict2[time] = {
            'all': np.count_nonzero(rainType > 0),
            'stra.': np.count_nonzero(rainType == 1),
            'conv.': np.count_nonzero(rainType == 2),
            'shall.': np.count_nonzero(rainType == 3),
            'other': np.count_nonzero(rainType == 4)
        }
    df1 = pd.DataFrame.from_dict(row_dict1, orient='index')
    df2 = pd.DataFrame.from_dict(row_dict2, orient='index')

    # 向df1最后追加平均值row.
    df1.loc['total', :] = df1.mean()
    df1 = df1.round(decimals=2)     # 控制小数位.
    # 向df2最后追加总和row.
    df2.loc['total', :] = df2.sum()
    df2 = df2.astype(int)       # loc赋值后莫名其妙会变成float.

    # 为每种雨型的统计追加百分比字符串.
    formatter = lambda x: f'{x:.1f}'
    for col in df2.columns[1:]:
        ratio = df2[col] / df2['all'] * 100
        df2[col] = df2[col].astype(str) + ' (' + ratio.map(formatter) + '%)'

    # 将两个df连接起来.
    df = pd.concat([df1, df2], axis=1)
    # 保存文件.
    df.to_csv(str(output_filepath))

def create_table_merged(records, output_filepath):
    '''统计合并后的两组的廓线占比与平均降水强度.'''
    dusty_ds = xr.load_dataset(records['dusty']['profile_filepath'])
    clean_ds = xr.load_dataset(records['clean']['profile_filepath'])
    ds_list = [dusty_ds, clean_ds]

    Rtypes = ['Stratiform', 'Convective', 'Shallow']
    names = ['Number', 'Rain']
    groups = ['Dusty', 'Clean']
    df_counts = pd.DataFrame(
        np.zeros((2, 3), dtype=int),
        index=groups, columns=Rtypes
    )
    df_means = pd.DataFrame(
        np.zeros((2, 3), dtype=float),
        index=groups, columns=Rtypes
    )
    # 统计三种雨型,不同污染分组的廓线数和平均降水速率.
    for i, ds in enumerate(ds_list):
        rainType = ds.rainType.data
        surfRr = ds.precipRateNearSurface.data
        for j in range(3):
            flag = rainType == j + 1
            df_counts.iloc[i, j] = np.count_nonzero(flag)
            df_means.iloc[i, j] = surfRr[flag].mean()

    # 计算廓线数的百分比,并将其与廓线数合并为字符串.
    df_ratios = df_counts.div(df_counts.sum(axis=1), axis=0) * 100
    formatter = lambda x: f'{x:.1f}'
    df_strs = \
        df_counts.astype(str) + \
        ' (' + df_ratios.applymap(formatter)  + '%)'
    # 降水速率保留两位小数.
    df_means = df_means.round(decimals=2)

    # 使用多重索引.
    multi_index = pd.MultiIndex.from_product([Rtypes, names])
    df_all = pd.DataFrame(
        np.zeros((2, 6)),
        index=groups, columns=multi_index
    )
    # 为了避免columns无法对齐,用values进行赋值.
    df_all.loc[:, (slice(None), 'Number')] = df_strs.values
    df_all.loc[:, (slice(None), 'Rain')] = df_means.values

    # 保存文件.
    df_all.to_csv(str(output_filepath))

if __name__ == '__main__':
    # 读取dusty cases和clean cases.
    result_dirpath = Path(config['result_dirpath'])
    with open(str(result_dirpath / 'found_cases.json'), 'r') as f:
        records = json.load(f)
    dusty_cases = records['dusty']['cases']
    clean_cases = records['clean']['cases']

    # 重新创建输出目录.
    output_dirpath = result_dirpath / 'tables'
    recreate_dir(output_dirpath)

    create_table_cases(dusty_cases, output_dirpath / 'dusty_table.csv')
    create_table_cases(clean_cases, output_dirpath / 'clean_table.csv')
    create_table_merged(records, output_dirpath / 'merged_table.csv')
