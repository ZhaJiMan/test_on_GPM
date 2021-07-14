#----------------------------------------------------------------------------
# 2021/05/08
# 统计找出的污染个例与清洁个例的月份频率,并画在一张图上.
#
# 虽然统计了12个月,但具体画图的月份范围需要在程序中修改.
# 目前只画出了春季(3-5)的频率.
#----------------------------------------------------------------------------
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

# 读取配置文件,将config作为全局变量.
with open('config.json', 'r') as f:
    config = json.load(f)

def count_month(dates):
    '''统计月份的出现频次.'''
    months = dates.month
    counts = np.zeros(12, dtype=int)
    for i in range(12):
        counts[i] = np.count_nonzero(months == i + 1)

    return counts

if __name__ == '__main__':
    result_dirpath = Path(config['result_dirpath'])
    dusty_filepath = result_dirpath / 'dusty_cases.json'
    clean_filepath = result_dirpath / 'clean_cases.json'
    with open(str(dusty_filepath), 'r') as f:
        dusty_cases = json.load(f)
    with open(str(clean_filepath), 'r') as f:
        clean_cases = json.load(f)

    # 统计月份频次.
    dusty_dates = pd.to_datetime([case['rain_time'] for case in dusty_cases])
    clean_dates = pd.to_datetime([case['rain_time'] for case in clean_cases])
    dusty_freq = count_month(dusty_dates)
    clean_freq = count_month(clean_dates)

    # 画图.
    fig, ax = plt.subplots()
    x = np.arange(1, 13)
    width = 0.35
    ax.bar(
        x - width / 2, dusty_freq, width, color='C1',
        label=f'Dusty({len(dusty_dates)})'
    )
    ax.bar(
        x + width / 2, clean_freq, width, color='C0',
        label=f'Clean({len(clean_dates)})'
    )
    ax.legend(loc='upper left')
    ax.set_xlim(2.5, 5.5)       # 只显示春季.
    ax.set_ylim(0, None)
    ax.set_xlabel('Month', fontsize='medium')
    ax.set_ylabel('Number', fontsize='medium')
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
    ax.set_title('Frequencies of Cases', fontsize='medium')

    output_filepath = result_dirpath / 'month_freq.png'
    fig.savefig(str(output_filepath), dpi=300, bbox_inches='tight')
    plt.close(fig)
