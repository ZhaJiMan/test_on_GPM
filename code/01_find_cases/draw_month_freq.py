'''
2022-04-12
画出污染个例和清洁个例出现在不同月份频次的柱状图.

输入:
- cases_dusty.json
- cases_clean.json

输出:
- 月份频次的直方图.

注意:
- 统计了每个月的频次, 但最后只显示春季的结果.
'''
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# 读取配置文件, 作为全局变量.
with open('config.json') as f:
    config = json.load(f)

def count_month_freq(cases):
    '''统计月份的出现频次.'''
    dates = pd.to_datetime([case['rain_time'] for case in cases])
    freq = np.zeros(12, dtype=int)
    for i in range(12):
        freq[i] = np.count_nonzero(dates.month == i + 1)

    return freq

if __name__ == '__main__':
    # 读取两组个例.
    dirpath_result = Path(config['dirpath_result'])
    with open(str(dirpath_result / 'cases_dusty.json')) as f:
        cases_dusty = json.load(f)
    with open(str(dirpath_result / 'cases_clean.json')) as f:
        cases_clean = json.load(f)

    ncase_dusty = len(cases_dusty)
    ncase_clean = len(cases_clean)
    # 统计月份频次.
    freq_dusty = count_month_freq(cases_dusty)
    freq_clean = count_month_freq(cases_clean)
    print(freq_dusty)
    print(freq_clean)

    # 画直方图.
    fig, ax = plt.subplots()
    x = np.arange(12)
    width = 0.35
    ax.bar(
        x - width / 2, freq_dusty, width, color='C1',
        label=f'Dusty ({ncase_dusty})'
    )
    ax.bar(
        x + width / 2, freq_clean, width, color='C0',
        label=f'Clean ({ncase_clean})'
    )
    ax.legend(loc='upper left')
    ax.set_xlim(1.5, 4.5)  # 只显示春季.
    ax.set_ylim(0, None)
    ax.set_ylabel('Number', fontsize='medium')
    ax.set_xticks([2, 3, 4])
    ax.set_xticklabels(['March', 'April', 'May'])
    ax.yaxis.set_major_locator(mticker.MultipleLocator(4))
    ax.set_title('Frequencies of Cases', fontsize='medium')

    filepath_output = dirpath_result / 'month_freq.png'
    fig.savefig(str(filepath_output), dpi=300, bbox_inches='tight')
    plt.close(fig)
