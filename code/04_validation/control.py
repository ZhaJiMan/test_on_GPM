#----------------------------------------------------------------------------
# 2021/10/15
# 将一些参数写入到json文件中,以便于其它子程序调用.并创建存储结果的目录.
#----------------------------------------------------------------------------
import json
from pathlib import Path

if __name__ == '__main__':
    # 运行该程序时就会将参数写入到config.json文件中.
    config = {
        'map_extent': [100, 130, 25, 50],
        'DPR_extent': [110, 124, 32, 45],
        'start_time': '2014-03-08',
        'end_time': '2020-12-31',
        'data_dirpath': '/d4/wangj/dust_precipitation/data',
        'temp_dirpath': '/d4/wangj/dust_precipitation/data/validation_temp',
        'result_dirpath': '/d4/wangj/dust_precipitation/results/04_validation',
    }
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)

    # 创建临时数据的目录.
    temp_dirpath = Path(config['temp_dirpath'])
    if not temp_dirpath.exists():
        temp_dirpath.mkdir()
    # 创建result目录.
    result_dirpath = Path(config['result_dirpath'])
    if not result_dirpath.exists():
        result_dirpath.mkdir()
