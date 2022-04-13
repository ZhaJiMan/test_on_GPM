'''
2022-02-10
创建配置文件, 保存其它脚本所需的参数, 并创建存储结果的目录.
'''
import json
from pathlib import Path
import sys
sys.path.append('../modules')

import helper_tools

if __name__ == '__main__':
    # 运行该程序时会将参数写入到config.json文件中.
    config = {
        'extents_map': [100, 130, 25, 50],
        'extents_DPR': [110, 124, 32, 45],
        'time_start': '2014-01-01',
        'time_end': '2020-12-31',
        'dirpath_data': '/d4/wangj/dust_precipitation/data',
        'dirpath_result': '/d4/wangj/dust_precipitation/results/00_region_properties',
    }
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)

    # 创建result目录.
    dirpath_result = Path(config['dirpath_result'])
    helper_tools.new_dir(dirpath_result)
