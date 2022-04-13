'''
2022-04-12
创建配置文件, 保存其它脚本共用的参数, 并创建存储结果的目录.
'''
import json
from pathlib import Path
import subprocess
import sys
sys.path.append('../modules')

import helper_tools

if __name__ == '__main__':
    # 运行该程序时就会将配置写入到config.json文件中.
    config = {
        'extents_map': [100, 130, 25, 50],
        'extents_DPR': [110, 124, 32, 45],
        'dirpath_input': '/d4/wangj/dust_precipitation/results/01_find_cases',
        'dirpath_data': '/d4/wangj/dust_precipitation/data',
        'dirpath_result': '/d4/wangj/dust_precipitation/results/02_extract_profiles',
    }
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)

    # 创建result目录.
    dirpath_result = Path(config['dirpath_result'])
    helper_tools.new_dir(dirpath_result)

    # # 按顺序运行脚本.
    # subprocess.run(['python', 'extract_profiles.py'])
    # subprocess.run(['python', 'draw_rain_distribution.py'])
    # subprocess.run(['python', 'draw_CFAD_for_cases.py'])
    # subprocess.run(['python', 'draw_Rtype_plot.py'])
    # subprocess.run(['python', 'simple_table.py'])

