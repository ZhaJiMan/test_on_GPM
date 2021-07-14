#----------------------------------------------------------------------------
# 2021/05/08
# 将一些参数写入到json文件中,以便于其它子程序调用.并创建存储结果的目录.
#----------------------------------------------------------------------------
import json
from pathlib import Path

if __name__ == '__main__':
    # 运行该程序时就会将配置写入到config.json文件中.
    config = {
        'map_extent': [100, 130, 25, 50],
        'DPR_extent': [110, 124, 32, 45],
        'input_dirpath': '/d4/wangj/dust_precipitation/results/01_find_cases',
        'data_dirpath': '/d4/wangj/dust_precipitation/data',
        'result_dirpath': '/d4/wangj/dust_precipitation/results/02_extract_profiles'
    }
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)

    # 创建result目录.
    result_dirpath = Path(config['result_dirpath'])
    if not result_dirpath.exists():
        result_dirpath.mkdir()
