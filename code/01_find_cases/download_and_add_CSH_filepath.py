'''
2022-04-12
下载污染个例和清洁个例对应的CSH文件, 并将文件路径写入个例记录中.

流程:
- 预先准备含下载URL的文本文件, 根据个例的DPR文件路径中的轨道号搜索对应的
  URL并使用wget下载, 最后将下载文件的路径写入到个例记录中.
- 为了节省下载时间, 会保留之前下载好的文件, 并检查是否有重复下载.

输入:
- cases_dusty.json
- cases_clean.json

输出:
- CSH文件.
- cases_dusty.json
- cases_clean.json

参考:
- https://disc.gsfc.nasa.gov/data-access
'''
import json
from pathlib import Path
import subprocess
import sys
sys.path.append('../modules')

import helper_tools

# 读取配置文件,作为全局变量.
with open('config.json', 'r') as f:
    config = json.load(f)

if __name__ == '__main__':
    # 读取两组个例.
    dirpath_result = Path(config['dirpath_result'])
    filepath_dusty = dirpath_result / 'cases_dusty.json'
    filepath_clean = dirpath_result / 'cases_clean.json'
    with open(str(filepath_dusty)) as f:
        cases_dusty = json.load(f)
    with open(str(filepath_clean)) as f:
        cases_clean = json.load(f)

    # URL文件和输出目录的路径.
    dirpath_data = Path(config['dirpath_data'])
    filepath_url = dirpath_data / 'subset_GPM_2HCSH.txt'
    dirpath_output = dirpath_data / 'CSH'
    helper_tools.new_dir(dirpath_output)

    # 将URL与文件轨道号关联.
    dict_url = {}
    with open(str(filepath_url)) as f:
        for line in f:
            url = line.strip('\n')
            filename_CSH = url.split('/')[-1]
            orbit_number = filename_CSH.split('.')[-3]
            dict_url[orbit_number] = url

    # 根据个例的轨道号下载对应的CSH文件.
    for case in cases_dusty + cases_clean:
        case_number = case['case_number']
        orbit_number = case_number.split('_')[0]
        url = dict_url[orbit_number]
        filename_CSH = url.split('/')[-1]
        filepath_CSH = dirpath_output / filename_CSH
        # 不同个例可能对应于同一轨文件, 要避免重复下载.
        if not filepath_CSH.exists():
            args = [
                'wget',
                '--load-cookies', '~/.urs_cookies',
                '--save-cookies', '~/.urs_cookies',
                '--auth-no-challenge=on',
                '--keep-session-cookies',
                '-N', url,
                '-P', str(dirpath_output)
            ]
            subprocess.run(args)
        case['filepath_CSH'] = str(filepath_CSH)

    # 重新写成json文件.
    with open(str(filepath_dusty), 'w') as f:
        json.dump(cases_dusty, f, indent=4)
    with open(str(filepath_clean), 'w') as f:
        json.dump(cases_clean, f, indent=4)
