'''
2022-04-12
为每个污染和清洁个例反演VPH潜热, 并将文件路径写入个例记录.

流程:
- 读取个例的DPR和ENV文件, 命令行调用NCL脚本反演得到VPH文件, 并将文件路径写入
  个例记录中.
- 为节省运行时间, 保留每次反演的结果, 并检查是否发生重复反演.

输入:
- cases_dusty.json
- cases_clean.json

输出:
- VPH文件.
- cases_dusty.json
- cases_clean.json

注意:
- 脚本使用了多进程.
'''
import json
from pathlib import Path
import subprocess
from multiprocessing import Pool
import sys
sys.path.append('../modules')

import helper_tools

# 读取配置文件,作为全局变量.
with open('config.json', 'r') as f:
    config = json.load(f)

def to_VPH_filename(filename_DPR):
    '''根据DPR文件名生成VPH文件名.'''
    parts = filename_DPR.split('.')
    parts[0] = '2A-LH'
    parts.pop(3)
    parts.pop(-2)
    parts[-1] = 'nc'
    filename_VPH = '.'.join(parts)

    return filename_VPH

def call_ncl_script(filepath_DPR, filepath_ENV, filepath_VPH):
    '''调用NCL脚本用DPR数据和ENV数据反演VPH, 并保存到指定路径.'''
    args = [
        'ncl',
        '-Q', 'LHretrieve.ncl',
        f'file1="{str(filepath_DPR)}"',
        f'fienv="{str(filepath_ENV)}"',
        f'newfi="{str(filepath_VPH)}"'
    ]
    subprocess.run(args)

if __name__ == '__main__':
    # 读取两组个例.
    dirpath_result = Path(config['dirpath_result'])
    filepath_dusty = dirpath_result / 'cases_dusty.json'
    filepath_clean = dirpath_result / 'cases_clean.json'
    with open(str(filepath_dusty)) as f:
        cases_dusty = json.load(f)
    with open(str(filepath_clean)) as f:
        cases_clean = json.load(f)

    # 创建输出目录.
    dirpath_output = Path(config['dirpath_data'], 'VPH')
    helper_tools.new_dir(dirpath_output)

    # 为了避免两个反演进程同时操作同一文件, 用tasks记录反演任务.
    tasks = []
    p = Pool(10)
    for case in cases_dusty + cases_clean:
        filepath_DPR = Path(case['filepath_DPR'])
        filepath_ENV = Path(case['filepath_ENV'])
        filename_VPH = to_VPH_filename(filepath_DPR.name)
        filepath_VPH = dirpath_output / filename_VPH

        # 若VPH文件已经存在, 或之前已经加入任务列表, 则跳过反演.
        if filepath_VPH.exists() or filepath_VPH in tasks:
            pass
        else:
            tasks.append(filepath_VPH)
            p.apply_async(
                call_ncl_script,
                args=(filepath_DPR, filepath_ENV, filepath_VPH)
            )
        case['filepath_VPH'] = str(filepath_VPH)
    p.close()
    p.join()

    # 重新写成json文件.
    with open(str(filepath_dusty), 'w') as f:
        json.dump(cases_dusty, f, indent=4)
    with open(str(filepath_clean), 'w') as f:
        json.dump(cases_clean, f, indent=4)
