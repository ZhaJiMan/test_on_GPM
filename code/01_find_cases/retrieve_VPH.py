#----------------------------------------------------------------------------
# 2021/07/08
# 反演每个污染和清洁个例的VPH潜热.
#
# 反演通过调用LH_retrieve.ncl脚本实现.需要利用到个例的DPR和ENV文件.
#----------------------------------------------------------------------------
import json
from pathlib import Path
import subprocess
from multiprocessing import Pool

import sys
sys.path.append('../modules')
from helper_funcs import recreate_dir

# 读取配置文件,作为全局变量.
with open('config.json', 'r') as f:
    config = json.load(f)

def process_one_case(case, output_dirpath):
    '''反演个例的VPH文件,并返回对应的路径.'''
    DPR_filepath = Path(case['DPR_filepath'])
    ENV_filepath = Path(case['ENV_filepath'])

    # 根据DPR文件名生成VPH文件名.
    DPR_filename = DPR_filepath.name
    parts = DPR_filename.split('.')
    parts[0] = '2A-LH'
    parts.pop(3)
    parts.pop(-2)
    parts[-1] = 'nc'
    VPH_filename = '.'.join(parts)
    VPH_filepath = output_dirpath / VPH_filename

    # 生成用于命令行的参数.
    file1 = '"' + str(DPR_filepath) + '"'
    fienv = '"' + str(ENV_filepath) + '"'
    newfi = '"' + str(VPH_filepath) + '"'

    # 调用NCL脚本.需要小心单引号和双引号的使用.
    subprocess.run(
        f"ncl -Q LHretrieve.ncl 'file1={file1}' 'fienv={fienv}' 'newfi={newfi}'",
        shell=True
    )

    return VPH_filepath

if __name__ == '__main__':
    input_filepath = Path(config['result_dirpath']) / 'found_cases.json'
    with open(str(input_filepath), 'r') as f:
        records = json.load(f)
    dusty_cases = records['dusty']['cases']
    clean_cases = records['clean']['cases']

    # 每次运行时重新创建输出目录.
    output_dirpath = Path(config['data_dirpath']) / 'VPH'
    recreate_dir(output_dirpath)

    # 对每个个例进行反演.
    all_cases = dusty_cases + clean_cases
    p = Pool(16)
    VPH_filepaths = p.starmap(
        process_one_case,
        zip(all_cases, [output_dirpath] * len(all_cases))
    )
    p.close()
    p.join()
    # 将反演得到的文件路径记录下来.
    for case, VPH_filepath in zip(all_cases, VPH_filepaths):
        case['VPH_filepath'] = str(VPH_filepath)

    # 重新保存修改后的case.
    with open(str(input_filepath), 'w') as f:
        json.dump(records, f, indent=4)
