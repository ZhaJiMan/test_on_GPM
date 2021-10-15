#----------------------------------------------------------------------------
# 2021/07/08
# 反演每个污染和清洁个例的VPH潜热,并将生成的文件路径记录到个例中.
#
# 反演通过调用LH_retrieve.ncl脚本实现.需要利用到个例的DPR和ENV文件.
#
# 反演速度非常慢,故采用多进程.因为不同个例可能共用同一个VPH文件,
# 所以改动程序时需要避免出现不同进程同时操作同一个文件的情况.
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

def to_VPH_filename(DPR_filename):
    '''根据DPR文件名生成VPH文件名.'''
    # 根据DPR文件名生成VPH文件名.
    parts = DPR_filename.split('.')
    parts[0] = '2A-LH'
    parts.pop(3)
    parts.pop(-2)
    parts[-1] = 'nc'
    VPH_filename = '.'.join(parts)

    return VPH_filename

def call_ncl_script(DPR_filepath, ENV_filepath, VPH_filepath):
    '''调用NCL脚本,利用DPR数据和ENV数据反演VPH,并保存到指定路径.'''
    # 生成用于命令行的参数.
    file1 = '"' + str(DPR_filepath) + '"'
    fienv = '"' + str(ENV_filepath) + '"'
    newfi = '"' + str(VPH_filepath) + '"'
    # 调用NCL脚本.需要小心单引号和双引号的使用.
    subprocess.run(
        f"ncl -Q LHretrieve.ncl 'file1={file1}' 'fienv={fienv}' 'newfi={newfi}'",
        shell=True
    )

if __name__ == '__main__':
    result_dirpath = Path(config['result_dirpath'])
    with open(str(result_dirpath / 'found_cases.json'), 'r') as f:
        records = json.load(f)
    dusty_cases = records['dusty']['cases']
    clean_cases = records['clean']['cases']

    # 每次运行时重新创建输出目录.
    output_dirpath = Path(config['temp_dirpath']) / 'VPH'
    recreate_dir(output_dirpath)

    p = Pool(8)
    works = []      # 反演进程的列表.
    for case in dusty_cases + clean_cases:
        DPR_filepath = Path(case['DPR_filepath'])
        ENV_filepath = Path(case['ENV_filepath'])
        VPH_filename = to_VPH_filename(DPR_filepath.name)
        VPH_filepath = output_dirpath / VPH_filename

        # 不同个例可能处在同一轨GPM数据上,避免重复反演.
        if not VPH_filepath in works:
            works.append(VPH_filepath)
            p.apply_async(
                func=call_ncl_script,
                args=(DPR_filepath, ENV_filepath, VPH_filepath)
            )
        case['VPH_filepath'] = str(VPH_filepath)
    p.close()
    p.join()

    # 重新保存修改后的case.
    with open(str(result_dirpath / 'found_cases.json'), 'w') as f:
        json.dump(records, f, indent=4)
