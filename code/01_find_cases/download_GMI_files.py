#----------------------------------------------------------------------------
# 2021/05/08
# 为找出的所有个例下载对应的GMI 1B资料,并将文件路径写入到记录中.
#
# 因为服务器上暂无完整的GMI数据,所以这里进行手动下载.
# 下载可能需要数分钟,如有需求可以改写成多进程.
#
# 下载通过wget实现,命令见GES DISC网站说明.
# 需要提前准备好含有2014~2020年数据的下载链接的subset文本文件.
#----------------------------------------------------------------------------
import json
from pathlib import Path
import subprocess
import sys
sys.path.append('../modules')
from helper_funcs import recreate_dir

# 读取配置文件,作为全局变量.
with open('config.json', 'r') as f:
    config = json.load(f)

if __name__ == '__main__':
    input_filepath = Path(config['result_dirpath']) / 'found_cases.json'
    with open(str(input_filepath), 'r') as f:
        records = json.load(f)
    dusty_cases = records['dusty']['cases']
    clean_cases = records['clean']['cases']

    # 因为GMI目录内有下载用的txt文件,所以不要重新创建GMI目录.
    GMI_dirpath = Path(config['data_dirpath']) / 'GMI'
    download_list_filepath = GMI_dirpath / 'subset_GPM_1BGMI.txt'
    # 重新创建输出目录.
    dusty_dirpath = GMI_dirpath / 'dusty_cases'
    clean_dirpath = GMI_dirpath / 'clean_cases'
    recreate_dir(dusty_dirpath)
    recreate_dir(clean_dirpath)

    # 读取含有下载链接的文本文件,将其中的链接与orbit number对应起来.
    link_dict = {}
    with open(str(download_list_filepath), 'r') as f:
        for line in f:
            link = line.strip('\n')
            GMI_filename = link.split('/')[-1]
            orbit_number = GMI_filename.split('.')[-3]
            link_dict[orbit_number] = link

    # 根据每个case含有的DPR文件的orbit number下载对应的GMI文件.
    # 并将下载到的GMI文件路径写入到case中.
    cases_list = [dusty_cases, clean_cases]
    dirpath_list = [dusty_dirpath, clean_dirpath]
    for cases, dirpath in zip(cases_list, dirpath_list):
        for case in cases:
            DPR_filepath = Path(case['DPR_filepath'])
            DPR_filename = DPR_filepath.name
            orbit_number = DPR_filename.split('.')[-3]
            link = link_dict[orbit_number]
            command = f'wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --auth-no-challenge=on --keep-session-cookies --content-disposition -N {link} -P {str(dirpath)}'
            subprocess.run(command, shell=True)
            GMI_filename = link.split('/')[-1]
            GMI_filepath = dirpath / GMI_filename
            case['GMI_filepath'] = str(GMI_filepath)

    # 重新保存修改后的case.
    with open(str(input_filepath), 'w') as f:
        json.dump(records, f, indent=4)
