#----------------------------------------------------------------------------
# 2021/05/08
# 为找出的所有个例下载对应的GMI 1B资料,并将文件路径写入到记录中.
#
# 因为服务器上暂无完整的GMI数据,所以这里进行手动下载.下载可能需要数分钟.
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
    result_dirpath = Path(config['result_dirpath'])
    with open(str(result_dirpath / 'found_cases.json'), 'r') as f:
        records = json.load(f)
    dusty_cases = records['dusty']['cases']
    clean_cases = records['clean']['cases']

    # 设置用于下载GMI数据的txt文件的路径,并重新创建存储GMI数据的目录.
    data_dirpath = Path(config['data_dirpath'])
    temp_dirpath = Path(config['temp_dirpath'])
    download_list_filepath = data_dirpath / 'subset_GPM_1BGMI.txt'
    output_dirpath = temp_dirpath / 'GMI'
    recreate_dir(output_dirpath)

    # 读取含有下载链接的文本文件,将其中的链接与orbit number对应起来.
    link_dict = {}
    with open(str(download_list_filepath), 'r') as f:
        for line in f:
            link = line.strip('\n')
            GMI_filename = link.split('/')[-1]
            orbit_number = GMI_filename.split('.')[-3]
            link_dict[orbit_number] = link

    # 根据每个case含有的DPR文件的orbit number下载对应的GMI文件.
    for case in dusty_cases + clean_cases:
        DPR_filepath = Path(case['DPR_filepath'])
        DPR_filename = DPR_filepath.name
        orbit_number = DPR_filename.split('.')[-3]
        link = link_dict[orbit_number]
        GMI_filename = link.split('/')[-1]
        GMI_filepath = output_dirpath / GMI_filename
        # 不同个例可能处在同一轨GPM数据上,避免重复下载.
        if not GMI_filepath.exists():
            command = f'wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --auth-no-challenge=on --keep-session-cookies --content-disposition -N {link} -P {str(output_dirpath)}'
            subprocess.run(command, shell=True)
        case['GMI_filepath'] = str(GMI_filepath)

    # 重新保存修改后的case.
    with open(str(result_dirpath / 'found_cases.json'), 'w') as f:
        json.dump(records, f, indent=4)
