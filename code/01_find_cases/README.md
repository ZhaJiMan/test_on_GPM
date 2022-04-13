## 脚本执行的顺序

### 1.寻找降水个例

- find_rain_cases

- draw_rain_cases

### 2.给降水个例添加气溶胶数据

- add_AOD_records_to_rain_cases

- add_CAL_record_to_rain_cases

### 3.筛选污染和清洁个例

- find_dusty_cases

### 4.给污染和清洁个例添加其它数据

- merge_and_add_ERA5_filepath_to_dusty_cases

- download_and_add_CSH_filepath

- download_and_add_GMI_filepath

- retrieve_and_add_VPH_filepath

### 5.画出污染和清洁个例

- draw_dusty_cases

- draw_month_freq
