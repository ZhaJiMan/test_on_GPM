# GPM DPR 系列产品的文件路径

存放 GPM 文件的两个目录：

- /data04/0/gpm

- /data00/0/GPM

不同产品的路径为：

- DPR/V06/{YYYY}/{YYYYMM}/2A.GPM.DPR.V8-20180723.{YYYYMMDD}-S{hhmmss}-E{hhmmss}.{orbit number}.V06A.HDF5

- ENV/V06/{YYYY}/2A-ENV.GPM.DPR.V8-20180723.{YYYYMMDD}-S{hhmmss}-E{hhmmss}.{orbit number}.V06A.HDF5

- CSH/V05/{YYYY}/2B.GPM.DPRGMI.2HCSHv3-1.{YYYYMMDD}-S{hhmmss}-E{hhmmss}.{orbit number}.V05A.HDF5

- SLH/V06/{YYYY}/2A.GPM.DPR.GPM-SLH.{YYYYMMDD}-S{hhmmss}-E{hhmmss}.{orbit number}.{V06A or V06B}.HDF5

需要注意：

- 只有 /data00 下有 CSH 产品，/data04 下没有。

- /data00 目录下的 SLH 产品版本是 V06A，而 /data04 下是 V06B。

- 缺失几十个 ENV 文件,数个 SLH 文件.

- 有些目录下除了 HDF5 文件，还可能还有一些 txt 或 PDF 文件。

显然 DPR 产品的路径组织和其它产品不同，以后也许需要修改。

# 时间覆盖范围.

/data00/0/GPM 目录下：

- DPR: from 2014-03-08 (000144) to 2019-05-30 (029841)

- ENV: from 2014-03-08 (000144) to 2019-06-01 (029868)

- CSH: from 2014-03-08 (000144) to 2018-09-30 (026078)

- SLH: from 2014-03-08 (000144) to 2019-06-17 (030122)

在不考虑 CSH 的情况下，上面产品的公共时间范围为：

from 2014-03-08 (000144) to 2019-05-30 (029841)

/data04/0/gpm 目录下的 DPR、ENV 和 SLH 三种产品的公共时间范围为：

from 2019-05-29 (029827) to 2020-12-31 (038881)

如果想获取时间上连续的记录，有以下两种建议：

- 在 029827 到 029841 之间选取一个轨道号来分隔两个目录下的数据。

- 2019-05-30 当天及以后的数据选自 /data04/gpm，而这天以前的数据都选自 /data00/GPM。
