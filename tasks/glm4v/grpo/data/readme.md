# 数据处理流程

数据流程略复杂，特此写个文档梳理一下。

TODO：把此数据处理流程标准化

## 1、 数据格式转换 and write image to lmdb
* gcore 的 dataset 对数据格式有特定要求
* 对图片采用了lmdb 存储， 方便直接读取
## 2、filter sample if necessary
* 根据训练配置，过滤掉超长样本
## 3、build data meta
* 文件列表等元信息
## 4、训练之前拉起 lmdb 服务， 响应dataloader 图片请求 
* 训练时，图片从 lmdb 读取