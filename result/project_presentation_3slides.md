# 香港月度用电预测项目 3 页展示稿

## 第 1 页：数据集结构设计

- 项目从“周度/小时级负荷曲线预测”调整为“香港下一个月总用电量预测”
- 原因是公开数据更适合月度任务，数据来源也更稳定
- 数据分成四层：raw、silver、model、result
- raw 保存官方原始文件
- silver 保存清洗后的月度表
- model 保存可直接训练的样本表
- result 保存图表、结论和报告
- 当前核心训练表是 `city_monthly_train_ready.csv`
- 训练表规模为 157 行、96 列，时间范围是 2013-01 到 2026-01
- 建筑类特征不再使用静态 footprint，而是改用月度建筑活动数据

## 第 2 页：数据收集流程

- 目标数据来自 C&SD 的香港月度用电量表
- 特征数据来自香港天文台、节假日开放数据、入境处、运输署、港灯、人口预测，以及 Buildings Department 的月度建筑活动数据
- 日线数据按月聚合
- 月线数据直接使用
- 年线数据按年份对齐到每个月
- 静态背景数据保留在 silver 层，不再强行进入训练
- 再加入 lag 1m、2m、3m、6m、12m、滚动平均和同比变化
- 新增建筑活动特征包括开工同意书、occupation permits、落成楼面面积和新增住宅单位
- 最后得到 train-ready 月度训练表

## 第 3 页：训练与结果

- 主模型 baseline 包括 seasonal naive、ridge regression 和 random forest
- Random Forest 表现最好：MAE 454.19，RMSE 614.60，R2 0.9429
- 另外做了 simple linear regression 和 multiple linear regression
- Simple regression 只用 `electricity_lag_12m`
- Multiple regression 加入天气、节假日、客流、交通、燃气、建筑活动和季节特征
- Simple regression 结果：MAE 1155.41，RMSE 1346.22，R2 0.7261
- Multiple regression 结果：MAE 684.98，RMSE 857.50，R2 0.8889
- 结论：历史用电很重要，建筑活动等动态外生变量也能明显提升预测效果
