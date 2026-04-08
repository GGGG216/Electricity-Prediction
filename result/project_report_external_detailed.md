# 香港月度用电预测项目对外说明报告

## 1. 报告目的

这份报告不是写给项目开发者看的，而是写给第一次接触这个项目的外部读者看的。

它的目标是用尽量简单、完整、可追溯的方式说明四件事：

- 这个项目到底要解决什么问题
- 项目为什么从周度/小时级预测改成了月度预测
- 项目实际使用了哪些数据集，每个数据集包含什么内容
- 这些数据最终是怎样被组合成一个可以训练模型的数据集，并产出预测结果的

## 2. 项目一句话概括

这个项目使用香港官方公开数据，建立了一套“从原始数据收集、到月度特征构建、再到用电量预测和结果分析”的完整流程，用来预测香港下一个月的总用电量。

## 3. 最开始的想法，和为什么后来改了

项目最开始的设想是预测香港每周甚至每小时的用电曲线。

这个想法在方向上是合理的，因为城市用电确实会受到天气、客流、季节、建筑结构和人口活动的影响。

但是在真正收集香港公开数据之后，我们发现一个关键问题：

- 公开数据里，最稳定、最完整的目标数据是“月度总用电量”
- 而不是“全港小时级负荷曲线”

同时，很多解释变量本身也更适合月度任务，例如：

- 月度交通客流
- 月度公共运输统计
- 月度能源统计
- 年度人口预测
- 静态建筑 footprint

所以项目最终做了一个重要调整：

- 原设想：预测每周或每小时的负荷曲线
- 最终实现：预测香港下一个月的总用电量

这个调整让“目标数据”和“解释变量”在时间粒度上统一起来，也让整个项目更加严谨、可复现、适合课程展示。

## 4. 项目最终回答的问题是什么

项目现在回答的问题是：

- 如果我们已经知道香港这个月以前的用电情况
- 同时也知道天气、节假日、人口活动、交通和城市结构等信息
- 那么我们能不能预测香港下一个月的总用电量

换句话说，这不是一个纯时间序列项目，也不是一个纯城市统计项目，而是一个“把历史用电和外部城市特征结合起来做预测”的项目。

## 5. 项目数据结构总览

整个项目的数据分成四层。

### 5.1 raw 层

`raw` 层保存的是从官方来源直接下载下来的原始文件。

这层的作用是：

- 保留最原始的数据
- 方便以后重新清洗
- 避免因为清洗错误而丢失原始来源
- 让项目可追溯

### 5.2 silver 层

`silver` 层保存的是清洗后的中间数据表。

这层的特点是：

- 列名统一
- 日期格式统一
- 每个表只保留建模真正需要的字段
- 日线数据被汇总成月线
- 年线数据被整理为适合月度建模的形式

### 5.3 model 层

`model` 层保存的是模型训练直接使用的表。

最核心的文件是：

- `city_monthly_train_ready.csv`

这张表的含义是：

- 每一行代表一个月份
- 每一列代表这个月份已经可以观察到的一个特征
- 目标列代表“下一个月的总用电量”

### 5.4 result 层

`result` 层保存的是最终给人阅读和展示的内容。

包括：

- 数据分析结果
- 模型评估结果
- 图表
- 错误处理说明
- 报告和展示稿

## 6. 详细数据集说明

下面按项目真正使用的数据表，逐个说明它们的来源、内容、时间范围、主要字段和作用。

## 6.1 weather_monthly.csv

### 文件定位

- 文件名：`data/silver/weather_monthly.csv`
- 行数：1622
- 列数：12
- 时间范围：1884-01 到 2026-02

### 数据来源

- 香港天文台 HKO 的日度天气历史数据

原始日度表包括：

- 日平均温度
- 日最高温度
- 日最低温度
- 日平均相对湿度
- 日降雨量

### 进入模型前的处理方式

项目把日度天气先整理成月度天气，再形成一个月一行的表。

### 主要字段

- `period_month`：月份
- `temp_mean_c_avg`：该月日平均温度的平均值
- `temp_max_c_avg`：该月日最高温度的平均值
- `temp_max_c_peak`：该月出现过的最高温度
- `temp_min_c_avg`：该月日最低温度的平均值
- `temp_min_c_low`：该月出现过的最低温度
- `temp_range_c_avg`：该月平均日温差
- `rh_mean_pct_avg`：该月平均相对湿度
- `rainfall_mm_sum`：该月总降雨量
- `rainfall_day_count`：该月下雨天数
- `hot_day_count`：该月炎热天数
- `cool_day_count`：该月较凉天数

### 它的作用

天气是解释用电波动最重要的变量之一。

例如：

- 天气更热时，空调负荷通常更高
- 天气更潮湿时，建筑耗能模式可能改变
- 降雨和温差变化也会影响商业和居民活动

## 6.2 calendar_monthly.csv

### 文件定位

- 文件名：`data/silver/calendar_monthly.csv`
- 行数：1716
- 列数：5
- 时间范围：1884-01 到 2026-12

### 数据来源

- 香港政府公开节假日数据
- 同时结合日历规则生成周末和工作日统计

### 主要字段

- `period_month`：月份
- `days_in_month`：该月总天数
- `weekend_days`：该月周末天数
- `public_holiday_days`：该月公众假期天数
- `business_days`：该月工作日天数

### 它的作用

日历变量可以解释“同样的天气下，为什么某些月份的用电结构还是不一样”。

例如：

- 节假日多的月份，商业和通勤活动可能下降
- 工作日更多的月份，办公楼和交通相关活动更强

## 6.3 immigration_city_monthly.csv

### 文件定位

- 文件名：`data/silver/immigration_city_monthly.csv`
- 行数：63
- 列数：11
- 时间范围：2021-01 到 2026-03

### 数据来源

- 香港入境事务处的日度旅客流量数据

原始数据按口岸和方向记录。
项目在清洗时把它汇总成城市层面的月度总量。

### 主要字段

- `period_month`：月份
- `immigration_arrival_hk_residents`：香港居民入境人数
- `immigration_departure_hk_residents`：香港居民离境人数
- `immigration_arrival_mainland_visitors`：内地访客入境人数
- `immigration_departure_mainland_visitors`：内地访客离境人数
- `immigration_arrival_other_visitors`：其他访客入境人数
- `immigration_departure_other_visitors`：其他访客离境人数
- `immigration_arrival_total_passengers`：总入境人数
- `immigration_departure_total_passengers`：总离境人数
- `immigration_total_cross_border_passengers`：总跨境旅客量
- `immigration_net_arrivals_total_passengers`：净流入旅客量

### 它的作用

这组数据是“城市活动强度”的代理变量。

如果一个月旅客和跨境人流更多，通常意味着：

- 商业活动更活跃
- 交通活动更密集
- 零售、办公、服务类负荷更强

## 6.4 transport_public_total_monthly.csv

### 文件定位

- 文件名：`data/silver/transport_public_total_monthly.csv`
- 行数：157
- 列数：2
- 时间范围：2013-01 到 2026-01

### 数据来源

- 香港运输署月度交通与运输摘要

### 主要字段

- `period_month`：月份
- `public_transport_total_avg_daily_pax`：该月平均每日公共交通总客运量

### 它的作用

这是另一个反映“城市整体活跃程度”的核心变量。

它比单一模式更概括，可以快速表示：

- 这个月整体通勤是否更强
- 城市活动强度是否上升

## 6.5 transport_public_mode_wide_monthly.csv

### 文件定位

- 文件名：`data/silver/transport_public_mode_wide_monthly.csv`
- 行数：157
- 列数：8
- 时间范围：2013-01 到 2026-01

### 数据来源

- 香港运输署月度公共交通客运量表

### 主要字段

- `period_month`
- `public_transport_ferries_avg_daily_pax`
- `public_transport_fran_bus_avg_daily_pax`
- `public_transport_lrb_avg_daily_pax`
- `public_transport_plb_avg_daily_pax`
- `public_transport_rs_avg_daily_pax`
- `public_transport_railways_avg_daily_pax`
- `public_transport_tax_avg_daily_pax`

### 它的作用

相比总量表，这张表更细，能让模型区分不同交通模式的变化。

例如：

- 铁路流量更能反映核心通勤
- 渡轮与特定区域活动相关
- 巴士和小巴更能反映地面交通结构

## 6.6 transport_cross_harbour_monthly.csv

### 文件定位

- 文件名：`data/silver/transport_cross_harbour_monthly.csv`
- 行数：157
- 列数：4
- 时间范围：2013-01 到 2026-01

### 数据来源

- 香港运输署的过海交通月度统计

### 主要字段

- `period_month`
- `cross_harbour_av_pax`
- `cross_harbour_db_pax`
- `cross_harbour_total_pax`

### 它的作用

过海交通是香港非常有代表性的城市活动指标。

它能帮助模型识别：

- 港岛与九龙、新界之间的活动强度变化
- 通勤和跨区出行的变化

## 6.7 city_energy_monthly.csv

### 文件定位

- 文件名：`data/silver/city_energy_monthly.csv`
- 行数：566
- 列数：8
- 时间范围：1979-01 到 2026-02

### 数据来源

- 香港政府统计处 C&SD 月度能源统计

这是整个项目最关键的数据表之一，因为它里面包含了真正要预测的目标。

### 主要字段

- `period_month`
- `electricity_commercial_tj`：商业用电量
- `electricity_domestic_tj`：住宅用电量
- `electricity_industrial_tj`：工业用电量
- `electricity_street_lighting_tj`：街灯用电量
- `electricity_total_tj`：总用电量
- `gas_total_tj`：总燃气用量
- `electricity_export_to_mainland_tj`：输往内地电量

### 它的作用

这张表同时承担两个角色：

第一，它提供项目的预测目标：

- `electricity_total_tj`

第二，它也提供额外解释信息：

- 商业、住宅、工业、街灯等分项用电量
- 燃气总量作为能源活动背景变量

## 6.8 hkelectric_re_generation_monthly.csv

### 文件定位

- 文件名：`data/silver/hkelectric_re_generation_monthly.csv`
- 行数：61
- 列数：4
- 时间范围：2021-01 到 2026-01

### 数据来源

- 港灯发布的月度可再生能源发电量数据

### 主要字段

- `period_month`
- `hkelectric_re_generation_solar_photovoltaic`
- `hkelectric_re_generation_wind_power`
- `hkelectric_re_generation_grand_total`

### 它的作用

这张表不是用来直接代表总负荷，而是用来补充能源结构背景。

它可以帮助模型理解：

- 某些月份能源系统背景是否发生变化
- 可再生发电是否反映了季节性条件变化

## 6.9 city_population_yearly.csv

### 文件定位

- 文件名：`data/silver/city_population_yearly.csv`
- 行数：11
- 列数：4
- 时间范围：2019 到 2029

### 数据来源

- 香港人口预测相关公开空间数据

### 主要字段

- `year`
- `population`
- `district_area_m2`
- `population_density_per_km2`

### 它的作用

这是一组“慢变化”的城市结构变量。

它们不是每天都变，但决定了香港整体的负荷底盘：

- 人口越多，整体生活和商业活动基础越大
- 人口密度越高，空间利用和建筑运行结构也会不同

## 6.10 ev_city_static.csv

### 文件定位

- 文件名：`data/silver/ev_city_static.csv`
- 行数：1
- 列数：7
- 类型：静态表

### 数据来源

- 香港公共电动车充电设施公开数据

### 主要字段

- `ev_standard_chargers`
- `ev_medium_iec62196_chargers`
- `ev_quick_chademo_chargers`
- `ev_quick_ccs_combo_chargers`
- `ev_quick_iec62196_chargers`
- `ev_quick_gbt_dc_chargers`
- `ev_total_chargers`

### 它的作用

这组数据代表香港当前电动车充电基础设施规模。

虽然它不是月度动态变化数据，但它提供了一个重要的城市结构背景：

- 电力使用方式正在变化
- 充电网络规模可以反映电力需求结构升级趋势

## 6.11 building_footprints_city_static.csv

### 文件定位

- 文件名：`data/silver/building_footprints_city_static.csv`
- 行数：1
- 列数：6
- 类型：静态表

### 数据来源

- 香港建筑 footprint 空间数据

### 主要字段

- `building_count`
- `building_block_type_count`
- `footprint_area_m2_sum`
- `footprint_area_m2_mean`
- `building_height_m_mean`
- `completed_building_count`

### 它的作用

这是一组描述香港整体建筑存量结构的静态变量。

它代表的是城市电力需求的空间和物理基础：

- 建筑数量
- 建筑占地面积
- 平均建筑高度
- 已建成建筑规模

这些变量不会解释短期波动，但它们能解释“为什么香港的负荷底盘是现在这个规模”。

## 7. 模型训练前，数据是怎样被统一到一起的

上面这些表来自不同部门、不同格式、不同时间粒度。

如果直接拿来训练，是不能用的。

所以项目做了统一处理。

### 7.1 时间统一

- 日线数据先按月汇总
- 月线数据直接保留
- 年线数据按年份对齐到每个月
- 静态数据复制到所有月份

### 7.2 字段统一

不同来源的数据字段名称完全不一样。

所以项目把字段重新命名成统一风格，例如：

- 所有月度时间字段统一为 `period_month`
- 温度和交通字段都改成有意义、可解释的英文名

### 7.3 目标列构造

模型最终要预测的是：

- `target_electricity_total_t_plus_1m`

它表示：

- 对于当前月份 t
- 目标不是这个月自己的总用电量
- 而是下一个月 `t+1` 的总用电量

这样才是真正的预测任务，而不是事后拟合。

### 7.4 历史特征构造

项目给模型加入了一组历史用电特征，包括：

- `electricity_lag_1m`
- `electricity_lag_2m`
- `electricity_lag_3m`
- `electricity_lag_6m`
- `electricity_lag_12m`
- 3、6、12 个月滚动均值
- 同比变化

这些特征很重要，因为用电序列本身存在强烈的季节性和惯性。

## 8. 最终训练数据集长什么样

模型最终使用的是：

- `data/model/city_monthly_train_ready.csv`

它的规模是：

- 157 行
- 89 列
- 时间范围：2013-01 到 2026-01

为什么从 2013 年开始？

因为：

- 公共交通月度数据从 2013 年起比较完整
- 从这个时间点开始，主要外生变量的覆盖更稳定

也就是说，虽然某些原始表更早就有数据，但为了保证训练表质量，项目选择了“交集更稳定”的时间窗口。

## 9. 数据处理时遇到的主要问题

项目不是把表简单拼起来就结束了，而是在数据工程阶段解决了一些非常现实的问题。

### 9.1 不同数据源的时间粒度不同

问题：

- 有的数据是日线
- 有的是月线
- 有的是年线

处理：

- 全部对齐到月度任务

### 9.2 不同数据源的时间起点不同

问题：

- 用电目标数据很早就有
- 但客流、交通、人口、可再生发电的起点较晚

处理：

- 使用可用性标记
- 并选择 2013 年以后作为主训练窗口

### 9.3 文件格式与后缀不一致

问题：

- 港灯有一份文件文件名像 CSV
- 但实际内容是 Excel 工作簿

处理：

- 在读取前先检查文件签名，再选择正确读取方式

### 9.4 统计表 period 字段混合了不同时间单位

问题：

- 同一个统计系统中，period 可能表示年、季度或月

处理：

- 显式解析 period
- 仅保留合法月度记录作为月度目标表

### 9.5 年度人口数据无法直接按月使用

问题：

- 人口预测是年度表
- 但模型是月度任务

处理：

- 先按年份映射到每个月
- 如果月份超出可用区间，则裁剪到最近可用年份

## 10. 模型部分是怎么做的

项目没有一开始就追求最复杂的模型，而是先搭建了几组能清楚解释结果的 baseline。

### 10.1 主模型 baseline

一共训练了三种主模型：

- Seasonal naive
- Ridge regression
- Random Forest

其中：

- Seasonal naive 的意思是“直接用去年同一个月作为预测”
- Ridge 是带正则化的线性模型
- Random Forest 是非线性树模型

### 10.2 解释性回归模型

为了让外部读者更容易理解“哪些变量有帮助”，项目又补做了两组线性回归：

- Simple linear regression
- Multiple linear regression

前者只使用一个最强季节性变量：

- `electricity_lag_12m`

后者使用一组更完整的解释变量，包括：

- 历史用电
- 天气
- 节假日
- 客流
- 交通
- 燃气
- 季节编码

## 11. 结果是什么

### 11.1 主模型 baseline 结果

- Seasonal naive：MAE 1210.57，RMSE 1396.17，R2 0.7054
- Ridge：MAE 928.35，RMSE 1375.55，R2 0.7141
- Random Forest：MAE 456.93，RMSE 609.79，R2 0.9438

这说明：

- 去年同月这个简单规则已经能抓到一部分规律
- 线性模型有一定提升
- Random Forest 表现最好，说明当前任务存在明显的非线性关系

### 11.2 回归验证结果

- Simple regression：MAE 1155.41，RMSE 1346.22，R2 0.7261
- Multiple regression：MAE 784.25，RMSE 944.58，R2 0.8652

这说明：

- 只看历史季节性已经能做出一个还不错的预测
- 但把天气、节假日、客流、交通和燃气等变量加进去以后，效果明显更好

## 12. 这些指标该怎么理解

为了方便非技术读者理解，这里简单解释三个最常见的指标。

### MAE

MAE 表示平均绝对误差。

它可以理解成：

- 模型平均每个月会错多少

数值越小越好。

### RMSE

RMSE 也表示误差，但会对大误差惩罚更重。

如果一个模型 RMSE 很大，通常说明它在某些月份会出现明显偏差。

### R2

R2 可以理解成模型对目标变化的解释程度。

它越接近 1，说明模型越能解释数据里的波动规律。

## 13. 结果告诉了我们什么

这个项目最重要的结论有四点。

第一，香港月度总用电量是可以预测的。

第二，历史用电确实很重要，因为季节性非常明显。

第三，仅靠历史用电还不够，天气、节假日、客流和交通等动态变量提供了额外信息。

第四，从公开数据现实条件来看，月度任务比最初设想的小时级任务更可靠，也更适合教学项目。

## 14. 项目的边界和限制

这份报告也需要坦诚说明项目当前的边界。

### 14.1 这是“香港整体月度预测”，不是分区预测

因为当前最稳定的公开目标数据是香港整体月度总量，而不是分区月度负荷。

### 14.2 这是“总量预测”，不是小时负荷曲线预测

因为公开数据没有提供稳定、持续的全港小时级目标标签。

### 14.3 部分结构特征是静态或慢变化的

例如：

- EV charger
- 建筑 footprint
- 年度人口

这些变量更适合解释长期底盘，而不一定能解释每个月的短期波动。

## 15. 项目已经交付了什么

到现在为止，这个项目已经形成了一个完整的、可运行的交付物。

包括：

- 自动数据收集流程
- 月度清洗后的 silver 数据层
- train-ready 训练数据集
- baseline 训练和评估代码
- 解释性回归验证
- 错误处理说明
- 图表结果包
- 外部读者可读的报告

## 16. 如果未来继续做，可以怎么扩展

如果后续要继续做，可以考虑三个方向。

第一，加入更多经济活动数据，例如旅游、消费、就业和商业景气指标。

第二，尝试更多模型，例如 XGBoost、LightGBM 或专门的时间序列模型。

第三，如果未来能拿到更细粒度的官方或合作数据，再把项目扩展回周度甚至小时级预测。

## 17. 最后总结

这不是一个只停留在概念层面的项目。

它已经完成了：

- 数据源识别
- 原始数据采集
- 月度特征工程
- 训练表构造
- 模型训练
- 结果评估
- 报告输出

因此，这个项目可以被看成一套完整的“香港月度用电预测”工作流，而不是单一的一份代码或图表。
