# SiamMask_master
# 项目介绍：
* 运动目标跟踪一直以来都是一项具有挑战性的 工作, 也是研究的热点方向. 现阶段, 随着硬件设施 的不断完善和人工智能技术的快速发展, 运动目标 跟踪技术越来越重要. 目标跟踪在现实生活中有很 多应用, 包括交通视频监控、运动员比赛分析、智能人机交互 、跟踪系统的设计 等. 由于在目标跟踪中存在形态变化、图像分辨率低、背景复杂等情 况, 因此研究出一个性能优良的跟踪器势在必行。

* 早期的目标跟踪算法主要是根据目标建模或者 对目标特征进行跟踪, 主要的方法有:
* 1) 基于目标模型建模的方法: 通过对目标外观模型进行建模, 然 后在之后的帧中找到目标. 例如, 区域匹配、特征点 跟踪、基于主动轮廓的跟踪算法、光流法等. 最常用 的是特征匹配法, 首先提取目标特征, 然后在后续的 帧中找到最相似的特征进行目标定位, 常用的特征 有: SIFT 特征、SURF 特征、Harris 角点等.
* 2) 基于搜索的方法: 随着研究的深入, 人们发现基 于目标模型建模的方法对整张图片进行处理, 实 时性差. 人们将预测算法加入跟踪中, 在预测值附近 进行目标搜索, 减少了搜索的范围. 常见一类的预测 算法有 Kalman滤波、粒子滤波方法. 另一种减小搜索范围的方法是内核方法: 运用最速下降法的原理, 向梯度下降方向对目标模板逐步迭代, 直到迭代到最优位置。

## 使用的模型是：
![](image/模型.png)

## 网络的工作流如下所示:
![](image/网络的工作流.png)

## 追踪效果如下所示：
![](image/追踪效果.png)
