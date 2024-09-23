# 天池多模态线下比赛

## 环境配置

​	本队伍在这次比赛中使用了大约8张A40，4张A100，16张A800，并且含有不能联网的机器，所以，服务器数据会有一些乱，比较难管理，但是对于每次处理的数据集都使用[线上赛](https://c2e0u86auj.feishu.cn/docx/T7ZUdAqRsoca4exLYpDcSmmbnOH?from=from_copylink )和[线下赛](https://c2e0u86auj.feishu.cn/docx/BXJvdySBxosyexxoNUTcWpa2nnh?from=from_copylink)。机器之间的环境使用conda打包迁移，如需提供，可联系我们。

## 最优成绩思路介绍

​	我们的最开始思路是根据论文里提供的图片做一个多算子的叠加，得到最优数据。论文策略如下：

![](https://p.ipic.vip/umv9g3.png)

​	然后我们对整个数据集进行分析，得到这些结果

![](https://p.ipic.vip/6lqn8r.png)

![](https://p.ipic.vip/dvy0ha.png)

![](https://p.ipic.vip/bs15tg.png)

​	可以看到，其实400K的数据集，结果已经满足*nsfw*，*action*的要求了。所以，我们使用*phrase_grounding_recall_filter*过滤出一半的数据，进行训练，结果并不好。然后，我们使用*image_text_matching_filter*过滤一半的数据，最后得到123037条数据，我们将其复制一份进行训练，得到较好的结果。

​	接下来，为了成绩可以继续提升，我们使用*image_text_similarity_filter*过滤出分数在0.01-0.28之内的数据得到5.9K数据，使用*image_captioning_mapper*生成这些数据的caption，再使用image_text_similarity_filter大于0.28的条件进行过滤，得到4K数据，然后将其替换进去。


## 训练与推理

用官方提供的训练脚本进行训练和评测



## 总结

​	我们的详细比赛记录，以及各个算子组合产生的数据集都记录在[线上赛](https://c2e0u86auj.feishu.cn/docx/T7ZUdAqRsoca4exLYpDcSmmbnOH?from=from_copylink )和[线下赛](https://c2e0u86auj.feishu.cn/docx/BXJvdySBxosyexxoNUTcWpa2nnh?from=from_copylink)中，强烈推荐查看。

​	