# scow-tutorial
本教程介绍如何在基于 [SCOW](https://www.pkuscow.com/) 的英伟达集群上申请计算资源并运行各类计算任务。

[教程入口](https://github.com/PKUHPC/scow-tutorial/blob/main/tutorial_scow.md)

## 贡献指南
欢迎贡献新教程或改进现有教程。请参考现有教程的格式，编写教程后提交pr，仓库管理员会不定期进行合并。
1. 每篇教程开头需要注明：
* 集群类型：使用中文名称，超算平台 ｜ 智算平台
* 所需镜像
* 所需模型
* 所需数据集
* 所需硬件资源：单机单卡 ｜ 单机多卡 ｜ 多机多卡
* 目标
* 参考文档
2. 教程中涉及环境依赖安装的，必须明确具体版本.
例如应当使用`pip3 install torch==2.5.1`，不应当使用`pip3 install torch`；
3. 截图应当尽可能包含网址等完整的页面，便于读者理解当前步骤要在哪个页面操作；
4. 尽可能使用图片进行步骤指引，保持每个必要的步骤，对于必要的步骤进行解释说明，对于必要的步骤执行后的结果进行验证说明，便于读者理解必要步骤所起的作用；
5. 运行结果必须配上相应的截图，以及使用箭头或者高亮标注出结果中的关键词，例如，successfully, done, complete等，便于读者理解运行结果是否符合期望结果；
6. 模型下载使用超算集群（HPC）中的shell模式进行，模型的训练、微调、推理等使用智算集群（AI）中启动交互式或命令行模式进行；
7. 数据集的创建和管理使用AI集群进行；
8. 教程中涉及到的命令、代码，都能进行复制，在粘贴后能直接执行或运行；
9. 在教程规范使用环境变量：
* $SCOW_AI_MODEL_PATH：模型存放的统一路径
* $SCOW_AI_DATASET_PATH：数据集存放的统一路径
* $SCOW_AI_ALGORITHM_PATH：算法存放的统一路径
10. 基础镜像，在没有特别说明的情况下，都使用基础镜像
* 将 `app-store-images.pku.edu.cn/nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04` 作为基础镜像，粘贴到 远程镜像地址框中
* 将 `app-store-images.pku.edu.cn/hiyouga/llamafactory:0.9.4` 作为llamafactory基础镜像，粘贴到 远程镜像地址框中
11. 每个教程都使用markdown格式，每个教程中使用的图片，统一放在教程目录下的assets文件夹中，与教程的markdown文件并列
## issue
如果遇到问题，欢迎提交issue。

