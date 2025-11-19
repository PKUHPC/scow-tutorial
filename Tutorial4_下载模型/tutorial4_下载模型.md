# Tutorial4: 下载模型

* 集群类型：超算平台
* 所需镜像：无
* 所需模型：无
* 所需数据集：无
* 所需资源：无
* 目标：本节旨在使用超算平台展示如何下载大模型 [Qwen3-4B] (https://modelscope.cn/models/Qwen/Qwen3-4B-Instruct-2507) 。

## 1、使用超算平台下载模型

1.1.1 登录[SCOW平台](scow.pku.edu.cn)，选取超算平台

![alt text](image.png)

1.1.2 点击登录集群，选择你要使用的集群（未名二号、未名一号、未名生科一号），选择data节点，点击打开

![alt text](image-1.png)
![alt text](image-2.png)

1.1.3 拷贝命令 `pwd` 粘贴到界面，并按 回车键，查看当前路径

![alt text](image-3.png)

1.1.4 拷贝命令 `mkdir model` 粘贴到界面，并按 回车键，这样就在当前目录下新创建了一个名为 model 的目录，下载的模型都可以统一放在这个目录下面

1.1.5 拷贝命令 `cd model` 粘贴到界面，并按 回车键，这样就进入到刚新创建的名为 model 的目录里

![alt text](image-4.png)

1.1.6 拷贝命令 `pip install modelscope` 粘贴到界面，并按 回车键。
这里是安装了modelscope工具，此工具由模型下载的镜像网站提供

1.1.7 拷贝命令 modelscope download --model Qwen/Qwen3-4B-Instruct-2507 --local_dir ./Qwen/Qwen/Qwen3-4B-Instruct-2507 粘贴到界面，并按 回车键。
这里是通过刚安装的modelscope这个工具去镜像网站下载模型 Qwen3-4B-Instruct-2507

```bash
pip install modelscope
modelscope download --model Qwen/Qwen3-4B-Instruct-2507 --local_dir ./Qwen/Qwen/Qwen3-4B-Instruct-2507
```
