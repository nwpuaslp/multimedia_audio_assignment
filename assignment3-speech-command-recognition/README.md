# assignment3-speech-command-recognition

#### 作业内容：

**（a）学习神经网络的基础知识，掌握多层感知器（MLP）和全连接网络（FCNN）的基本原理。**

**（b）学习语音识别的流程，编写语音指令识别代码，并按照给定的训练集和测试集进行实验，或者指令识别准确率（错误率）。**

**（c）[选做] 学习卷积神经网络（CNN）和回归神经网络（RNN）的基础知识，尝试使用这些网络进行语音指令识别的实验，并和FCNN的结果进行对比。**

#### 说明：

##### 你需要提交以下内容：

+ 实验报告，按照报告模板撰写，并保存格式为audio_assignment3-学号-姓名-报告.docx。
+ 附件，包括补充完整后的代码。

将以上内容打包压缩成audio_assignment3-学号-姓名.zip 上交。

##### 提示：

+ (a)可以参考cs231n课程3、4、5节和李航《统计学习方法》MLP部分
+ 作业使用的数据集在raw_commands_data文件夹中，完整数据集共有30个指令词，考虑到同学们电脑的计算能力，我们仅使用6个指令词。按照testing_list.txt划分出测试集，整个处理过程在对应文件夹下的make_data.py中。
+ 无论是使用pytorch还是tensorflow，都应当首先使用对应文件夹下的make_dataset.py准备和划分数据集，命令如下
  ```shell
  python make_dataset.py --commands_fold dir1 --out_path dir2  # dir1-->数据集路径(../raw_commands_data)   dir2-->处理后的数据集存放路径(./dataset)
  ```
+ 代码中搭建模型和训练部分已空出，需要自己补充完整，关于需要补充部分的说明在代码中已用两行'#'标出
+ 需要补充的部分的输入和输出含义都已经在代码中做了说明，网络输入的各维度含义也作了说明，具体维度可以在代码中合适位置设置断点输出查看
+ 以下给出一些供参考的文章：
cs231n:https://www.bilibili.com/video/BV1Dx411n7UE
MLP基础知识讲解:https://blog.csdn.net/xierhacker/article/details/53282038
FCNN基础知识讲解:https://zhuanlan.zhihu.com/p/104576756
CNN基础知识讲解:https://blog.csdn.net/m0_37490039/article/details/79378143
RNN基础知识讲解:https://blog.csdn.net/zhaojc1995/article/details/80572098
用CNN做关键词检出:https://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf
用DNN做关键词检出:https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6854370

##### 注意事项：
+ 使用的pytorch版本为1.1.0，tensorflow版本为1.14.0，其余需要的库和版本在对应文件夹下的requirements.txt中
+ 供参考的识别准确度如下（训练20个epoch的结果）：

| accuracy | pytorch | tensorflow |
|:--------:|:-------:|:----------:|
| CNN      | 92.26%  | 86.84%     |
| DNN      | 63.48%  | 71.06%     |

