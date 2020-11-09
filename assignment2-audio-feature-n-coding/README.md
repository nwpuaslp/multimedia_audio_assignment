# assignment2-audio-feature-n-coding

#### 作业内容：

**(1)了解语音处理的一般流程，计算一段语音的短时能量、短时过零率和短时自相关函数，将这些特征取值绘图，并观察不同语音单元在这些特征取值上的差异。**

**(2)编写一段程序，利用mu-law变换的公式，将16位Linear PCM格式存储的音频文件转换为8位non-linear PCM格式，并比较转换前后音频声音质量和文件大小。**



#### 说明：

##### 你需要提交以下内容：

+ 完成(1)和(2)的python代码文件
+ (1)中短时能量、短时过零率和短时自相关函数绘制出的图片。短时能量和短时过零率可以绘制整段语音，短时自相关函数你可以根据清音和浊音分别选择一帧绘图。
+ (2)中转换后的8位non-linear PCM格式文件，用.wav格式保存。
+ 实验报告，按照报告模板撰写，并保存格式为audio_assignment2-学号-姓名-报告.docx。

将以上内容打包压缩成audio_assignment2-学号-姓名.zip 上交。

##### 注意事项：

+ 可以用librosa等python包来读取、生成wav文件。

+ 要想更细致的观察音频，可以使用Adobe Audition等工具。

+ (1)(2)所使用的音频文件需要是你在assignment1中的录音，且需要保证这段录音采样率8k、16位Linear PCM格式存储。可以使用SoX等工具查看你的音频文件是否满足要求，如果不满足需要先转换成符合要求的格式。

+ 仅仅依赖mulaw公式比较难产生质量高的8位PCM，在自己手写转换代码后，可以使用ffmpeg工具，执行以下命令来获取高质量的8位non-linear PCM音频。

  ```shell
  ffmpeg -i src.wav -acodec pcm_mulaw out.wav # src.wav-->yourinput out.wav-->youroutput
  ```

  