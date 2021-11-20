# seq2seq-fairseq

本项目是fairseq的简化版，重写了RNN和Transfomer的机器翻译流程，

1. 数据集：
   <a href='https://drive.google.com/open?id=1EX8eE5YWBxCaohBO8Fh4e2j3b9C2bTVQ'>Google Drive下载</a> 或者 [百度网盘链接](链接：https://pan.baidu.com/s/17wQqIns_dyPVgNGYo191xg )

2. 提取码：3lx5 
   --来自百度网盘超级会员V1的分享

   数据集划分：分成两个个部分。训练集：516万；验证集：3.9万

   从训练集里抽取50万做训练样本、5000做验证集；

   从验证集里抽取4000做测试

3. 相关库

   除了常见的深度学习的相关库外，还需要安装fairseq

   本任务使用的是下面的版本，如果你的电脑python或torch版本是其他的，可以去官网查看对应版本，并下载，链接在这：[fairseq](https://pypi.org/project/fairseq/)

   

   ```python
   PyTorch version >= 1.4.0
   Python version >= 3.6
   For training new models, you will also NVIDIA GPU
   # 安装方式
   1. 
   git clone https://github.com/pytorch/fairseq
   cd fairseq
   pip install --editable ./
   2.
   pip install fairseq==0.10.0
   
   3. 如果2报错，可以使用下面的镜像网站下载
   pip install fairseq==0.10.2 -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
   ```

4. 本项目是把原始数据集下载后放在TranslationData/raw_data目录下，当然你也可以换成自己的目录（当然也需要在main.py中的data_dir参数设置，做相应的修改），然后在终端执行如下操作：

   

   1）对数据进行清洗、切分训练、验证、测试集、子词切分等操作，最后得到数据是断词后的数据（train.en 、 train.zh 、 valid.en  、valid.zh、 test.en、 test.zh）

   ```python
   python -m data_process
   ```

   2）对处理过的数据进行转换为二进制数据资料（这里的二进制资料不是通常认为的011010这样的只有0和1的数据，它其实就是一个词表，每个切分后的子词都有一个id）

   'en' 、'zh'，是1）中数据集的后缀，三个pref是前缀，destdir是经过处理后的数据路径

   ```python
   python -m fairseq_cli.preprocess --source-lang 'en' --target-lang 'zh' --trainpref './TranslationData/raw_data/train' --validpr
   ef './TranslationData/raw_data/valid' --testpref './TranslationData/raw_data/test' --destdir './TranslationData/data_bin' --joined-dictionary --workers 2
   ```

   3）运行模型

   RNN

   ```python
   python main --model RNN
   ```

   Trabsformer

   ```python
   python main --model Transformer
   ```

   

* 备注

  ```
  如果报错：没有TranslationConfig类
  
  --如果导入的fairseq中没有这个类，需要去git中找main版本,
  --把translation的文件复制到本地translation,
  --并在TranslationTask的构造函数__init__()下添加self.cfg = cfg
  ```

  [链接](https://github.com/pytorch/fairseq)
