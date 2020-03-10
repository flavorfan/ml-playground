# 0 ml-playground

# 1 文档结构
logs            :  logs for tensorboard
ckpts           :  checkpoints 
outputs         :  save plot result
training_plot   : training loss and accuracy

## train_fan_autoencoder.py 参数

### 可视化/结果保存
-s number of samples to visualize when decoding
//-o path to output visualization file 
//-p path to output train loss plot file
-m model name 

### 模型结构
-d n_dim of layers
-c code_dim

### 训练参数
-e epochs
-b batch_size

### 案例

* 784 - [256,128] - 16 - [ 128, 256 ] - 784 
python train_fan_autoencoder.py  -s 10 -m fc_256_128_16 -d 256 128 -c 16

* 784 - [256,128] - 16 - [ 128, 256 ] - 784 
python train_fan_autoencoder.py  -s 10 -m fc_256_128_2 -d 256 128 -c 2 -e 25

* 784 - [1000,500,250] - 2 - [250,500,1000] - 784 
python train_fan_autoencoder.py  -s 10  -m fc_1000_500_250_2 -d 1000 500 250 -c 2 -e 100

* rnn auto-encoder
python train_fan_autoencoder.py  -t rnn -s 10  -m rnn_64_32_2 -d 64 32 -c 2 -e 100 -b 128

## restore_training.py 参数
python restore_training.py  -s 10  -m fc_1000_500_250_2 -d 1000 500 250 -c 2 -e 100

python restore_training.py  -s 10 -m fc_256_128_16 -d 256 128 -c 16 -e 20

## train_denoising_autoencoder.py 
python train_denoising_autoencoder.py  -t rnn -s 10  -m dn_rnn_64_32_16 -d 64 32 -c 16 -e 50 -b 128



# 参考
Autoencoders with Keras, TensorFlow, and Deep Learning
https://www.pyimagesearch.com/2020/02/17/autoencoders-with-keras-tensorflow-and-deep-learning/
已读，代码运行
有源码-工业化的封装代码，模型构建和训练脚本分离，有

Building Autoencoders in Keras
https://blog.keras.io/building-autoencoders-in-keras.html
读一遍，复现部分
使用keras api进行ae的实现，原理透彻，需要继续实现




Autoencoders — Guide and Code in TensorFlow 2.0
https://medium.com/red-buffer/autoencoders-guide-and-code-in-tensorflow-2-0-a4101571ce56

Deep Autoencoder in TensorFlow 2.0
https://www.deeplearning-academy.com/p/ai-wiki-deep-autoencoder

Anomaly Detection with Autoencoder in TensorFlow 2.0
https://www.deeplearning-academy.com/p/ai-wiki-anomaly-detection


