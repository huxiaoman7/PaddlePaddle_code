# 记录一些未解决的问题
## 2017.12.11
------
- 1Q:在tensorflow里以运行单机多卡的方式运行时，tf会一开始占用全部内存，但是实际上只用了第一个内存运行程序（内部机制？），在PaddlePaddle中如何进行单机多卡的方式？我发现我以前用PaddlePaddle跑的程序可能都是单机单卡的方式，并没有把8个Tesla K80全部用上，下班仔细研究下
- 2Q:在tf的models下面有个slim（不是contrib模块下的slim）, deplotyment模块下的model_deploy.py支持分布式训练吗？我目前知道是支持单机多卡，但是注释里写的是支持多个机器并行训练，但是不知道是否支持集群的分布式训练（多机多卡）？在代码里看到有定义ps和worker，当num_gpu>1时，会把不同的worker指定到不同的gpu上，但是这也只是在一台机器上啊 = =并没有看到定义ClusterSpec集群，所以很疑惑，这个code是用别的方式“连接”不同机器还是说不支持集群方式的分布式训练?
------
- 2Q:终于弄明白了，曾经有人提过issue：[https://github.com/tensorflow/tensorflow/issues/14232](https://github.com/tensorflow/tensorflow/issues/14232),在issue里说道，目前model_deploy.py
