MiniGrid-twoarmy-17x17-v2环境是用于最终测试，另外v0是最简单的，只有中间有巡逻；v5是在v0的基础上，在下面的空间(room1)里增加了固定位置的障碍物；v6是在v5的基础上，在下面的空间里固定位置的障碍物改为随机位置的障碍物；v4是在v6的基础上，在上面的空间（room2)里增加了巡逻单位；v2是在v4基础上，在目标位置左侧或下侧有规律地放置障碍物。
数据生成：datacol_predictor.py
阶段1：训练train_encoder_decoder.py 
阶段2：fixed encoder and decoder, train predictor
阶段3：fixed encoder, decoder and predictor, train baseline1 or ours