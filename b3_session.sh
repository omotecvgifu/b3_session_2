#!/bin/sh

#python b3_session_cnn.py name=pyfile seed=0 epochs=20 transform.pattern=4 dataloader.batch_size=512 model.enc_dim=256 optimizer.lr=1.0e-4
# python b3_session_cnn.py name=test3 seed=0 epochs=1
# python b3_session_cnn.py name=test4 seed=42 epochs=1

#実験開始
python b3_session_cnn.py -m name=test_compare_transform seed=0,42,100 epochs=0 transform.pattern=0,1,2,3,4,5 #transformのパターン


