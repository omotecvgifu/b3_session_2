#!/bin/sh

#python b3_session_cnn.py name=pyfile seed=0 epochs=20 transform.pattern=4 dataloader.batch_size=512 model.enc_dim=256 optimizer.lr=1.0e-4
# python b3_session_cnn.py name=test3 seed=0 epochs=1
# python b3_session_cnn.py name=test4 seed=42 epochs=1

#実験開始
#transformのパターン 
#python b3_session_cnn.py --multirun name=compare_transform3 seed=0,42,100 transform.pattern=null,a,b,c,d,ab,ac,ad,bc,bd,cd,abc,abd,acd,bcd,abcd
#モデルのパターン
python b3_session_cnn.py --multirun name=compare_model transform.pattern=null,abcd seed=0,42,100 model.type=0,1 model.enc_dim=64,128,256
python ./tests/gmail.py
python b3_session_cnn.py --multirun name=compare_model transform.pattern=null,abcd seed=0,42,100 model.type=2 model.enc_dim=64,128
python ./tests/gmail.py
#
#python b3_session_cnn.py --multirun name=compare_model transform.pattern=null,abcd seed=0,42,100 model.type=2 model.enc_dim=64,128 

