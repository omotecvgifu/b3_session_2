#!/bin/sh

#python b3_session_cnn.py name=pyfile seed=0 epochs=20 transform.pattern=4 dataloader.batch_size=512 model.enc_dim=256 optimizer.lr=1.0e-4
# python b3_session_cnn.py name=test3 seed=0 epochs=1
# python b3_session_cnn.py name=test4 seed=42 epochs=1

#実験開始
#transformのパターン 
python b3_session_cnn.py --multirun name=compare_transform seed=0,42,100 transform.pattern=a,b,c,d,e,ab,ac,ad,ae,bc,bd,be,cd,ce,de,abc,abd,abe,acd,ace,ade,bcd,bce,bde,cde,abcd,abce,abde,acde,bcde,abcde

