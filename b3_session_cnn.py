##----ライブラリ
# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageFolder
# from torchvision.utils import make_grid

# モデル構造の表示
# from torchinfo import summary

# その他
import numpy as np
# import matplotlib.pyplot as plt
# import pathlib
from tqdm import tqdm
# from PIL import Image

#チューニング
from torch.utils.tensorboard import SummaryWriter
# from datetime import datetime
import random
import os
import datetime

#yml
from omegaconf import DictConfig, ListConfig,OmegaConf
import hydra

#mlflow
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME,MLFLOW_USER,MLFLOW_SOURCE_NAME

from mypkg import decorator,myExp


##--------------------



##--------------------実験を行うメインのクラス

#hydraを使用して実験設定を読み込み
@hydra.main(version_base=None, config_path="./", config_name="config")
def main2(cfg:DictConfig):
    main(cfg)

@decorator.error_gmail
def main(cfg: DictConfig):

    #========== 実験を行うための前準備 ==========#
    #cfgはconfig.ymlをpythonのdict形式に変換したもの
    print("実験設定")
    print(OmegaConf.to_yaml(cfg))#config表示

    #ディレクトリ取得
    cwd = hydra.utils.get_original_cwd()
    mlflow.set_tracking_uri("file://" + "/home/omote/b3_session_projects/mlruns")#絶対pathの先頭に"file://"をつける

    #時刻設定
    dt_now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    time_now = dt_now.strftime('%Y年%m月%d日 %H:%M:%S')

    #MLflow Tracking を使用して実験過程、結果を記録するwriterを作成
    EXPERIMENT_NAME = f"{cfg.name}" #mlflow uiの「Experiments」に表示される実験名
    run_tags = {'seed':cfg.seed,
        "time":time_now,
        MLFLOW_RUN_NAME:cfg.name,
        MLFLOW_USER:"hideaki Omote",
        MLFLOW_SOURCE_NAME:__file__,
        }
    writer = myExp.MlflowWriter(experiment_name= EXPERIMENT_NAME,run_tags=run_tags) #writerを作成しexperiment=実験を開始
    writer.log_params_from_omegaconf_dict(cfg)#cfgのハイパーパラメーターを一括でwriterで記録(mlflow ui　の Parameters)

    #seedを固定化
    myExp.fix_seed(cfg.seed) #seed固定化
    fix_seed_dataLoader = myExp.initialize_fix_seed_dataLoader(cfg.seed)#seed固定のDataLoader関数を作成　DataLoader関数の代わりに使用する

    #デバイス設定
    device = "cuda" if torch.cuda.is_available() else "cpu" #デバイス設定
    print(f"device：{device}\n")


    #========== 学習、検証用のデータを準備 ==========#
    #データ拡張
    train_transform = myExp.make_transform(**cfg.transform)
     
    test_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    print(test_transform)
    dataset = myExp.load_cifar10(transform=test_transform, train=True) #訓練・検証用データが一緒なのでなのでtransformしない
   
    # データ数とデータ形状を確認
    print("データ数とデータ形状")
    print(f'data size : {len(dataset)}')
    print(f'data shape : {dataset[0][0].shape}\n')
    # 学習用(Train)と検証用(Validation)への分割数を決める
    valid_ratio = 0.1                               # 検証用データの割合を指定
    valid_size = int(len(dataset) * valid_ratio)    # 検証用データの数
    train_size = len(dataset) - valid_size          # 学習用データ = 全体 - 検証用データ
    print("訓練、検証データの数")
    print(f'train samples : {train_size}')
    print(f'valid samples : {valid_size}\n')
    # 読み込んだデータをランダムに選んで分割
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    train_dataset.dataset.transform = train_transform #訓練用のデータセットにtransform適用


    # データローダーを作成
    train_loader = fix_seed_dataLoader(
        train_dataset,          # データセットを指定
        #batch_size=256,          # バッチサイズを指定
        shuffle=True,           # シャッフルの有無を指定
        drop_last=True,         # バッチサイズで割り切れないデータの使用の有無を指定
        pin_memory=True,        # 少しだけ高速化が期待できるおまじない
        num_workers=4,          # DataLoaderのプロセス数を指定
        **cfg.dataloader
    )
    valid_loader = fix_seed_dataLoader(
        valid_dataset,
        #batch_size=256,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        **cfg.dataloader
    ) 

        #========== テスト ==========#
    test_dataset = myExp.load_cifar10(transform=test_transform, train=False) #データセット作成

    #データローダー作成
    test_loader = fix_seed_dataLoader(
        test_dataset,
        #batch_size=256,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        **cfg.dataloader    
    )
    print(f"test : {len(test_dataset):5.0f} [set]")


    #========== モデルと損失関数、最適化関数を作成 ==========#
    # モデルの構築
    model = myExp.Classifier(class_num=10,in_w=32, in_h=32,**cfg.model)
    # 計算に使用するデバイスへモデルを転送
    model.to(device)
    # 重みの初期化
    model.apply(myExp.initialize_weights)
    # 誤差関数(損失関数)を指定
    criterion = nn.CrossEntropyLoss()  # 交差エントロピー
    print("損失関数")
    print(criterion)
    print("\n")
    # 最適化アルゴリズムを指定
    optimizer = myExp.make_optimizer(model.parameters(), **cfg.optimizer)  # 最適化関数Adam
    #　学習率調整スケジューラを作成
    scheduler = myExp.make_scedular(optimizer=optimizer,**cfg.scheduler)
    print("最適化関数")
    print(optimizer)
    print("\n")


    #========== 学習と検証 ==========#
    # 学習過程を保存するためのリストを用意
    history = {
        'train_loss': [],
        'train_acc' : [],
        'valid_loss': [],
        'valid_acc' : [],
        'test_loss' : [],
        'test_acc'  : []
    }

    # エポックの数だけ繰り返す
    for epoch in range(cfg.epochs):
        #========== 1エポックの処理 ==========#
         # プログレスバーの用意
        loop = tqdm(train_loader, unit='batch', desc='| Train | Epoch {:>3} |'.format(epoch+1))#loop=train_loader

        train_metrics = myExp.train_model(loop,model,criterion,optimizer,device)#訓練
        valid_metrics = myExp.valid_model(valid_loader,model,criterion,device)#検証

        #訓練、検証結果を取り出す
        train_loss = train_metrics['train_loss']
        valid_loss = valid_metrics['valid_loss']
        train_acc = train_metrics['train_acc']
        valid_acc = valid_metrics['valid_acc']

        # 学習過程をprint用のhistoryに追加
        history['train_loss'].append(train_metrics['train_loss'])
        history['valid_loss'].append(valid_metrics['valid_loss'])
        history['train_acc'].append(train_metrics['train_acc'])
        history['valid_acc'].append(valid_metrics['valid_acc'])

        # 学習率を調整
        if scheduler == "ReduceLROnPlateau":
            scheduler.step(valid_loss)

        now_lr = optimizer.param_groups[0]["lr"]
        writer.log_metric_step("lr",now_lr,epoch+1)

        #学習過程をmlflowに記録
        #log_metric_stepは連続的に変化する値を保存
        writer.log_metric_step('train_loss', train_loss,epoch+1) #引数：評価指標の名前,値,step
        writer.log_metric_step('train_acc', train_acc,epoch+1)
        writer.log_metric_step('valid_loss', valid_loss,epoch+1)
        writer.log_metric_step('valid_acc', valid_acc,epoch+1)

        #loop = tqdm(test_loader, unit='batch', desc='| Test | Epoch {:>3} |'.format(epoch+1)) #loop = test_loader
        test_metrics = myExp.test_model(test_loader,model,criterion,device)#テストデータで評価

        #テスト結果を取り出し
        test_loss = test_metrics['test_loss']
        test_acc = test_metrics['test_acc']

        #テスト結果を保存
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        # #print("テストデータに対する結果")
        # test_loss_str = f"test_loss   ：{test_loss:3.5f}"
        #print(test_loss_str)
        #test_acc_str = f"test_acc    ：{test_acc*100:2.3f}%"
        #print(test_acc_str)

        #writerでテスト結果を保存
        #log_metric関数は単一の評価指標を記録するのに使用
        writer.log_metric_step('test_loss', test_loss,epoch+1) #引数：　評価指標の名前,値
        writer.log_metric_step('test_acc', test_acc,epoch+1)

        print(f"| Train | Epoch   {epoch+1} |: train_loss:{train_loss:.3f}, train_acc:{train_acc*100:3.3f}% | valid_loss:{valid_loss:.5f}, valid_acc:{valid_acc*100:3.3f}%, | test_loss:{test_loss:.5f}, test_acc:{test_acc*100:3.3f}%, | now_lr:{now_lr:.1E}%")
        #過学習を起こしているなら次の10n回目で学習中断
        # if (valid_loss - train_loss) / valid_loss >= 0.5 and (epoch+1)%10 == 0 :
        #     print("過学習なので停止")
        #     break

    #最大値を保存
    writer.log_metric('train_loss_max', max(history["train_loss"])) #引数：評価指標の名前,値,step
    writer.log_metric('train_acc_max',  max(history["train_acc"]))
    writer.log_metric('valid_loss_max',  max(history["valid_loss"]))
    writer.log_metric('valid_acc_max',  max(history["valid_acc"]))
    writer.log_metric('test_loss_max',  max(history["test_loss"])) #引数：　評価指標の名前,値
    writer.log_metric('test_acc_max',  max(history["test_acc"]))
    writer.log_metric('train_loss_max_step', np.argmax(history["train_loss"])+1) #引数：評価指標の名前,値,step
    writer.log_metric('train_acc_max_step',  np.argmax(history["train_acc"])+1)
    writer.log_metric('valid_loss_max_step',  np.argmax(history["valid_loss"])+1)
    writer.log_metric('valid_acc_max_step',  np.argmax(history["valid_acc"])+1)
    writer.log_metric('test_loss_max_step',  np.argmax(history["test_loss"])+1) #引数：　評価指標の名前,値
    writer.log_metric('test_acc_max_step',  np.argmax(history["test_acc"])+1)

    print('Finished Training\n')
    

    # #========== テスト ==========#
    # test_dataset = load_cifar10(transform=test_transform, train=False) #データセット作成

    # #データローダー作成
    # test_loader = fix_seed_dataLoader(
    #     test_dataset,
    #     #batch_size=256,
    #     shuffle=False,
    #     pin_memory=True,
    #     num_workers=4,
    #     **cfg.dataloader    
    # )
    # print(f"test : {len(test_dataset):5.0f} [set]")

    # loop = tqdm(test_loader, unit='batch', desc='| Test | Epoch {:>3} |'.format(epoch+1)) #loop = test_loader
    # test_metrics = test_model(loop,model,criterion,device)#テストデータで評価

    # #テスト結果を取り出し
    # test_loss = test_metrics['test_loss']
    # test_acc = test_metrics['test_acc']

    # print("テストデータに対する結果")
    # test_loss_str = f"test_loss   ：{test_loss:3.5f}"
    # print(test_loss_str)
    # test_acc_str = f"test_acc    ：{test_acc*100:2.3f}%"
    # print(test_acc_str)

    # #writerでテスト結果を保存
    # #log_metric関数は単一の評価指標を記録するのに使用
    # writer.log_metric('test_loss', test_loss) #引数：　評価指標の名前,値
    # writer.log_metric('test_acc', test_acc)




    writer.set_terminated()#一つのexperimentまたはrunを終えるときは必ず呼び出す

if __name__ == '__main__':
    main2()




