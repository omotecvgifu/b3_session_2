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

from mypkg import decorator


##--------------------










##------------シード関係
#クロージャを利用してseed値を固定 https://qiita.com/naomi7325/items/57d141f2e56d644bdf5f
def initialize_fix_seed_dataLoader(seed):
    def fix_seed_dataLoader(dataloader,**args):#シードを固定したDataLoader　DataLoaderの代わりに使う
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        g = torch.Generator()
        g.manual_seed(seed)
        return DataLoader(dataloader,**args,worker_init_fn=seed_worker,generator=g)
    
    return fix_seed_dataLoader

#シードを固定化する関数
def fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed) #cpuとcudaも同時に固定
    #torch.cuda.manual_seed(seed) #上記で呼び出される
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
##------------------------------------------------------


##-------------mlflowを使った記録するクラス関係
class MlflowWriter():
    # def __init__(self, experiment_name,tags):
    #     self.client = MlflowClient()
    #     try:
    #         self.experiment_id = self.client.create_experiment(experiment_name,tags)
    #     except Exception as e:
    #         print(e)
    #         self.experiment_id = self.client.get_experiment_by_name(experiment_name).experiment_id

    #     self.experiment = self.client.get_experiment(self.experiment_id)
    #     print("New experiment started")
    #     print(f"Name: {self.experiment.name}")
    #     print(f"Experiment_id: {self.experiment.experiment_id}")
    #     print(f"Artifact Location: {self.experiment.artifact_location}")

    def log_params_from_omegaconf_dict(self, params):
        for param_name, element in params.items():
            self._explore_recursive(param_name, element)

    def _explore_recursive(self, parent_name, element):
        if isinstance(element, DictConfig):
            for k, v in element.items():
                if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                    self._explore_recursive(f'{parent_name}.{k}', v)
                else:
                    self.client.log_param(self.run_id, f'{parent_name}.{k}', v)
        elif isinstance(element, ListConfig):
            for i, v in enumerate(element):
                self.client.log_param(self.run_id, f'{parent_name}.{i}', v)
        else:
            self.client.log_param(self.run_id, f'{parent_name}', element)

    def log_param(self, key, value):
        self.client.log_param(self.run_id, key, value)

    def log_metric(self, key, value):
        self.client.log_metric(self.run_id, key, value)

    def log_metric_step(self, key, value, step):
        self.client.log_metric(self.run_id, key, value, step=step)

    def log_artifact(self, local_path):
        self.client.log_artifact(self.run_id, local_path)

    def log_dict(self, dictionary, file):
        self.client.log_dict(self.run_id, dictionary, file)
    
    def log_figure(self, figure, file):
        self.client.log_figure(self.run_id, figure, file)
        
    def set_terminated(self):
        self.client.set_terminated(self.run_id)

    def create_new_run(self, tags=None):
        self.run = self.client.create_run(self.experiment_id, tags=tags)
        self.run_id = self.run.info.run_id
        print(f"New run started: {tags['mlflow.runName']}")
    def __init__(self, experiment_name,run_tags=None, **kwargs):
        self.client = MlflowClient(**kwargs)
        try:
            self.experiment_id = self.client.create_experiment(experiment_name)
        except:
            self.experiment_id = self.client.get_experiment_by_name(experiment_name).experiment_id

        self.run_id = self.client.create_run(self.experiment_id,tags=run_tags).info.run_id

    # def log_params_from_omegaconf_dict(self, params):
    #     for param_name, element in params.items():
    #         self._explore_recursive(param_name, element)

    # def _explore_recursive(self, parent_name, element):
    #     if isinstance(element, DictConfig):
    #         for k, v in element.items():
    #             if isinstance(v, DictConfig) or isinstance(v, ListConfig):
    #                 self._explore_recursive(f'{parent_name}.{k}', v)
    #             else:
    #                 self.client.log_param(self.run_id, f'{parent_name}.{k}', v)
    #     elif isinstance(element, ListConfig):
    #         for i, v in enumerate(element):
    #             self.client.log_param(self.run_id, f'{parent_name}.{i}', v)

    def log_torch_model(self, model):
        with mlflow.start_run(self.run_id):
            mlflow.pytorch.log_model(model, 'models')

    def log_param(self, key, value):
        self.client.log_param(self.run_id, key, value)

    def log_metric(self, key, value):
        self.client.log_metric(self.run_id, key, value)

    def log_artifact(self, local_path):
        self.client.log_artifact(self.run_id, local_path)

    def set_terminated(self):
        self.client.set_terminated(self.run_id)

    
##----------------------------------------------------------------



##------------データセットロード関係
def load_cifar10(transform, train:bool):
    cifar10_dataset = CIFAR10(
                        root='./data',
                        train=train,
                        download=True,
                        transform=transform
                        )
    return cifar10_dataset
##-------------------------

##-----------------transformsを選択
"""
pattern
"None":標準
"a":画像を水平反転
"b":画像を回転
"c":ランダムクロップ
"d": ランダムアージング
"""
#入力は
def make_transform(pattern=None):
    

    #必ず使うtransform
    default_transform_list = [
        transforms.ToTensor(),
        transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) ),
        ]
    #使用するtransform indexが番号に対応
    use_transform_list = {
        "a":transforms.RandomHorizontalFlip(p=0.5),#0:画像を水平反転
        #"b":transforms.RandomVerticalFlip(p=0.5), #1:画像を垂直反転
        "b":transforms.RandomRotation(degrees=(-10,10)), #2:画像回転
        "c":transforms.RandomCrop(32, padding=4), #3:画像を切り抜き

        #transforms.GaussianBlur(kernel_size=15),
        #transforms.RandomPosterize(bits=1, p=1.0),
        #transforms.RandomAdjustSharpness(p=1.0, sharpness_factor=3),

        "d":transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0), #4:画像の一部を塗りつぶす
    }


    #transformを作成
    if(pattern == None):#空ならデフォルト
        print("transform")
        print(pattern)
        print(default_transform_list)
        print("\n")
        return  transforms.Compose(default_transform_list)

    else:#それいがいなら加工したtransformをわたす。
        
        pattern_split = list(pattern)

        pre_transform_list = []
        suf_transform_list = []

        for value in pattern_split:
            use_transform = use_transform_list[value]
            if isinstance(use_transform,transforms.RandomErasing):
                suf_transform_list.append(use_transform)
            else:
                pre_transform_list.append(use_transform)

        transform_list = pre_transform_list + default_transform_list + suf_transform_list
        
        print("transform")
        print(pattern)
        print(transform_list)
        print("\n")
        return transforms.Compose(transform_list)

##----------モデル関係
# Convolution層、BatchNormalization層(option)、ReLU層をまとめたブロック
# ブロックを作成することで、モデル定義のコードが煩雑になることを防ぐ
class EncoderBlock(nn.Module):
    def __init__(self, in_feature, out_future, use_bn=True):
        super().__init__()
        self.use_bn = use_bn

        self.in_feature = in_feature
        self.out_feature = out_future

        # BatchNormalizationを使用する場合、バイアス項は無くても良い
        # 標準化の際にバイアスもまとめて処理されるため
        self.conv = nn.Conv2d(in_feature, out_future, kernel_size=3, stride=1, padding=1, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_future)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out) if self.use_bn else x  # bn層を使わない場合はxを代入
        out = self.relu(out)
        return out

class Classifier(nn.Module):
    def __init__(self, class_num, enc_dim, in_w, in_h,type=0):
        super().__init__()

        if(type == 0):
            # self.enc_dim = enc_dim
            self.in_w = in_w
            self.in_h = in_h
            self.fc_dim = enc_dim*4 * int(in_h/2/2/2) * int(in_w/2/2/2) #pooling回数分割る
            self.class_num = class_num

            self.encoder = nn.Sequential(
                EncoderBlock(3      , enc_dim),
                EncoderBlock(enc_dim, enc_dim),
                nn.MaxPool2d(kernel_size=2),#h,wが1/2

                EncoderBlock(enc_dim  , enc_dim*2),
                EncoderBlock(enc_dim*2, enc_dim*2),
                nn.MaxPool2d(kernel_size=2),

                EncoderBlock(enc_dim*2, enc_dim*4),
                EncoderBlock(enc_dim*4, enc_dim*4),
                nn.MaxPool2d(kernel_size=2),
            )


            self.fc = nn.Sequential(
                nn.Linear(self.fc_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, self.class_num),
            )
        elif(type == 10):
            self.in_w = in_w
            self.in_h = in_h
            self.fc_dim = enc_dim*4 * int(in_h/2/2/2) * int(in_w/2/2/2) #pooling回数分割る
            self.class_num = class_num

            self.encoder = nn.Sequential(
                EncoderBlock(3      , enc_dim),
                EncoderBlock(enc_dim, enc_dim),
                EncoderBlock(enc_dim, enc_dim),
                nn.MaxPool2d(kernel_size=2),#h,wが1/2

                EncoderBlock(enc_dim  , enc_dim*2),
                EncoderBlock(enc_dim*2, enc_dim*2),
                EncoderBlock(enc_dim*2, enc_dim*2),
                nn.MaxPool2d(kernel_size=2),

                EncoderBlock(enc_dim*2, enc_dim*4),
                EncoderBlock(enc_dim*4, enc_dim*4),
                EncoderBlock(enc_dim*4, enc_dim*4),
                nn.MaxPool2d(kernel_size=2),
            )


            self.fc = nn.Sequential(
                nn.Linear(self.fc_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, self.class_num),
            )
        elif(type == 1):
            self.in_w = in_w
            self.in_h = in_h
            self.fc_dim = enc_dim*8 * int(in_h/2/2/2) * int(in_w/2/2/2) #pooling回数分割る
            self.class_num = class_num

            self.encoder = nn.Sequential(
                EncoderBlock(3      , enc_dim),
                EncoderBlock(enc_dim, enc_dim),
                nn.MaxPool2d(kernel_size=2),#h,wが1/2

                EncoderBlock(enc_dim  , enc_dim*2),
                EncoderBlock(enc_dim*2, enc_dim*2),
                nn.MaxPool2d(kernel_size=2),

                EncoderBlock(enc_dim*2, enc_dim*4),
                EncoderBlock(enc_dim*4, enc_dim*4),
                nn.MaxPool2d(kernel_size=2),

                EncoderBlock(enc_dim*4, enc_dim*8),
                EncoderBlock(enc_dim*8, enc_dim*8),
            )


            self.fc = nn.Sequential(
                nn.Linear(self.fc_dim,1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, self.class_num),
            )
        
        elif(type == 2):
            self.in_w = in_w
            self.in_h = in_h
            self.fc_dim = enc_dim * int(in_h/2/2/2) * int(in_w/2/2/2) #pooling回数分割る
            self.class_num = class_num

            self.encoder = nn.Sequential(
                EncoderBlock(3      , enc_dim),
                EncoderBlock(enc_dim, enc_dim*2),
                EncoderBlock(enc_dim*2, enc_dim*2),
                nn.MaxPool2d(kernel_size=2),#h,wが1/2

                EncoderBlock(enc_dim*2, enc_dim*4),
                EncoderBlock(enc_dim*4, enc_dim*4),
                EncoderBlock(enc_dim*4, enc_dim*8),
                nn.MaxPool2d(kernel_size=2),

                EncoderBlock(enc_dim*8, enc_dim*8),
                EncoderBlock(enc_dim*8, enc_dim*4),
                EncoderBlock(enc_dim*4, enc_dim*4),
                nn.MaxPool2d(kernel_size=2),

                EncoderBlock(enc_dim*4, enc_dim*2),
                EncoderBlock(enc_dim*2, enc_dim*2),
                EncoderBlock(enc_dim*2, enc_dim),
            )


            self.fc = nn.Sequential(
                nn.Linear(self.fc_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, self.class_num),
            )
    def forward(self, x):
        out = self.encoder(x)
        out = out.view(-1, self.fc_dim)

        out = self.fc(out)

        return out

def initialize_weights(m):
    if isinstance(m, nn.Conv2d): # Convolution層が引数に渡された場合
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu') # kaimingの初期化
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)   # bias項は0に初期化
    elif isinstance(m, nn.BatchNorm2d):         # BatchNormalization層が引数に渡された場合
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):              # 全結合層が引数に渡された場合
        nn.init.kaiming_normal_(m.weight.data)  # kaimingの初期化
        nn.init.constant_(m.bias.data, 0)       # biasは0に初期化
##--------------------------------------------


##optimizer関係
def make_optimizer(params, name, **kwargs):
    # Optimizer作成
    return optim.__dict__[name](params, **kwargs)

##--------------------

##--------scedular関係
def make_scedular(optimizer,name):
        #　学習率調整スケジューラを作成
    if name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer, mode = 'min',patience=5,factor=0.1) 
    elif name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=40,eta_min=1.0e-6)
    elif name == None:
        scheduler = None
    else:
        raise Exception("スケジューラが存在しない")
    print("スケジューラー")
    print(scheduler)
    return scheduler
##---------------------訓練関係
def train_model(train_loader,model,criterion,optimizer,device):
        #========== 学習用データへの処理 ==========#
    # モデルを学習モードに設定
    model.train()

    #ミニバッチ＝イテレーションごとの
    batch_loss = []
    batch_acc  = []

    # ミニバッチ単位で繰り返す
    for batch in train_loader:
        x, label = batch
        x = x.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        # 学習のメイン処理
        optimizer.zero_grad()       # (1) パラメータの勾配を初期化
        out = model(x)              # (2) データをモデルに入力(順伝播)
        loss = criterion(out, label)    # (3) 誤差関数の値を算出
        loss.backward()             # (4) パラメータの勾配を算出(逆伝播)
        optimizer.step()            # (5) 勾配の情報からパラメータを更新

        # 正答率(accuracy)を算出
        pred_class = torch.argmax(out, dim=1)        # モデルの出力から予測ラベルを算出
        acc = torch.sum(pred_class == label)/len(label) # 予測ラベルと正解ラベルが一致すれば正解

        # 1バッチ分の誤差と正答率をリストに保存
        batch_loss.append(loss)
        batch_acc.append(acc)

    # バッチ単位の結果を平均し、1エポック分の誤差を算出
    train_avg_loss = torch.tensor(batch_loss).mean()
    train_avg_acc  = torch.tensor(batch_acc).mean()

    metrics = {"train_loss":train_avg_loss,"train_acc":train_avg_acc}

    return metrics
##-------------------------------------------

##---------------------検証関係
def valid_model(valid_loader,model,criterion,device):
        #========== 検証用データへの処理 ==========#
    # モデルを評価モードに設定
    model.eval()
    batch_loss = []
    batch_acc = []

    # 勾配計算に必要な情報を記録しないよう設定
    with torch.no_grad():
        # ミニバッチ単位で繰り返す
        for batch in valid_loader:
            x, label = batch
            x = x.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            # 順伝播と誤差の計算のみ行う
            out = model(x)
            loss = criterion(out, label)

            # 正答率(accuracy)を算出
            pred_class = torch.argmax(out, dim=1)
            acc = torch.sum(pred_class == label)/len(label)

            # 1バッチ分の誤差をリストに保存
            batch_loss.append(loss)
            batch_acc.append(acc)

        # バッチ単位の結果を平均し、1エポック分の誤差を算出
        valid_avg_loss = torch.tensor(batch_loss).mean()
        valid_avg_acc  = torch.tensor(batch_acc).mean()

    metrics = {"valid_loss":valid_avg_loss,"valid_acc":valid_avg_acc}
    return metrics
##----------------------------------------------------

##----------------------テスト関係
# モデルを評価モードに設定
def test_model(test_loader,model,criterion,device):
    model.eval()
    batch_loss = []
    batch_acc = []

    # 勾配計算に必要な情報を記録しないよう設定
    with torch.no_grad():
        # ミニバッチ単位で繰り返す
        for batch in test_loader:
            x, label = batch
            x = x.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            # 順伝播のみ行う
            out = model(x)
            batch_loss = criterion(out, label)

            # 正答率(accuracy)を算出
            pred_class = torch.argmax(out, dim=1)
            acc = torch.sum(pred_class == label)/len(label)

            # 1バッチ分のaccをリストに保存
            batch_acc.append(acc)

        # バッチ単位の結果を平均し、1エポック分の誤差を算出
        test_avg_loss = torch.tensor(batch_loss).mean().item()
        test_avg_acc  = torch.tensor(batch_acc).mean().item()

    metrics = {"test_loss":test_avg_loss,"test_acc":test_avg_acc}
    return metrics

##--------------------------------
