from torchvision import transforms

##-----------------transformsを選択
"""
pattern
"":標準
"a":画像を水平反転
"b":画像を垂直反転
"c":画像を回転
"d":ランダムクロップ
"e": ランダムアージング
"""
#入力は
def make_transform(pattern=""):
    

    #必ず使うtransform
    default_transform_list = [
        transforms.ToTensor(),
        transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) ),
        ]
    #使用するtransform indexが番号に対応
    use_transform_list = {
        "a":transforms.RandomHorizontalFlip(p=0.5),#0:画像を水平反転
        "b":transforms.RandomVerticalFlip(p=0.5), #1:画像を垂直反転
        "c":transforms.RandomRotation(degrees=(-15,15)), #2:画像回転
        "d":transforms.RandomCrop(32, padding=4), #3:画像を切り抜き

        #transforms.GaussianBlur(kernel_size=15),
        #transforms.RandomPosterize(bits=1, p=1.0),
        #transforms.RandomAdjustSharpness(p=1.0, sharpness_factor=3),

        "e":transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0), #4:画像の一部を塗りつぶす
    }


    #transformを作成
    if(pattern == ""):#空ならデフォルト
        print(pattern)
        return  transforms.Compose([default_transform_list])

    else:#それいがいなら加工したtransformをわたす。
        print(list(pattern))
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
        print(transform_list)
        return transforms.Compose(transform_list)
        

print(make_transform("abce"))