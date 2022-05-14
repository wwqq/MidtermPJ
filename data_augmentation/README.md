

# CutOut CutMix MixUp data augmentation experiments and visualization

## Training

### Baseline & CutOut

```
$cd Cutout_None
```

For Cutout：

```
python train.py --dataset cifar100 --model vgg --data_augmentation --cutout --length 8 --batch_size 64 --epochs 100
```

Baseline：

```
python train.py --dataset cifar100 --model vgg --data_augmentation --batch_size 64 --epochs 100
```


### CutMix & Mixup

```
$cd CutMix_Mixup
```

For Mixup：

```
python train.py --lr=0.1 --seed=42 --decay=1e-4 --batch-size 64 --model vgg --method mixup --epoch 100 --name mixup_vgg
```

For CutMix：

```
python train.py --lr=0.25 --seed=42 --decay=1e-4 --batch-size 64 --model vgg --method cutmix --epoch 100 --name cutmix_vgg
```



