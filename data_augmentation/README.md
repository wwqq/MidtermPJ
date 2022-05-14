

# 不同数据增强方法下图像分类的性能对比实验

周笑宇 欧阳宛露 杨越 向敏

## Training

运行以下两种数据增强时需要先

```
$cd CutMix_Mixup
```

运行Mixup的命令：

```
python train.py --lr=0.1 --seed=20210618 --decay=1e-4 --batch-size 64 --model ResNet50 --method mixup --epoch 100 --name mixup_resnet50

python train.py --lr=0.1 --seed=20210618 --decay=1e-4 --batch-size 64 --model vgg --method mixup --epoch 100 --name mixup_vgg
```

运行CutMix的命令：

```
python train.py --lr=0.25 --seed=20210618 --decay=1e-4 --batch-size 64 --model ResNet50 --method cutmix --epoch 100 --name cutmix_resnet50

python train.py --lr=0.25 --seed=20210618 --decay=1e-4 --batch-size 64 --model vgg --method cutmix --epoch 100 --name cutmix_vgg
```

以下两种情形运行时需要先

```
$cd Cutout_None
```

运行Cutout的命令：

```
python train.py --dataset cifar100 --model resnet50 --data_augmentation --cutout --length 8 --batch_size 64 --epochs 100

python train.py --dataset cifar100 --model vgg --data_augmentation --cutout --length 8 --batch_size 64 --epochs 100
```

不对图像做以上三种数据增强时的命令：

```
python train.py --dataset cifar100 --model resnet50 --data_augmentation --batch_size 64 --epochs 100

python train.py --dataset cifar100 --model vgg --data_augmentation --batch_size 64 --epochs 100
```

