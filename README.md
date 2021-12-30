# ContrastiveLeak

CUDA_VISIBLE_DEVICES=5 python3 ce_classifier.py --batch_size 512 --dataset CIFAR10 --model resnet18 --mode shadow --data_path /home/xinlei.he/simclr/data/

## Train contrastive classifiers:

0. pretrain resnet18 with STL10 unlabeled dataset
```
python3 run.py --gpu 0 --dataset STL10 --model resnet18 --adv_training no --mode target --pretrain yes
```
1. Finetune in downstream dataset, e.g., CIFAR10

   Pretraining stage:
   ```
   python3 run.py --gpu 0 --dataset CIFAR10 --model resnet18 --adv_training no --mode target
   python3 run.py --gpu 0 --dataset CIFAR10 --model resnet18 --adv_training no --mode shadow
   ```
   
   Linear stage:
   
   ```
   python3 classifier.py --batch_size 512 --dataset CIFAR10 --model resnet18 --mode target
   python3 classifier.py --batch_size 512 --dataset CIFAR10 --model resnet18 --mode shadow
   ```

## Train normal classifiers with cross-entropy loss:
```
python3 ce_classifier.py --batch_size 512 --dataset CIFAR10 --model resnet18 --mode target
python3 ce_classifier.py --batch_size 512 --dataset CIFAR10 --model resnet18 --mode shadow
```

## Conduct membership inference attack
`
method = ["CE", "SimCLR"]
mia_type = ['nn-based', "metric-based", "label-only"]
`
```
python3 MIA.py --batch_size 512 --dataset CIFAR10 --model resnet18 --method SimCLR --mia_type nn-based
```

Note that for dataset that used for attribute inference, you may need to specify the original_label and  aux_label
```
python3 MIA.py --batch_size 512 --dataset UTKFace --original_label Gender --aux_label Race --model resnet18 --method SimCLR --mia_type metric-based
```

## Conduct attribute inference attack
```
python3 OL.py --batch_size 512 --dataset UTKFace --model resnet18 --original_label Gender --aux_label Race --method SimCLR 
```


## Adversarial Training (Talos)
```
# Pretraining stage
python3 run.py --batch_size 512 --dataset UTKFace --model resnet18 --mode target --adv_training yes --adv_image augmented --adv_location embedding --adv_factor 1 --original_label Gender --aux_label Race 
python3 run.py --batch_size 512 --dataset UTKFace --model resnet18 --mode shadow --adv_training yes --adv_image augmented --adv_location embedding --adv_factor 1 --original_label Gender --aux_label Race 

# Linear stage
python3 classifier_with_adv_pretrained_SimCLR.py --batch_size 512 --dataset UTKFace --model resnet18 --mode target --adv_training yes --adv_image augmented --adv_location embedding --adv_factor 1 --original_label Gender --aux_label Race
python3 classifier_with_adv_pretrained_SimCLR.py --batch_size 512 --dataset UTKFace --model resnet18 --mode shadow --adv_training yes --adv_image augmented --adv_location embedding --adv_factor 1 --original_label Gender --aux_label Race

```

##  Membership Inference Attack Against Talos 
```
python3 MIA_adv_simclr.py --batch_size 512 --dataset UTKFace --model resnet18  --original_label Gender --aux_label Race --adv_factor 1 --method SimCLR

```

##  Attribute Inference Attack Against Talos 
```
python3 OL_adv_SimCLR.py --batch_size 512 --dataset UTKFace --model resnet18  --original_label Gender --aux_label Race --adv_factor 1 --method SimCLR

```



