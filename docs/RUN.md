
### Main Experiments

To reproduce the main results of ROSITA on all datasets, run

```
sh scripts/run.sh clip Rosita 0
```

The baselines can be run using the same script by choosing the VLM backbone and the TTA method as follows
```
sh scripts/run.sh [clip/maple] [ZSEval/ROSITA/TPT/TPTContinual/PromptAlign/PromptAlignContinual] GPU_ID
```


To run a specific experiment, select the weak, strong OOD datasets, base VLM and the TTA method to evaluate from the following list.

```
python owtta.py --dataset [cifar10OOD/cifar100OOD/ImageNetCOOD/ImageNetROOD/VisDAOOD] --strong_OOD [MNIST/SVHN/Tiny/cifar10/cifar100] --model [coop/maple] --tta_method [ZSEval/ROSITA/TPT/TPTContinual/PromptAlign/PromptAlignContinual]
```

For e.g., to run PromptAlign using MaPLe as the VLM, CIFAR-10C as weak OOD and Tiny ImageNet as strong OOD, the following command should be used.
```
python owtta.py --dataset cifar10OOD --strong_OOD Tiny --model maple --tta_method ROSITA
```

### Supported Weak and Strong OOD combinations

```
python owtta.py --dataset cifar10OOD --strong_OOD [MNIST/SVHN/Tiny/cifar100] --model [coop/maple] --tta_method ROSITA
python owtta.py --dataset cifar100OOD --strong_OOD [MNIST/SVHN/Tiny/cifar10] --model [coop/maple] --tta_method ROSITA
python owtta.py --dataset ImageNetCOOD --strong_OOD [MNIST/SVHN] --model [coop/maple] --tta_method ROSITA
python owtta.py --dataset ImageNetROOD --strong_OOD [MNIST/SVHN] --model [coop/maple] --tta_method ROSITA
python owtta.py --dataset VisDAOOD --strong_OOD [MNIST/SVHN] --model [coop/maple] --tta_method ROSITA
```

### Supported VLM and Method combinations

```
python owtta.py --dataset cifar10OOD --strong_OOD Tiny --model coop --tta_method [ZSEval/ROSITA/TPT/TPTContinual]
python owtta.py --dataset cifar10OOD --strong_OOD Tiny --model coop --tta_method [ZSEval/ROSITA/TPT/TPTContinual/PromptAlign/PromptAlignContinual]
```


