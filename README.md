# GeneCAI

Official Github repo for paper titled "[GeneCAI:  Genetic evolution for acquiring  Compact  AI](https://dl.acm.org/doi/10.1145/3377930.3390226)" published in 2020 GECCO conference. The repo contains the source codes for applying structured pruning and decomposition on pre-trained models trained with the CIFAR10 dataset. 

## Running the Code
To compress the models, run `main.py` which expects to receive the checkpoint of a pretrained model in the following path: `dataset_name/architecutre_name/checkpoint/ckpt.t7`. For a complete list of parameters required by`main.py` run `python main.py --help`. In the following, we provide example commands for compressing a ResNet56 architecture:
1.  Place the pretrained checkpoint for the non-compressed model in `CIFAR10/ResNet56/checkpoint/ckpt.t7`
2. Find optimized per-layer decomposition paramaters for the pretrained model:
	`python main.py --phase D --arch ResNet56 --num_population 100 --acc_threshold 90`
3. Select one config from path `CIFAR10/ResNet56/best_configs/decomposition` (e.g., `iter_xxx_acc_yyy_flops_zzz.pkl`) and perform fine-tuning to recover the accuracy opf the decomposed model:
	`python main.py --phase D_FT --arch ResNet56 --config iter_xxx_acc_yyy_flops_zzz.pkl`
	This saves the pretrained checkpoint to `CIFAR10/ResNet56/checkpoint/decomposition/flops_xxx_acc_yyy.t7`
4. Find optimized per-layer pruning paramaters for the decomposed model:
	`python main.py --phase CP --arch ResNet56 --compressed_ckpt 'CIFAR10/ResNet56/checkpoint/decomposition/flops_xxx_acc_yyy.t7' --num_population 100 --acc_threshold 60`
5. Select one config from path `CIFAR10/ResNet56/best_configs/channel_pruning` (e.g., `iter_xxx_acc_yyy_flops_zzz.pkl`) and perform fine-tuning to recover the accuracy of the decomposed+pruned model:
	`python main.py --phase CP_FT --arch ResNet56 -compressed_ckpt 'CIFAR10/ResNet56/checkpoint/decomposition/flops_xxx_acc_yyy.t7' --config iter_xxx_acc_yyy_flops_zzz.pkl`
	This saves the pretrained checkpoint to `CIFAR10/ResNet56/checkpoint/pruned/flops_xxx_acc_yyy.t7`
