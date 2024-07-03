## sample scripts for running the distillation code
## use resnet32x4 and resnet8x4 as an example
#
### kd
##python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --trial 1
### FitNet
##python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill hint --model_s resnet8x4 -a 0 -b 100 --trial 1
### AT
##python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill attention --model_s resnet8x4 -a 0 -b 1000 --trial 1
### SP
##python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill similarity --model_s resnet8x4 -a 0 -b 3000 --trial 1
### CC
##python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill correlation --model_s resnet8x4 -a 0 -b 0.02 --trial 1
### VID
##python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill vid --model_s resnet8x4 -a 0 -b 1 --trial 1
### RKD
##python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill rkd --model_s resnet8x4 -a 0 -b 1 --trial 1
### PKT
##python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill pkt --model_s resnet8x4 -a 0 -b 30000 --trial 1
### AB
##python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill abound --model_s resnet8x4 -a 0 -b 1 --trial 1
### FT
##python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill factor --model_s resnet8x4 -a 0 -b 200 --trial 1
### FSP
##python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill fsp --model_s resnet8x4 -a 0 -b 50 --trial 1
### NST
##python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill nst --model_s resnet8x4 -a 0 -b 50 --trial 1
### CRD
##python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill crd --model_s resnet8x4 -a 0 -b 0.8 --trial 1
##
### CRD+KD
##python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill crd --model_s resnet8x4 -a 1 -b 0.8 --trial 1
#
## kd
##python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8 -r 0.1 -a 0.9 -b 0 --trial 0
##python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8 --adjust_gamma linear -r 0.1 -a 0.9 -b 0 --trial 0
##python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8 --adjust_gamma exponential -r 0.1 -a 0.9 -b 0 --trial 0
##python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8 --adjust_gamma logarithmic -r 0.1 -a 0.9 -b 0 --trial 0
##python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8 --adjust_gamma sigmoid -r 0.1 -a 0.9 -b 0 --trial 0
##python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8 --adjust_gamma piecewise -r 0.1 -a 0.9 -b 0 --trial 0
#python train_student.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8 -r 0.1 -a 0.9 -b 0 --trial 0
## FitNet
##python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill hint --model_s resnet8 -a 0 -b 100 --trial 0
#python train_student.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill hint --model_s resnet8 -a 0 -b 100 --trial 0
## AT
##python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill attention --model_s resnet8 -a 0 -b 1000 --trial 0
#python train_student.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill attention --model_s resnet8 -a 0 -b 1000 --trial 0
## SP
##python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill similarity --model_s resnet8 -a 0 -b 3000 --trial 0
#python train_student.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill similarity --model_s resnet8 -a 0 -b 3000 --trial 0
## CC
##python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill correlation --model_s resnet8 -a 0 -b 0.02 --trial 0
#python train_student.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill correlation --model_s resnet8 -a 0 -b 0.02 --trial 0
## VID
##python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill vid --model_s resnet8 -a 0 -b 1 --trial 0
#python train_student.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill vid --model_s resnet8 -a 0 -b 1 --trial 0
## RKD
##python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill rkd --model_s resnet8 -a 0 -b 1 --trial 0
#python train_student.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill rkd --model_s resnet8 -a 0 -b 1 --trial 0
## PKT
##python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill pkt --model_s resnet8 -a 0 -b 30000 --trial 0
#python train_student.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill pkt --model_s resnet8 -a 0 -b 30000 --trial 0
## AB
##python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill abound --model_s resnet8 -a 0 -b 1 --trial 0
#python train_student.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill abound --model_s resnet8 -a 0 -b 1 --trial 0
#
## FT
##python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill factor --model_s resnet8 -a 0 -b 200 --trial 0
#python train_student.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill factor --model_s resnet8 -a 0 -b 200 --trial 0
## FSP
##python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill fsp --model_s resnet8 -a 0 -b 50 --trial 0
#python train_student.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill fsp --model_s resnet8 -a 0 -b 50 --trial 0
## NST
##python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill nst --model_s resnet8 -a 0 -b 50 --trial 0
#python train_student.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill nst --model_s resnet8 -a 0 -b 50 --trial 0
## CRD
##python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill crd --model_s resnet8 -a 0 -b 0.8 --trial 0
#python train_student.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill crd --model_s resnet8 -a 0 -b 0.8 --trial 0
## CRD+KD
##python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill crd --model_s resnet8 -a 1 -b 0.8 --trial 0
#python train_student.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill crd --model_s resnet8 -a 1 -b 0.8 --trial 0
#python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet14 -r 0.1 -a 0.9 -b 0 --trial 1
#python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet20 -r 0.1 -a 0.9 -b 0 --trial 1
#python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet32 -r 0.1 -a 0.9 -b 0 --trial 1
#python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet44 -r 0.1 -a 0.9 -b 0 --trial 1
#python train_student_ta.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --path_ta ./save/student_model/resnet56_T:resnet110_cifar100_dkd_r:1.0_a:0.0_b:2.0_1_DKD/resnet56_best.pth --model_s resnet8 --trial 1
#python train_student_ta.py --seed 2125 --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --path_ta ./save/student_model/resnet56_T:resnet110_cifar100_dkd_r:1.0_a:0.0_b:2.0_1_DKD/resnet56_best.pth --model_s resnet8 --trial 1
#python train_student_ta.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --path_ta ./save/student_model/resnet44_T:resnet110_cifar100_kd_r:0.1_a:0.9_b:0.0_1/resnet44_best.pth --model_s resnet8 --trial 1
#python train_student_ta.py --seed 2125 --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --path_ta ./save/student_model/resnet44_T:resnet110_cifar100_kd_r:0.1_a:0.9_b:0.0_1/resnet44_best.pth --model_s resnet8 --trial 1
#python train_student_ta.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --path_ta ./save/student_model/resnet56_T:resnet110_cifar100_dkd_r:1.0_a:0.0_b:2.0_1_DKD/resnet56_best.pth --model_s resnet44 --trial 1
#python train_student_ta.py --seed 2125 --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --path_ta ./save/student_model/resnet56_T:resnet110_cifar100_dkd_r:1.0_a:0.0_b:2.0_1_DKD/resnet56_best.pth --model_s resnet44 --trial 1
#python train_student_ta.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --path_ta ./save/student_model/resnet56_T:resnet110_cifar100_dkd_r:1.0_a:0.0_b:2.0_1_DKD/resnet56_best.pth --model_s resnet32 --trial 1
#python train_student_ta.py --seed 2125 --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --path_ta ./save/student_model/resnet56_T:resnet110_cifar100_dkd_r:1.0_a:0.0_b:2.0_1_DKD/resnet56_best.pth --model_s resnet32 --trial 1
#python train_student_ta.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --path_ta ./save/student_model/resnet56_T:resnet110_cifar100_dkd_r:1.0_a:0.0_b:2.0_1_DKD/resnet56_best.pth --model_s resnet20 --trial 1
#python train_student_ta.py --seed 2125 --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --path_ta ./save/student_model/resnet56_T:resnet110_cifar100_dkd_r:1.0_a:0.0_b:2.0_1_DKD/resnet56_best.pth --model_s resnet20 --trial 1
#python train_student_ta.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --path_ta ./save/student_model/resnet56_T:resnet110_cifar100_dkd_r:1.0_a:0.0_b:2.0_1_DKD/resnet56_best.pth --model_s resnet14 --trial 1
#python train_student_ta.py --seed 2125 --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --path_ta ./save/student_model/resnet56_T:resnet110_cifar100_dkd_r:1.0_a:0.0_b:2.0_1_DKD/resnet56_best.pth --model_s resnet14 --trial 1
python train_student_alpha.py --model_s resnet8x4 --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth -r 0.1 -a 0.9 -b 0 --seed 2125 --Tmax 10
python train_student_alpha.py --model_s resnet8x4 --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth -r 0.1 -a 0.9 -b 0 -seed 3407 --Tmax 10
python train_student_alpha.py --model_s resnet8x4 --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth -r 0.1 -a 0.9 -b 0 --seed 2125 --Tmax 20
python train_student_alpha.py --model_s resnet8x4 --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth -r 0.1 -a 0.9 -b 0 -seed 3407 --Tmax 20

python train_student_alpha.py --model_s plane4 --path_t ./save/models/plane10_cifar100/teacher_plane10_cifar100_best_checkpoint.tar -r 0.1 -a 0.9 -b 0 --seed 2125 --Tmax 10


