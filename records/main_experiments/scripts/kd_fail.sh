# # find out when kd fails
# python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8 -r 0.1 -a 0.9 -b 0 --trial 2
# python train_student.py --path_t ./save/models/resnet8x4_vanilla/resnet8x4_last.pth --distill kd --model_s resnet8 -r 0.1 -a 0.9 -b 0 --trial 2
# python train_student.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8 -r 0.1 -a 0.9 -b 0 --trial 2
# python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8 -r 0.1 -a 0.9 -b 0 --trial 2
# python train_student.py --path_t ./save/models/resnet44_vanilla/resnet44_last.pth --distill kd --model_s resnet8 -r 0.1 -a 0.9 -b 0 --trial 2
# python train_student.py --path_t ./save/models/resnet32_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8 -r 0.1 -a 0.9 -b 0 --trial 2
# python train_student.py --path_t ./save/models/resnet20_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8 -r 0.1 -a 0.9 -b 0 --trial 2
# python train_student.py --path_t ./save/models/resnet14_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8 -r 0.1 -a 0.9 -b 0 --trial 2
# python train_student.py --path_t ./save/models/plane10_vanilla/plane10_best.pth --distill kd --model_s plane2 -r 0.1 -a 0.9 -b 0 --trial 2
# python train_student.py --path_t ./save/models/plane10_cifar100/teacher_plane10_cifar100_best_checkpoint.tar --distill kd --model_s plane2 -r 0.1 -a 0.9 -b 0 --trial 2
# python train_teacher.py --model resnet32x4 --trial 1 --dataset cifar10
# python train_teacher.py --model resnet8x4 --trial 1 --dataset cifar10
# python train_teacher.py --model resnet110 --trial 1 --dataset cifar10
# python train_teacher.py --model resnet56 --trial 1 --dataset cifar10
# python train_teacher.py --model resnet44 --trial 1 --dataset cifar10
# python train_teacher.py --model resnet32 --trial 1 --dataset cifar10
# python train_teacher.py --model resnet20 --trial 1 --dataset cifar10
# python train_teacher.py --model resnet14 --trial 1 --dataset cifar10
# python train_teacher.py --model resnet8 --trial 1 --dataset cifar10
# python train_teacher.py --model plane8 --trial 1 --epochs 160 --dataset cifar10
# python train_teacher.py --model plane10 --trial 1 --epochs 160 --dataset cifar10
# python train_teacher.py --model plane8 --trial 1 --epochs 160 --dataset cifar10
# python train_teacher.py --model plane6 --trial 1 --epochs 160 --dataset cifar10
# python train_teacher.py --model plane4 --trial 1 --epochs 160 --dataset cifar10
# python train_teacher.py --model plane2 --trial 1 --epochs 160 --dataset cifar10
# python train_student.py --model_s resnet8 --trial 3  --distill kd --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth -r 0.1 -a 0.9 -b 0
# python train_student.py --model_s resnet14 --trial 3 --distill kd --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth -r 0.1 -a 0.9 -b 0
# python train_student.py --model_s resnet20 --trial 3 --distill kd --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth -r 0.1 -a 0.9 -b 0
# python train_student.py --model_s resnet32 --trial 3 --distill kd --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth -r 0.1 -a 0.9 -b 0
# python train_student.py --model_s resnet44 --trial 3 --distill kd --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth -r 0.1 -a 0.9 -b 0
# python train_student.py --model_s resnet56 --trial 3 --distill kd --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth -r 0.1 -a 0.9 -b 0
# python train_student.py --model_s resnet110 --trial 3 --distill kd --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth -r 0.1 -a 0.9 -b 0
# python train_student.py --model_s plane10 --trial 3 --distill kd --path_t ./save/models/plane10_cifar100/plane10_best.pth -r 0.1 -a 0.9 -b 0 --epochs 160
# python train_student.py --model_s plane8 --trial 3  --distill kd --path_t ./save/models/plane10_cifar100/plane10_best.pth -r 0.1 -a 0.9 -b 0 --epochs 160
# python train_student.py --model_s plane6 --trial 3 --distill kd --path_t ./save/models/plane10_cifar100/plane10_best.pth -r 0.1 -a 0.9 -b 0 --epochs 160
# python train_student.py --model_s plane4 --trial 3 --distill kd --path_t ./save/models/plane10_cifar100/plane10_best.pth -r 0.1 -a 0.9 -b 0 --epochs 160
# python train_student.py --model_s plane2 --trial 3 --distill kd --path_t ./save/models/plane10_cifar100/plane10_best.pth -r 0.1 -a 0.9 -b 0
# python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg11 -r 0.1 -a 0.9 -b 0 --trial 2
# python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --trial 2
# python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8 -r 0.1 -a 0.9 -b 0 --trial 2
# python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8-r 0.1 -a 0.9 -b 0 --trial 2
# python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s plane2 -r 0.1 -a 0.9 -b 0 --trial 2
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --trial 3
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet110 -r 0.1 -a 0.9 -b 0 --trial 3
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet56 -r 0.1 -a 0.9 -b 0 --trial 3
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet44 -r 0.1 -a 0.9 -b 0 --trial 3
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet32 -r 0.1 -a 0.9 -b 0 --trial 3
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet20 -r 0.1 -a 0.9 -b 0 --trial 3
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet14 -r 0.1 -a 0.9 -b 0 --trial 3
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8 -r 0.1 -a 0.9 -b 0 --trial 3