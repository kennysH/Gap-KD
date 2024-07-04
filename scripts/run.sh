# train TA
python train_TA.py --path_t ./save/main_experiments/cifar100/Teaacher/resnet110_vanilla/ckpt_epoch_240.pth --distill kd \
--model_s resnet20 -r 0.1 -a 0.9 -b 0 --adjust rate_decay --Tmax 20
# train student
python train_student.py --model_s resnet8 --path_t ./save/main_experiments/cifar100/Teaacher/resnet110_vanilla/ckpt_epoch_240.pth \
--path_ta ./save/main_experiments/cifar100/TA/resnet20/resnet20_best.pth \
--gap_beta 5.051 --gap_alpha 1.917 --Tmax 20 --ce_weight 8.949
