python train_student_ta.py --model_s resnet8 --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth \
--path_ta ./save/student_model/resnet20_T:resnet110_cifar100_dkd_r:1.0_a:0.0_b:8.0_1/resnet20_best.pth \
--tadkd_beta 5.051 --tadkd_alpha 1.917 --Tmax 20 --ce_weight 8.949
python train_student_ta.py --model_s resnet8 --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth \
--path_ta ./save/student_model/resnet20_T:resnet110_cifar100_dkd_r:1.0_a:0.0_b:8.0_1/resnet20_best.pth \
--tadkd_beta 1.828 --tadkd_alpha 3.89 --Tmax 11 --ce_weight 0.2123
