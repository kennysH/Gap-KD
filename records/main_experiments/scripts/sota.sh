# cifar100,cnn2,
python train_student_ta.py --model_s plane2 --path_t ./save/models/plane10_cifar100_lr_0.01_decay_0.0005_trial_0/plane10_best.pth \
--path_ta ./save/student_model/plane4_T:plane10_cifar100_rate_decay_kd_r:0.1_a:0.9_b:0.0_1/plane4_best.pth \
--tadkd_beta 9.090317 --tadkd_alpha 5.875482 --Tmax 25 --ce_weight 3.102796 --tadkd_warmup 33
# cifar10, cnn2
python train_student_ta.py --model_s plane2 --path_t ./save/models/plane10_cifar10/plane10_best.pth \
--path_ta ./save/student_model/plane4_T:plane10_cifar10_rate_decay_kd_r:0.1_a:0.9_b:0.0_2/plane4_best.pth \
--tadkd_beta 9.090317 --tadkd_alpha 5.875482 --Tmax 25 --ce_weight 3.102796

python train_student_ta.py --model_s plane2 --path_t ./save/models/plane4_cifar100/plane4_best.pth \
--path_ta ./save/student_model/plane4_T:plane10_cifar100_rate_decay_kd_r:0.1_a:0.9_b:0.0_5_Tmax:20/plane4_best.pth \
--tadkd_beta 9.090317 --tadkd_alpha 5.875482 --Tmax 25 --ce_weight 3.102796 --tadkd_warmup 33