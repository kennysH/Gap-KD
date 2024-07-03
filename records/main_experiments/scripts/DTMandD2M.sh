python train_student_alpha.py --path_t ./save/models/plane10_cifar100/plane10_best.pth --distill kd --seed 3407 --model_s plane2 -r 0.1 -a 0.9 -b 0 --trial 6
python train_student_alpha.py --path_t ./save/models/plane10_cifar100/plane10_best.pth --distill kd --seed 2521 --model_s plane2 -r 0.1 -a 0.9 -b 0 --trial 7
python train_student_alpha.py --path_t ./save/models/plane10_cifar100/plane10_best.pth --distill kd --seed 0 --model_s plane2 -r 0.1 -a 0.9 -b 0 --trial 8
# python train_student_alpha.py --path_t ./save/models/plane10_cifar100/plane10_best.pth --distill kd --model_s plane2 -r 0.1 -a 0.9 -b 0 --trial 4 --adjust rate_decay --Tmax 20 
# python train_student_ta.py --model_s plane2 --path_t ./save/models/plane10_cifar100/plane10_best.pth \
# --path_ta ./save/student_model/plane4_T:plane10_cifar100_rate_decay_kd_r:0.1_a:0.9_b:0.0_1/plane4_best.pth \
# --tadkd_beta 9.090317 --tadkd_alpha 5.875482 --Tmax 25 --ce_weight 3.102796 --tadkd_warmup 33 --adjust none --trial 4
# python train_student_ta.py --model_s plane2 --path_t ./save/models/plane10_cifar10/plane10_best.pth \
# --path_ta ./save/student_model/plane4_T:plane10_cifar10_rate_decay_kd_r:0.1_a:0.9_b:0.0_2/plane4_best.pth \
# --tadkd_beta 9.090317 --tadkd_alpha 5.875482 --Tmax 25 --ce_weight 3.102796 --adjust none
# python train_student_alpha.py --path_t ./save/models/plane10_cifar100/plane10_best.pth --distill kd --model_s plane2 -r 0.1 -a 0.9 -b 0 --trial 5 --adjust rate_decay 


