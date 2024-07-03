python train_teacher.py --model plane10 --epochs 160 --learning_rate 0.1 --weight_decay 1e-4 --trial 2
python train_teacher.py --model plane8 --epochs 160 --learning_rate 0.1 --weight_decay 1e-4 --trial 2
python train_teacher.py --model plane6 --epochs 160 --learning_rate 0.1 --weight_decay 1e-4 --trial 2
python train_teacher.py --model plane4 --epochs 160 --learning_rate 0.1 --weight_decay 1e-4 --trial 2
python train_teacher.py --model plane2 --epochs 160 --learning_rate 0.1 --weight_decay 1e-4 --trial 2
python train_student_alpha.py --model_s plane8 --path_t ./save/models/plane10_cifar100_lr_0.1_decay_0.0001_trial_2/plane10_best.pth -r 0.1 -a 0.9 -b 0 --trial 2
python train_student_alpha.py --model_s plane6 --path_t ./save/models/plane10_cifar100_lr_0.1_decay_0.0001_trial_2/plane10_best.pth -r 0.1 -a 0.9 -b 0 --trial 2
python train_student_alpha.py --model_s plane4 --path_t ./save/models/plane10_cifar100_lr_0.1_decay_0.0001_trial_2/plane10_best.pth -r 0.1 -a 0.9 -b 0 --trial 2
python train_student_alpha.py --model_s plane2 --path_t ./save/models/plane10_cifar100_lr_0.1_decay_0.0001_trial_2/plane10_best.pth -r 0.1 -a 0.9 -b 0 --trial 2
