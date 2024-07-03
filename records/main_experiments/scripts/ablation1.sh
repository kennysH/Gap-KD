# kd
python train_student_alpha.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet20 -r 0.1 -a 0.9 -b 0 --trial 1 --adjust rate_decay 
python train_student_alpha.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet20 -r 0.1 -a 0.9 -b 0 --trial 1 
python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet32 -r 0.1 -a 0.9 -b 0 --trial 1 --adjust rate_decay
python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet32 -r 0.1 -a 0.9 -b 0 --trial 1 
python train_student_alpha.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill kd --model_s wrn_16_2 -r 0.1 -a 0.9 -b 0 --trial 1 --adjust rate_decay 
python train_student_alpha.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill kd --model_s wrn_16_2 -r 0.1 -a 0.9 -b 0 --trial 1
python train_student_alpha.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill kd --model_s wrn_40_1 -r 0.1 -a 0.9 -b 0 --trial 1 --adjust rate_decay
python train_student_alpha.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill kd --model_s wrn_40_1 -r 0.1 -a 0.9 -b 0 --trial 1
python train_student_alpha.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s ShuffleV1 -r 0.1 -a 0.9 -b 0 --trial 1 --adjust rate_decay
python train_student_alpha.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s ShuffleV1 -r 0.1 -a 0.9 -b 0 --trial 1 
python train_student_alpha.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s ShuffleV2 -r 0.1 -a 0.9 -b 0 --trial 1 --adjust rate_decay
python train_student_alpha.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s ShuffleV2 -r 0.1 -a 0.9 -b 0 --trial 1
# # FitNet
# python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill hint --model_s resnet20 -a 0 -b 100 --trial 1 --adjust rate_decay 
# # AT
# python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill attention --model_s resnet20 -a 0 -b 1000 --trial 1 --adjust rate_decay 
# # SP
# python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill similarity --model_s resnet20 -a 0 -b 3000 --trial 1 --adjust rate_decay 
# # CC
# python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill correlation --model_s resnet20 -a 0 -b 0.02 --trial 1 --adjust rate_decay 
# # VID
# python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill vid --model_s resnet20 -a 0 -b 1 --trial 1 --adjust rate_decay 
# # RKD
# python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill rkd --model_s resnet20 -a 0 -b 1 --trial 1 --adjust rate_decay 
# # PKT
# python train_student_alpha.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill pkt --model_s resnet20 -a 0 -b 30000 --trial 1 --adjust rate_decay 
# python train_student_alpha.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill pkt --model_s resnet20 -a 0 -b 30000 --trial 1 
# python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill pkt --model_s resnet20 -a 0 -b 30000 --trial 1 --adjust rate_decay 
# python train_student_alpha.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill pkt --model_s resnet20 -a 0 -b 30000 --trial 1 

# # AB
# python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill abound --model_s resnet20 -a 0 -b 1 --trial 1 --adjust rate_decay 
# # FT
# python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill factor --model_s resnet20 -a 0 -b 200 --trial 1 --adjust rate_decay 
# # FSP
# python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill fsp --model_s resnet20 -a 0 -b 50 --trial 1 --adjust rate_decay 
# # NST
# python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill nst --model_s resnet20 -a 0 -b 50 --trial 1 --adjust rate_decay 
# # CRD
# python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill crd --model_s resnet20 -a 0 -b 0.8 --trial 1 --adjust rate_decay 

# # CRD+KD
# python train_student_alpha.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill crd --model_s resnet20 -a 1 -b 0.8 --trial 1 --adjust rate_decay 
