"""
the general training framework
"""

from __future__ import print_function

import os
import argparse
import socket
import time
import wandb
import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import random
from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample

from helper.util import adjust_learning_rate

from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss
from crd.criterion import CRDLoss

from helper.loops import train_distill as train, validate
from helper.pretrain import init


def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
    # parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'attention', 'similarity',
                                                                      'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                      'rkd', 'pkt', 'abound', 'factor', 'nst'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for other losses')

    # adjust gamma
    parser.add_argument('--adjust_gamma', type=str, default='None', help="Adjustment strategy for gamma ('none', 'linear', 'exponential', 'logarithmic', 'sigmoid', 'piecewise')")
    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # hint layer
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])

    # 添加分阶段训练的参数
    parser.add_argument('--path_t1', type=str, default=None, help='teacher model snapshot for stage 1')
    parser.add_argument('--path_t2', type=str, default=None, help='teacher model snapshot for stage 2')
    parser.add_argument('--path_t3', type=str, default=None, help='teacher model snapshot for stage 3')

    parser.add_argument('--lr_stage1', type=float, default=0.05, help='learning rate for stage 1')
    parser.add_argument('--lr_stage2', type=float, default=0.005, help='learning rate for stage 2')
    parser.add_argument('--lr_stage3', type=float, default=0.0005, help='learning rate for stage 3')

    parser.add_argument('--early_stop_threshold_stage1', type=int, default=5, help='early stop threshold for stage 1')
    parser.add_argument('--early_stop_threshold_stage2', type=int, default=5, help='early stop threshold for stage 2')
    parser.add_argument('--early_stop_threshold_stage3', type=int, default=5, help='early stop threshold for stage 3')

    parser.add_argument('--epochs_stage1', type=int, default=60, help='number of epochs for stage 1')
    parser.add_argument('--epochs_stage2', type=int, default=60, help='number of epochs for stage 2')
    parser.add_argument('--epochs_stage3', type=int, default=60, help='number of epochs for stage 3')

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/student_model'
        opt.tb_path = '/path/to/my/student_tensorboards'
    else:
        opt.model_path = './save/student_model'
        opt.tb_path = './save/student_tensorboards'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # opt.model_t = get_teacher_name(opt.path_t)

    opt.model_name = 'Step_S:{}_{}_{}_r:{}_a:{}_b:{}_{}'.format(opt.model_s,  opt.dataset, opt.distill,
                                                                opt.gamma, opt.alpha, opt.beta, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model

def train_stage(i, model_s, model_t, train_loader, val_loader, epochs, early_stop_threshold, opt, feat_t, feat_s, module_list, trainable_list, n_data):
    best_acc = 0
    no_improve_epochs = 0
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif opt.distill == 'attention':
        criterion_kd = Attention()
    elif opt.distill == 'nst':
        criterion_kd = NSTLoss()
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    elif opt.distill == 'rkd':
        criterion_kd = RKDLoss()
    elif opt.distill == 'pkt':
        criterion_kd = PKT()
    elif opt.distill == 'kdsvd':
        criterion_kd = KDSVD()
    elif opt.distill == 'correlation':
        criterion_kd = Correlation()
        embed_s = LinearEmbed(feat_s[-1].shape[1], opt.feat_dim)
        embed_t = LinearEmbed(feat_t[-1].shape[1], opt.feat_dim)
        module_list.append(embed_s)
        module_list.append(embed_t)
        trainable_list.append(embed_s)
        trainable_list.append(embed_t)
    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        # add this as some parameters in VIDLoss need to be updated
        trainable_list.append(criterion_kd)
    elif opt.distill == 'abound':
        s_shapes = [f.shape for f in feat_s[1:-1]]
        t_shapes = [f.shape for f in feat_t[1:-1]]
        connector = Connector(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(connector)
        init_trainable_list.append(model_s.get_feat_modules())
        criterion_kd = ABLoss(len(feat_s[1:-1]))
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, opt)
        # classification
        module_list.append(connector)
    elif opt.distill == 'factor':
        s_shape = feat_s[-2].shape
        t_shape = feat_t[-2].shape
        paraphraser = Paraphraser(t_shape)
        translator = Translator(s_shape, t_shape)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(paraphraser)
        criterion_init = nn.MSELoss()
        init(model_s, model_t, init_trainable_list, criterion_init, train_loader, opt)
        # classification
        criterion_kd = FactorTransfer()
        module_list.append(translator)
        module_list.append(paraphraser)
        trainable_list.append(translator)
    elif opt.distill == 'fsp':
        s_shapes = [s.shape for s in feat_s[:-1]]
        t_shapes = [t.shape for t in feat_t[:-1]]
        criterion_kd = FSP(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(model_s.get_feat_modules())
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, opt)
        # classification training
        pass
    else:
        raise NotImplementedError(opt.distill)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss\



    # 第一阶段
    if i == 1:
        # 检查 CUDA 可用性并将模型转移到 GPU
        if torch.cuda.is_available():
            model_t.cuda()
            model_s.cuda()
            criterion_list.cuda()
            cudnn.benchmark = True
        # 验证教师模型 1 的准确率
        teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
        print('Teacher 1 accuracy: ', teacher_acc)
        # 设置优化器和训练
        optimizer = optim.SGD(trainable_list.parameters(), lr=opt.lr_stage1, momentum=opt.momentum,
                              weight_decay=opt.weight_decay)

    elif i == 2:
        if torch.cuda.is_available():
            model_t.cuda()
        # 验证教师模型 2 的准确率
        teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
        print('Teacher 2 accuracy: ', teacher_acc)
        # 设置优化器和训练
        optimizer = optim.SGD(trainable_list.parameters(), lr=opt.lr_stage2, momentum=opt.momentum,
                              weight_decay=opt.weight_decay)

    elif i == 3:
        if torch.cuda.is_available():
            model_t.cuda()
        # 验证教师模型 3 的准确率
        teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
        print('Teacher 1 accuracy: ', teacher_acc)
        # 设置优化器和训练
        optimizer = optim.SGD(trainable_list.parameters(), lr=opt.lr_stage3, momentum=opt.momentum,
                              weight_decay=opt.weight_decay)


    for epoch in range(1, epochs + 1):
        print("==> training...")
        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))


        if i == 1:
            # logger.log_value('train_acc', train_acc, epoch)
            wandb.log({'train_acc1': train_acc, 'epoch1': epoch})
            # logger.log_value('train_loss', train_loss, epoch)
            wandb.log({'train_loss1': train_loss, 'epoch1': epoch})

            test_acc, test_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)

            # logger.log_value('test_acc', test_acc, epoch)
            wandb.log({'test_acc1': test_acc, 'epoch1': epoch})
            # logger.log_value('test_loss', test_loss, epoch)
            wandb.log({'test_acc_top5_1': test_acc_top5, 'epoch1': epoch})
            # logger.log_value('test_acc_top5', test_acc_top5, epoch)
            wandb.log({'test_loss1': test_loss, 'epoch1': epoch})

        if i == 2:
            # logger.log_value('train_acc', train_acc, epoch)
            wandb.log({'train_acc2': train_acc, 'epoch2': epoch})
            # logger.log_value('train_loss', train_loss, epoch)
            wandb.log({'train_loss2': train_loss, 'epoch2': epoch})

            test_acc, test_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)
            # logger.log_value('test_acc', test_acc, epoch)
            wandb.log({'test_acc2': test_acc, 'epoch2': epoch})
            # logger.log_value('test_loss', test_loss, epoch)
            wandb.log({'test_acc_top5_2': test_acc_top5, 'epoch2': epoch})
            # logger.log_value('test_acc_top5', test_acc_top5, epoch)
            wandb.log({'test_loss2': test_loss, 'epoch2': epoch})

        if i == 3:
            # logger.log_value('train_acc', train_acc, epoch)
            wandb.log({'train_acc3': train_acc, 'epoch3': epoch})
            # logger.log_value('train_loss', train_loss, epoch)
            wandb.log({'train_loss3': train_loss, 'epoch3': epoch})

            test_acc, test_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)
            # logger.log_value('test_acc', test_acc, epoch)
            wandb.log({'test_acc3': test_acc, 'epoch3': epoch})
            # logger.log_value('test_loss', test_loss, epoch)
            wandb.log({'test_acc_top5_3': test_acc_top5, 'epoch3': epoch})
            # logger.log_value('test_acc_top5', test_acc_top5, epoch)
            wandb.log({'test_loss3': test_loss, 'epoch3': epoch})

        # save the best model
        if test_acc > best_acc:
            wandb.run.summary["best_accuracy"] = test_acc
            best_acc = test_acc
            # no_improve_epochs = 0

        # else:
        #     no_improve_epochs += 1
        #     if no_improve_epochs >= early_stop_threshold:
        #         break  # 早停

        # if no_improve_epochs >= early_stop_threshold:
        #     print(f"Early stopping at epoch {epoch} with test accuracy {best_acc}")
        #     break
def main():
    best_acc = 0
    # 设置种子
    seed_value = 0
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    best_acc = 0

    opt = parse_option()

    # tensorboard logger
    #logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    wandb.init(project="distill_image", name="steps_distill")
    wandb.config.update(opt)
    # dataloader
    if opt.dataset == 'cifar100':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
        else:
            train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
        n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model_s = model_dict[opt.model_s](num_classes=n_cls)
    model_t1 = load_teacher(opt.path_t1, n_cls)
    model_t2 = load_teacher(opt.path_t2, n_cls)
    model_t3 = load_teacher(opt.path_t3, n_cls)
    data = torch.randn(2, 3, 32, 32)
    # model_t.eval()
    model_t1.eval()
    model_t2.eval()
    model_t3.eval()
    model_s.eval()
    # feat_t, _ = model_t(data, is_feat=True)
    # feat_s, _ = model_s(data, is_feat=True)
    feat_t1, _ = model_t1(data, is_feat=True)
    feat_t2, _ = model_t2(data, is_feat=True)
    feat_t3, _ = model_t3(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)



    # 阶段 1

    module_list.append(model_t1)

    train_stage(1, model_s, model_t1, train_loader, val_loader, opt.epochs_stage1,
                opt.early_stop_threshold_stage1, opt, feat_t1, feat_s, module_list, trainable_list, n_data)
    # 在进入下一阶段之前，从 module_list 中移除当前教师模型
    # 假设 module_list 包含多个模块，包括 model_t1
    new_module_list = nn.ModuleList([m for m in module_list if m != model_t1])
    module_list = new_module_list

    # 阶段 2

    module_list.append(model_t2)

    # 设置优化器和训练
    train_stage(2, model_s, model_t2, train_loader, val_loader, opt.epochs_stage2,
                opt.early_stop_threshold_stage2, opt, feat_t1,feat_s, module_list, trainable_list, n_data)
    # 同样，在进入下一阶段之前，移除当前教师模型
    # 假设 module_list 包含多个模块，包括 model_t1
    new_module_list = nn.ModuleList([m for m in module_list if m != model_t2])
    module_list = new_module_list

    # 阶段 3
    module_list.append(model_t3)
    train_stage(3, model_s, model_t3, train_loader, val_loader, opt.epochs_stage3,
                opt.early_stop_threshold_stage3, opt, feat_t1,feat_s, module_list, trainable_list, n_data)
    # 训练结束后，如果需要，可以从 module_list 中移除最后一个教师模型
    # 假设 module_list 包含多个模块，包括 model_t1

    # save model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()

