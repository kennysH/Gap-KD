"""
the general training framework
"""

from __future__ import print_function

import os
import argparse
import socket
import sys
import time
import wandb

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
from dataset.cifar10 import get_cifar10_dataloaders, get_cifar10_dataloaders_sample
from helper.util import adjust_learning_rate

from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss, MDKD
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss
from crd.criterion import CRDLoss

from helper.loops import train_distill as train, validate
from helper.pretrain import init
from helper.util import AverageMeter, accuracy
import torch.nn.functional as F
os.environ["WANDB_MODE"] = "offline"
def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--seed', type=int, default=3407, help='seed')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
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
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100','cifar10'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='plane2',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'plane2', 'plane3', 'plain4', 'plane5', 'plane6', 'plane7', 'plane8', 'plane9', 'plane10'])
    parser.add_argument('--path_t', type=str, default='./save/models/plane10_cifar100/plane10_best.pth', help='teacher model snapshot')
    parser.add_argument('--path_ta', type=str, default='./save/models/plane8_cifar100/plane8_best.pth',help='teacher assistant model snapshot')
    # adjust 
    parser.add_argument('--adjust', type=str, default='rate_decay', help="Adjustment strategy for temperature('none', 'linear', 'exponential', 'logarithmic', 'sigmoid', 'piecewise', 'rate_decay')")
    # distillation
    parser.add_argument('--distill', type=str, default='gapkd', choices=['kd', 'hint', 'attention', 'similarity',
                                                                       'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                       'rkd', 'pkt', 'abound', 'factor', 'nst', 'mdkd'])
    parser.add_argument('--trial', type=str, default='5', help='trial id')

    #parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    #parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for KD')
    #parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for other losses')

    # KL distillation
    # parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # hint layer
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])

    # GapKD distillation
    parser.add_argument('--ce_weight', default=1.0, type=float, help='weight for GapKD_CE')   
    parser.add_argument('--gap_alpha', default=1.0, type=float, help='weight for GapKD_TCKD')
    parser.add_argument('--gap_beta', default=2.0, type=float, help='weight for GapKD_NCKD')
    parser.add_argument('--gap_T', default=4.0, type=float, help='temperature for GapKD')
    parser.add_argument('--gap_warmup', default=30, type=int, help='warmup epochs for GapKD')

     # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # temperature decay
    parser.add_argument('--Tmax', default=20, type=int, help='the maximum temperature')
    parser.add_argument('--Tmin', default=1, type=int, help='the minimum temperature')
    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'plane2', 'plane4', 'plane6', 'plane8','plane10']:
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

    opt.model_t = get_teacher_name(opt.path_t)
    opt.model_ta = get_teacher_name(opt.path_ta)

    opt.model_name = 'GapKD:{}_T:{}_TA:{}_{}_r:{}_a:{}_b:{}_{}_{}-{}'.format(opt.model_s, opt.model_t, opt.model_ta, opt.dataset,
                                                                opt.ce_weight, opt.gap_alpha, opt.gap_beta, opt.gap_warmup, opt.adjust, opt.trial)

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

def load_checkpoint(model, checkpoint_path):
	"""
	Loads weights from checkpoint
	:param model: a pytorch nn student
	:param str checkpoint_path: address/path of a file
	:return: pytorch nn student with weights loaded from checkpoint
	"""
	model_t = model_dict[model](num_classes=100)
	model_t.load_state_dict(torch.load(checkpoint_path)['model'])
	return model_t

def gapkd_loss(logits_student, logits_teacher, logits_ta, target, alpha, beta, temperature):
    # 获取目标类别的掩码
    gt_mask = _get_gt_mask(logits_student, target)
    # 获取非目标类别的掩码
    other_mask = _get_other_mask(logits_student, target)
    # 对学生模型的输出进行softmax操作并除以温度参数
    pred_student = F.softmax(logits_student / temperature, dim=1)
    # 对教师模型的输出进行softmax操作并除以温度参数
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    # 对助教模型的输出进行softmax操作并除以温度参数
    pred_ta = F.softmax(logits_ta / temperature, dim=1)
    # 对学生模型的预测结果进行掩码操作
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    # 对教师模型的预测结果进行掩码操作
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    # 对助教模型的预测结果进行掩码操作
    pred_ta = cat_mask(pred_ta, gt_mask, other_mask)
    # 对学生模型的预测结果取对数
    log_pred_student = torch.log(pred_student)
    # 计算教师模型和学生模型在目标类别上的KL散度
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    # 对助教模型的输出进行softmax操作并减去一个大数（通过gt_mask实现）
    pred_ta_part2 = F.softmax(
        logits_ta / temperature - 1000.0 * gt_mask, dim=1
    )
    # 对学生模型的输出进行log_softmax操作并减去一个大数（通过gt_mask实现）
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    # 计算助教模型和学生模型在非目标类别上的KL散度
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_ta_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    # 返回加权后的总损失
    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

def train_gapkd(epoch, train_loader, module_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()
    module_list[1].eval()

    model_s = module_list[0]
    model_t = module_list[-1]
    model_ta = module_list[1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        input, target, index = data
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()

        # _, logits_s = model_s(input, is_feat=True, preact=False)
        # _, logits_t = model_t(input, is_feat=True, preact=False)
        # _, logits_ta = model_ta(input, is_feat=True, preact=False)

        logits_s = model_s(input)
        logits_t = model_t(input)
        logits_ta = model_ta(input)

        loss_ce = opt.ce_weight * F.cross_entropy(logits_s, target)
        T = opt.gapkd_T
        loss_dkd = min(epoch / opt.gap_warmup, 1.0) * gapkd_loss(
            logits_s,
            logits_t,
            logits_ta,
            target,
            opt.gap_alpha,
            opt.gap_beta,
            T,
        )
        loss = loss_ce+loss_dkd

        acc1, acc5 = accuracy(logits_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg

def linear_decay(epoch, total_epochs, Tmax, Tmin):
    return Tmax - (Tmax - Tmin) * (epoch / total_epochs)

def exponential_decay(epoch, total_epochs, Tmax, Tmin):
    return Tmax * ((Tmin/Tmax) ** (epoch / total_epochs))

def logarithmic_decay(epoch, total_epochs, Tmax, Tmin):
    return Tmax - (Tmax - Tmin) * np.log(epoch + 1) / np.log(total_epochs + 1)

def piecewise_decay(epoch, total_epochs, Tmax, Tmin):
    if epoch < total_epochs / 2:
        return Tmax - (Tmax - Tmin) * 2 * (epoch / total_epochs)
    else:
        return Tmin
    
def sigmoid_decay(epoch, total_epochs, Tmax, Tmin):
    return Tmin + (Tmax - Tmin) / (1 + np.exp((epoch - total_epochs / 2) / (0.1 * total_epochs)))

def rate_decay(epoch, total_epochs, Tmax, Tmin):
    decay_rate = (Tmin / Tmax) ** (1 / (total_epochs - 1))
    return Tmax * (decay_rate ** (epoch - 1))

def main():
    opt = parse_option()
    # 设置种子

    seed_value = opt.seed
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    best_acc = 0



    # tensorboard logger
    #logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    wandb.init(project="GapKD", name="train_student")
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

    elif opt.dataset == 'cifar10':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_cifar10_dataloaders_sample(batch_size=opt.batch_size,
                                                                              num_workers=opt.num_workers,
                                                                              k=opt.nce_k,
                                                                              mode=opt.mode)
        else:
            train_loader, val_loader, n_data = get_cifar10_dataloaders(batch_size=opt.batch_size,
                                                                       num_workers=opt.num_workers,
                                                                       is_instance=True)
        n_cls = 10

    else:
        raise NotImplementedError(opt.dataset)

    # model
    model_t = load_teacher(opt.path_t, n_cls)
    model_s = model_dict[opt.model_s](num_classes=n_cls)
    model_ta = load_teacher(opt.path_ta, n_cls)

    data = torch.randn(2, 3, 32, 32)
    model_t.eval()
    model_s.eval()
    model_ta.eval()
    # feat_t, _ = model_t(data, is_feat=True)
    # feat_s, _ = model_s(data, is_feat=True)
    # feat_ta, _ = model_ta(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    #criterion_div = DistillKL(opt.kd_T)


    criterion_list = nn.ModuleList([])
    #criterion_list.append(criterion_cls)    # classification loss
    # criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    # criterion_list.append(criterion_kd)     # other knowledge distillation loss

    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_ta)
    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # validate teacher accuracy
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: ', teacher_acc)

    # routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        if opt.adjust == 'linear':
            opt.kd_T = linear_decay(epoch, opt.epochs, opt.Tmax, opt.Tmin)
        elif opt.adjust == 'exponential':
            opt.kd_T = exponential_decay(epoch, opt.epochs, opt.Tmax, opt.Tmin)
        elif opt.adjust == 'logarithmic':
            opt.kd_T = logarithmic_decay(epoch, opt.epochs, opt.Tmax, opt.Tmin)
        elif opt.adjust == 'piecewise':
            opt.kd_T = piecewise_decay(epoch, opt.epochs, opt.Tmax, opt.Tmin)
        elif opt.adjust == 'sigmoid':
            opt.kd_T = sigmoid_decay(epoch, opt.epochs, opt.Tmax, opt.Tmin)
        elif opt.adjust == 'rate_decay':
            opt.kd_T = rate_decay(epoch, opt.epochs, opt.Tmax, opt.Tmin)
        else:
            opt.kd_T = opt.kd_T

        time1 = time.time()
        train_acc, train_loss = train_gapkd(epoch, train_loader, module_list, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        #logger.log_value('train_acc', train_acc, epoch)
        wandb.log({'train_acc': train_acc, 'epoch': epoch})
        #logger.log_value('train_loss', train_loss, epoch)
        wandb.log({'train_loss': train_loss, 'epoch': epoch})

        test_acc, test_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)

        #logger.log_value('test_acc', test_acc, epoch)
        wandb.log({'test_acc': test_acc, 'epoch': epoch})
        #logger.log_value('test_loss', test_loss, epoch)
        wandb.log({'test_acc_top5': test_acc_top5, 'epoch': epoch})
        #logger.log_value('test_acc_top5', test_acc_top5, epoch)
        wandb.log({'test_loss': test_loss, 'epoch': epoch})
        wandb.log({'T': opt.gap_T, 'epoch': epoch})

        # save the best model
        if test_acc > best_acc:
            wandb.run.summary["best_accuracy"] = test_acc
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('saving the best model!')
            torch.save(state, save_file)

            # Save the accuracy in a text file
            acc_save_path = os.path.join(opt.save_folder, '{}_best_acc.txt'.format(opt.model_s))
            with open(acc_save_path, 'w') as acc_file:
                acc_file.write('Best Accuracy: {:.4f}'.format(best_acc))

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'accuracy': test_acc,
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch. 
    print('best accuracy:', best_acc)

    # save model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)

    wandb.finish()

if __name__ == '__main__':
    main()

