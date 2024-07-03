import torch
import torch.nn as nn
import torch.nn.functional as F



def mdkd_loss(logits_student, logits_teacher, logits_ta, target, alpha, beta, temperature):
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


class MDKD(nn.Module):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, student, teacher, ta, warm_up):
        super(MDKD, self).__init__()
        # self.ce_loss_weight = opt.DKD_CE_WEIGHT
        self.student = student
        self.teacher = teacher
        self.ta = ta
        self.warmup = warm_up

    def forward(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
            logits_ta, _ = self.ta(image)

   
        # loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) * mdkd_loss(
            logits_student,
            logits_teacher,
            logits_ta,
            target,
            self.alpha,
            self.beta,
            self.temperature,
        )

        return loss_dkd