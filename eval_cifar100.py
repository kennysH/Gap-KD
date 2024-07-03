import torch
from models import model_dict
from dataset.cifar100 import get_cifar100_dataloaders
from helper.loops import validate


class DummyOpt:
    def __init__(self):
        self.print_freq = 100  # Or any other appropriate value
def load_model(model_path, model_name, num_classes=100):
    model = model_dict[model_name](num_classes=num_classes)
    model.load_state_dict(torch.load(model_path)['model'])
    return model

def evaluate(model, val_loader, device):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    opt = DummyOpt()
    with torch.no_grad():
        acc, acc_top5, _ = validate(val_loader, model, criterion, opt)
    return acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "save/models/resnet110_vanilla/ckpt_epoch_240.pth"  # 更改为您的模型路径
    model_name = "resnet110"  # 根据您的模型更改

    # 加载模型
    model = load_model(model_path, model_name)
    model = model.to(device)  # 移动模型到正确的设备

    # 准备数据
    _, val_loader = get_cifar100_dataloaders(batch_size=64, num_workers=8)
    # 评估模型
    acc = evaluate(model, val_loader, device)
    print(f"Model Accuracy on CIFAR100 Test Set: {acc:.2f}%")

if __name__ == "__main__":
    main()
