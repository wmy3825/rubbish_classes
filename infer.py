import torch
from torchvision import transforms
from models import resnet
def try_gpu(i=0):
    return torch.device(f"cpu")


torch.serialization.add_safe_globals({'Residual': resnet()})
device=try_gpu()
print('我们使用的是',device)

def infer(image,model_file):
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为 224x224 像素
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化处理
])
    image = transform(image)
    image = torch.unsqueeze(image, 0)  # 图像数据展平
    image = image.to(device)
    model = torch.load(model_file, map_location=torch.device('cpu'))
    model = model.to(device)
    model.eval()
    output = model(image)
    prob = torch.nn.functional.softmax(output, dim=1)[0] * 100
    _, indices = torch.sort(output, descending=True)
    with open('data/label2.txt', 'r', encoding = 'UTF 8') as f:
        classes = [line.strip() for line in f.readlines()]
    labels = [(classes[idx], prob[idx].item()) for idx in indices[0][:1]]
    return labels
