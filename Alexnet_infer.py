import torch
from torchvision import transforms
from models import AlexNet
def try_gpu(i=0):
    # if torch.cuda.is_available():
    #     return torch.device("cuda:{}".format(i))
    return torch.device(f"cpu")
torch.serialization.add_safe_globals({'Residual': AlexNet})
device=try_gpu()
print('我们使用的是',device)

def Alexnet_infer(image,model_file):
    transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               transforms.Resize(   (227,227))])
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
