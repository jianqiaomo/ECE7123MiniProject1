## Load trained model. Adapted from Brightspace
import torch
import torchvision
import torchvision.transforms as transforms
from project1_model import project1_model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = project1_model().to(device)
model_path = './pt/project1_model.pt'
print(model.load_state_dict(torch.load(model_path, map_location=device), strict=False))

## Report accuracy on test dataset:
transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
