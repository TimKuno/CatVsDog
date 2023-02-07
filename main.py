import torch
from torchvision import transforms
from PIL import Image
from os import listdir
import random
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import shutil

# data preprocessing
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    normalize
])

train_dir = 'train/'
test_dir = 'test/'

# create train / test (80/20) split
if len(listdir(test_dir)) == 0:
    print("Start create train / test split")
    files = listdir(train_dir)
    for _ in range(int(len(listdir(train_dir)) * 0.2)):
        file = random.choice(files)  # randomize
        files.remove(file)
        shutil.move(train_dir + file, test_dir)
    print("Created train / test split")

# load / create train data
# TARGET: [isCat, isDog]
train_data_list = []
train_data = []
target_list = []
files = listdir(train_dir)
for i in range(len(listdir(train_dir))):
    file = random.choice(files)  # randomize
    files.remove(file)
    img = Image.open(train_dir + file)
    img_tensor = transform(img)  # (3, 256, 256)
    train_data_list.append(img_tensor)
    isCat = 1 if 'cat' in file else 0
    isDog = 1 if 'dog' in file else 0
    target = [isCat, isDog]
    target_list.append(target)
    if len(train_data_list) >= 64:
        train_data.append((torch.stack(train_data_list), target_list))  # create batch
        train_data_list = []
        target_list = []
        print(f'Loaded batch {len(train_data)} of {int(len(listdir(train_dir))/64)}')


# defining the model
class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=5)
        self.conv3 = nn.Conv2d(12, 18, kernel_size=5)
        self.conv4 = nn.Conv2d(18, 24, kernel_size=5)
        self.fc1 = nn.Linear(3456, 1024)
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = x.view(-1, 3456)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)


model = Cnn()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# defining the training of the model
def train(epoch):
    model.train()
    batch_id = 0
    for data, target in train_data:
        target = torch.Tensor(target)
        data = Variable(data)
        target = Variable(target)
        optimizer.zero_grad()
        out = model(data)
        loss = F.binary_cross_entropy(out, target)
        loss.backward()
        optimizer.step()
        print(f'Train Epoch: {epoch} [{100 * batch_id / len(train_data):.0f}%]'
              f'\t Loss: {loss.item():.6f}')
        batch_id += 1


#  defining the testing of the modell
def test():
    model.eval()
    correct = 0
    for _ in range(len(listdir(test_dir))):
        files = listdir(test_dir)
        file = random.choice(files)
        img = Image.open(test_dir + file)
        img_eval_tensor = transform(img)
        img_eval_tensor.unsqueeze_(0)
        data = Variable(img_eval_tensor)
        out = model(data)
        if int(out.data.max(1, keepdim=True)[1]) == 0 and 'cat' in file:
            correct += 1
        if int(out.data.max(1, keepdim=True)[1]) == 1 and 'dog' in file:
            correct += 1
    print(f'Accuracy: {correct / len(listdir(test_dir))}')


# train the model in order to test it afterwards for the evaluation
for epoch in range(1, 20):
    train(epoch)
test()  # it would also be possible to evaluate the model after each epoch
