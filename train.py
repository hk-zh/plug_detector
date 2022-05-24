import torch.optim as optim
import torch
import torch.nn as nn
from cnnNet import Net
from torch.utils.data import Dataset, DataLoader
from dataloader import MyDataset, ToTensor
from torchvision import transforms, utils

net = Net()

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-4)

train_dataset = MyDataset(csv_file='data/train_labels.csv',
                                root_dir='data/',
                                transform=transforms.Compose([
                                    ToTensor()
                                ]))

validation_dataset = MyDataset(csv_file='data/validation_labels.csv',
                                root_dir='data/',
                                transform=transforms.Compose([
                                    ToTensor()
                                ]))

train_dataloader = DataLoader(train_dataset, batch_size=8,
                        shuffle=True, num_workers=0)

validation_dataloader = DataLoader(validation_dataset, batch_size=4,
                        shuffle=True, num_workers=0)

best_val_loss = 1e10
for epoch in range(1000):  # loop over the dataset multiple times

    running_loss = 0.0
    num = 0
    net.train()
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data['image'], data['labels']

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        num += 1
    print(f'[{epoch + 1}] loss: {running_loss / num:.3f}')

    if epoch % 5 == 0:
        net.eval()
        loss = 0.0
        num = 0
        for i, data in enumerate(validation_dataloader, 0):
            inputs, labels = data['image'], data['labels']
            # forward
            outputs = net(inputs)
            loss += float(criterion(outputs, labels))
            num += 1

        print('validation loss =', loss/num)
        if best_val_loss > loss / num:
            best_val_loss = loss / num
            print('save the best model')
            torch.save(net.state_dict(), 'model/best.pth')


print('Finished Training')
