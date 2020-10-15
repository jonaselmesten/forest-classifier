#from google.colab import drive
#drive.mount('/content/drive')

#!/usr/bin/python
# !pip install split-folders
# !pip install split-folders tqdm
import torch
import pandas as pd
import numpy as np
from torchvision import transforms, datasets, utils
import torch.nn as nn
import matplotlib.pyplot as plt

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path

        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

#Preprocessing

data_transform = transforms.Compose([
        transforms.CenterCrop (100),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

#splitfolders.ratio("/content/drive/My Drive/CNN_lager_dataset/dataset",output="/content/drive/My Drive/CNN_lager_dataset/dataset-division",seed=1337, ratio=(.7, .3), group_prefix=None)

train_dataset = datasets.ImageFolder(root='/content/drive/My Drive/CNN_lager_dataset/dataset-division/train',
                                           transform=data_transform)
val_dataset = datasets.ImageFolder(root='/content/drive/My Drive/CNN_lager_dataset/dataset-division/val',
                                           transform=data_transform)
dataset_unlabeled = datasets.ImageFolder(root='/content/drive/My Drive/CNN_lager_dataset/images_tobeclassified',
                                           transform=data_transform)

# my_dataset -> image_loade-> do prediction
my_dataset = ImageFolderWithPaths('/content/drive/My Drive/CNN_lager_dataset/images_tobeclassified',
                                  transform=data_transform)
# for item in my_dataset:
#   print(item)

#train_size = int(0.7 * len(tree_dataset))
#test_size = len(tree_dataset) - train_size
#train_dataset, test_dataset = torch.utils.data.random_split(tree_dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=64, shuffle=True,
                                             num_workers=4)
# print(train_loader)
# print(len(train_loader))

val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=64, shuffle=False,
                                             num_workers=4)

image_loader = torch.utils.data.DataLoader(my_dataset,
                                              batch_size=1, shuffle=False,
                                              num_workers=0)
print(image_loader)

#claseeee = tree_dataset.classes
#train_classes = train_dataset.classes
#val_classes = val_dataset.classes
#train and val classes is same
classes = train_dataset.classes
print(classes)

# Hyperparameters
num_epochs = 5
num_classes = len(classes)
for item in classes:
  print(item)
batch_size = 8
learning_rate = 0.001
print(num_classes)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def valid_imshow_data(data):
    data = np.asarray(data)
    if data.ndim == 2:
        return True
    elif data.ndim == 3:
        if 3 <= data.shape[2] <= 4:
            return True
        else:
            print('The "data" has 3 dimensions but the last dimension '
                  'must have a length of 3 (RGB) or 4 (RGBA), not "{}".'
                  ''.format(data.shape[2]))
            return False
    else:
        print('To visualize an image the data must be 2 dimensional or '
              '3 dimensional, not "{}".'
              ''.format(data.ndim))
        return  False

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.permute(1,2,0)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def image_grid():
    images, labels = next(iter(train_loader))
    out = utils.make_grid(images)
    imshow(out, title=[classes[x] for x in labels])

def model_train():
    model.train()
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)
            #print(images[0].shape)

            # Run the forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            print("fuck", i, epoch)
            #if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))
        # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
        #               .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
        #                       (correct / total) * 100))
    # plot accuracy and loss of training dataset 
    # x = range(num_epochs)
    # line_acc,=plt.plot(x, acc_list, label='train_accuracy')
    # line_loss,=plt.plot(x, loss_list, color='red', linewidth=1.0, label='train_loss')
    # plt.legend(handles=[line_acc, line_loss], labels=['train_acc', 'train_loss'])
    # plt.xlabel("epoch")
    # plt.ylabel("loss and accuracy")
    # plt.show()


def model_eval():
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the {} test images: {} %'.format(len(val_dataset), (correct / total) * 100))
        #print(predicted.squeeze())

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(25 * 25 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

model = ConvNet()
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if torch.cuda.is_available():
    model = model.to(device)
    criterion = criterion.to(device)

model_train()

model_eval()
# Save the model and plot
#torch.save(model.state_dict(), './model/conv_net_model.ckpt')

def create_predictions():

    # record prediction, accuracy and path for each image
    outfile = []

    #disables some calculations not needed
    model.eval()

    with torch.no_grad():
        for images, labels, paths in image_loader:
            #make images and labels be on GPU instead on CPU
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            print(output)
            #pred = torch.max(output.data, 1)

            #make list of percentages of predictions
            sm = nn.Softmax(dim=1)
            percent_tensor = sm(output).cpu()
            percent_list = list(percent_tensor.numpy())

            #converts output from model to class index
            softmax = torch.exp(output).cpu()
            prob = list(softmax.numpy())
            prediction_index_tuple = np.argmax(prob, axis=1)
            prediction_index = prediction_index_tuple[0]


            percentage = "{:.2%}".format(percent_list[0][prediction_index])
            #print(percentage)

            #print(prediction_index[0])
            #print(classes[prediction_index[0]])
            #make paths from tuple to str
            paths = "".join(paths[0])
            #print(paths)
            outfile.append((classes[prediction_index], percentage, paths))
            #images = images.cpu()
            #labels = labels.cpu()
            #img = images[0].permute(1,2,0)
            #plt.imshow(img)
            #plt.pause(1)



    column_names = ['Prediction', 'Accuracy', 'Image']
    #print(outfile[0][1])
    out = np.asarray(outfile)
    #print(out)
    df = pd.DataFrame(out, columns=column_names)
    #print(df)
    df.to_csv('prediction_list_1.csv', index=False)

create_predictions()
