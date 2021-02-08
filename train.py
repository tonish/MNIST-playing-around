# imports
import torch
import torchvision
import torch.nn as nn # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F # All functions that don't have any parameters
from torch.utils.data import DataLoader # Gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms # Transformations we can perform on our dataset
from torch.utils.tensorboard import SummaryWriter
import utils
from tqdm import tqdm

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print ('device is' , device)

#hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 10e-3
batch_size = 64
num_epoch = 5
load_model = True
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

#load data
train_dataset = datasets.MNIST(root= 'MNIST/', train=True, transform= transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset = train_dataset, batch_size = 64, shuffle=True)

test_dataset = datasets.MNIST(root= 'MNIST/', train=False, transform= transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset = test_dataset, batch_size = 64, shuffle=True)


#Initilize network
# load pretrained model & modify it
model = torchvision.models.resnet50(pretrained=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3,bias=False)

# Fine tuning
# If you want to do finetuning then set requires_grad = True
# Remove these two lines if you want to train entire model,
# and only want to load the pretrain weights.
for param in model.parameters():
    param.requires_grad = False

# model.avgpool = utils.Identity()
model.fc = nn.Sequential(
    nn.Linear(2048, 100), nn.ReLU(), nn.Linear(100, num_classes))
model.to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# Define Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=5, verbose=True)

if load_model:
    utils.load_checkpoint(model, optimizer, torch.load("my_checkpoint.pth.tar"))

writer = SummaryWriter(f'runs/MNIST/MiniBatchSize {batch_size} LR {learning_rate}')
step = 0

#Train network
for epoch in range(num_epoch):
    loop = tqdm(train_loader)
    losses = []
    accuracies = []

    if epoch % 3 == 0:
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        utils.save_checkpoint(checkpoint)

    for batch_idx, (data, targets) in enumerate(train_loader):

        #get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        #get to correct shape
        # data = data.reshape(data.shape[0],-1)

        #forward
        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss.item())

        #backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent
        optimizer.step()

        # calculate running training accuracy
        features = data.reshape(data.shape[0], -1)
        img_grid = torchvision.utils.make_grid(data)
        _, predictions = scores.max(1)
        num_correct = (predictions == targets).sum()
        running_train_acc = float(num_correct) / float(data.shape[0])
        accuracies.append(running_train_acc)
        writer.add_scalar('Training loss', loss, global_step=step)
        writer.add_scalar('Training accuracy', running_train_acc, global_step=step)

        # Plot things to tensorboard
        class_labels = [classes[label] for label in predictions]
        writer.add_image("mnist_images", img_grid)
        writer.add_histogram("fc", model.fc[2].weight)
        writer.add_scalar("Training loss", loss, global_step=step)
        writer.add_scalar("Training Accuracy", running_train_acc, global_step=step)

        if batch_idx == 230:
            writer.add_embedding(
                features,
                metadata=class_labels,
                label_img=data,
                global_step=batch_idx,
            )
        step += 1
    loop.set_description(f"Epoch [{epoch}/{num_epoch}]")
    # loop.set_postfix()

    mean_loss = sum(losses) / len(losses)

    # After each epoch do scheduler.step, note in this scheduler we need to send
    # in loss for that epoch!
    scheduler.step(mean_loss)

    print(f"Cost at epoch {epoch} is {mean_loss:.5f}")

#check accuracy on training & test to see how good our model

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("checking accuracy on training data")
    else:
        print("checking accuracy on test data")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device = device)
            y = y.to(device = device)
            # x = x.reshape(x.shape[0], -1)

            scores = model(x)
            #64 images * 10
            _, predictions = scores.max(1)
            num_correct += (predictions ==y).sum()
            num_samples += predictions.size(0)

        print (f'Got {num_correct} / {num_samples} with accuracy {float(num_correct / float(num_samples)*100):.2f}')
        model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)