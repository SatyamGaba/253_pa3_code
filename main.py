from torchvision import utils
from basic_fcn import *
from resnet18 import *
from dataloader import *
from utils import *
# from utils import *
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
from tqdm import tqdm, tqdm_notebook
# import sys

torch.cuda.empty_cache()

train_dataset = CityScapesDataset(csv_file='train.csv', transforms=transforms.RandomCrop(512,1024))
val_dataset = CityScapesDataset(csv_file='val.csv')
test_dataset = CityScapesDataset(csv_file='test.csv')
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=3,
                          num_workers=0,
                          shuffle=True, 
                         )
val_loader = DataLoader(dataset=val_dataset,
                          batch_size=1,
                          num_workers=0,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                          batch_size=1,
                          num_workers=0,
                          shuffle=True)



def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
#         torch.nn.init.xavier_uniform(m.bias.data, 0)
        nn.init.constant_(m.bias, 0)
        
n_class = 34
epochs = 100
criterion = nn.CrossEntropyLoss() # Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html
model = Resnet18(n_class=n_class)
# model = FCN(n_class=n_class)
# model.apply(init_weights)
model = torch.load_state_dict(torch.load('resnet18_1'))
optimizer = optim.Adam(model.parameters(), lr=5e-3)


use_gpu = torch.cuda.is_available()
if use_gpu:
    print("GPU is available")
    model = model.cuda()
    
def train():
    '********  losses for all epochs  ********'
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        '******** running loss for one epoch  ********'
        running_loss = 0.0
        
        ts = time.time()
        for iter, (X, Y) in tqdm(enumerate(train_loader), desc ="Iteration num: "): # X=input_images, tar=one-hot labelled, y=segmentated
            optimizer.zero_grad()
            Y = Y.long()
            if use_gpu:
                inputs = X.cuda() # Move your inputs onto the gpu
                labels = Y.cuda() # Move your labels onto the gpu
                inputs.required_grad = False
                labels.required_grad = False

            else:
                inputs, labels =  X, Y # Unpack variables into inputs and labels

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            '******** accumulate running_loss for each batch ********'
            running_loss += loss.item() * inputs.size(0)

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
        
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        torch.save(model.state_dict(), "resnet18_"+str(epoch))
        
        '******** save average training loss for each epoch  ********'
        train_losses.append(running_loss/len(train_loader))
        '******** save average validation loss for each epoch  ********'
        val_loss = val(epoch)
        torch.cuda.empty_cache()
        val_losses.append(val_loss)
        
        model.train()
        torch.cuda.empty_cache()
        
    x = [i for i in range(epochs)]
    plt.title("Plot showing training and validation loss against number of epochs")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.plot(x, train_losses, color='r', label='training loss')
    plt.plot(x, val_losses, color = 'b', label = 'validation loss')
    
    plt.legend()
    plt.show()
    plt.plot()



def val(epoch):
    model.eval()

    #Complete this function - Calculate loss, accuracy and IoU for every epoch
    # Make sure to include a softmax after the output from your model
    # Evaluate
    total = 0
    correct = 0
    '******** running loss for one epoch  ********'
    running_loss = 0.0
    
    for iter, (inputs, labels) in tqdm(enumerate(val_loader)):
        inputs, labels = inputs, labels.long()
        if use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()

        outputs = model(inputs)
        
        '********  Calculating IOU for images for all valid classes  ********'
        out = iou(outputs, targets)
        #print(out)
        final = np.vstack((final, out))
        
        '******** caluculate validation loss  ********'
        loss = criterion(outputs, labels)
        '******** accumulate running_loss for each batch ********'
        running_loss += loss.item() * inputs.size(0)
        
        outputs = outputs.cpu()
        outputs = outputs.detach().numpy()
        labels = labels.cpu()
        labels = labels.detach().numpy()

        predict = np.argmax(outputs, axis = 1)
        correct = np.where(predict == labels, 1, 0).sum()
        total = predict.size

    '********  Average IOU for each class and overall average IOU  ********'
    final = np.mean(final, axis = 0)
    avg_final = np.mean(final)         

    
    print('Epoch : %d Validation Pixel Acc : %.3f' % (epoch + 1, 100.*correct/total))
    print('--------------------------------------------------------------')
    print('Epoch : %d Validation Avg IOU : %.3f' % (epoch + 1, avg_final))
    print('--------------------------------------------------------------')
    print('Epoch : %d Each class IOU : %.3f' % (epoch + 1, final))

    return (running_loss/len(val_loader))

def test():
    model.eval()

    #Complete this function - Calculate loss, accuracy and IoU for every epoch
    # Make sure to include a softmax after the output from your model
    # Evaluate
    total = 0
    correct = 0
    for iter, (X, Y) in tqdm(enumerate(val_loader)):
        inputs, labels = X, Y.long()
        if use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()

        outputs = model(inputs)
        
        '********  Calculating IOU for images for all valid classes  ********'
        out = iou(outputs, targets)
        #print(out)
        final = np.vstack((final, out))        
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
        
       

    '********  Average IOU for each class and overall average IOU  ********'
    final = np.mean(final, axis = 0)
    avg_final = np.mean(final)
          
        
    print('Epoch : %d Test Pixel Acc : %.3f' % (epoch + 1, 100.*correct/total))
    print('--------------------------------------------------------------')
    print('Epoch : %d Test Avg IOU : %.3f' % (epoch + 1, avg_final))
    print('--------------------------------------------------------------')
    print('Epoch : %d Each class IOU : %.3f' % (epoch + 1, final))
    

    #Complete this function - Calculate accuracy and IoU 
    # Make sure to include a softmax after the output from your model
    
if __name__ == "__main__":
    # val(0)  # show the accuracy before training
    train()
