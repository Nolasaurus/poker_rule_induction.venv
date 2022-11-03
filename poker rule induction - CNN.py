import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


train_data = pd.read_csv('C://Users/meyer/Python Projects/Poker Rule Induction/train.csv.zip')
test_data = pd.read_csv('C://Users/meyer/Python Projects/Poker Rule Induction/test.csv.zip')

train_data.dropna(axis=0, subset=['hand'], inplace=True)

y = train_data['hand']
X = train_data.drop(['hand'], axis=1)

### Convert to grid form 4 x 13
hand_list=[]
for index, row in X.iterrows():
    emptyA = [13*[0] for i in range(4)]
    temp_hand=emptyA.copy()

    for j in range(0,10,2):
        #print(row[j], row[j+1])
        q, r = row[j], row[j+1]
        temp_hand[q-1][r-1]=1
    hand_list.append(temp_hand)

hand_list = pd.DataFrame(hand_list)
nhl = []
for i in hand_list.index:
    nhl.append(hand_list.iloc[i].sum())
df = pd.DataFrame(nhl)

df = df.to_numpy()
y = y.to_numpy()

#y_train_labels = y_train_labels.reshape(-1,1)
#y_pred_labels  = y_pred_labels.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)


#####------------------------------------------
BATCH_SIZE = 32

torch_X_train = torch.from_numpy(X_train).type(torch.LongTensor)
torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor) # data type is long

# create feature and targets tensor for test set.
torch_X_test = torch.from_numpy(X_test).type(torch.LongTensor)
torch_y_test = torch.from_numpy(y_test).type(torch.LongTensor) # data type is long

####
torch_X_train = X_train.reshape(-1,1,4,13)
torch_X_test = X_test.reshape(-1,1,4,13)    
print(torch_X_train.shape)
print(torch_X_test.shape)

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(torch_X_train,torch_y_train)
test = torch.utils.data.TensorDataset(torch_X_test,torch_y_test)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = BATCH_SIZE, shuffle = False)


#####------------------------------------------

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32,64, kernel_size=5)
        self.fc1 = nn.Linear(3*3*64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1,3*3*64 )
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
 
cnn = CNN()
print(cnn)

it = iter(train_loader)
X_batch, y_batch = next(it)
print(cnn.forward(X_batch).shape)
 
#####
def fit(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters())#,lr=0.001, betas=(0.9,0.999))
    error = nn.CrossEntropyLoss()
    EPOCHS = 1
    model.train()
    for epoch in range(EPOCHS):
        correct = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            var_X_batch = Variable(X_batch).float()
            var_y_batch = Variable(y_batch)
            optimizer.zero_grad()
            output = model(var_X_batch)
            loss = error(output, var_y_batch)
            loss.backward()
            optimizer.step()

            # Total correct predictions
            predicted = torch.max(output.data, 1)[1] 
            correct += (predicted == var_y_batch).sum()
            #print(correct)
            if batch_idx % 250 == 0:
                print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                    epoch, batch_idx*len(X_batch), len(train_loader.dataset), 100.*batch_idx / len(train_loader), loss.data, float(correct*100) / float(BATCH_SIZE*(batch_idx+1))))
                



fit(cnn,train_loader)





def evaluate(model):
#model = mlp
    correct = 0 
    for test_imgs, test_labels in test_loader:
        #print(test_imgs.shape)
        test_imgs = Variable(test_imgs).float()
        output = model(test_imgs)
        predicted = torch.max(output,1)[1]
        correct += (predicted == test_labels).sum()
    print("Test accuracy:{:.3f}% ".format( float(correct) / (len(test_loader)*BATCH_SIZE)))
evaluate(cnn)





#output = pd.DataFrame({'Id': test_data.index+1,
#                       'hand': preds.round()})
#output.to_csv('submission.csv', index=False)
