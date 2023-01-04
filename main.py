import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import torchvision
from model import Net
import numpy as np
import pandas as pd
import os
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def myprint(string, filename = "/home/sunz19/PALD_NN/logs/log_test2.txt"):
    with open(filename, "a") as f:
        f.write("{}\n".format(string))
    print(string)
    
class MyDataset(Dataset):
    def __init__(self,df):
        self.data = df.values

    def __getitem__(self, index):
        return MyDataset.to_tensor(self.data[index])

    def __len__(self):
        return len(self.data)

    @staticmethod
    def to_tensor(data):
        return torch.from_numpy(data)


if __name__ == "__main__":
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    df = pd.read_csv('/home/sunz19/PALD_NN/COVID-19_MS_dataset_train.csv', low_memory=False)
    df = df.drop(['Person_ID','Sample_ID'], axis=1)
    df = df.replace('pos', 1)
    df = df.replace('neg', 0)
    y = df['PCR_result']
    X = df.drop(["PCR_result"], axis=1)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(pd.DataFrame(X), y, test_size=0.3, random_state=0)
    X_test, X_val, y_test, y_val = model_selection.train_test_split(X_test, y_test, test_size=0.5, random_state=1)
    Xy_train = pd.concat([y_train, X_train], axis = 1)
    Xy_val = pd.concat([y_val, X_val], axis = 1)
    train_dataset = MyDataset(Xy_train)
    test_dataset = MyDataset(Xy_val)
    dataset = ConcatDataset([train_dataset, test_dataset])
    batch_size = 20
    data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    data_loader_test = DataLoader(
        test_dataset, batch_size=20, shuffle=False, drop_last=False
    )
    net = Net(dim=2715).cuda()
    optimizer = torch.optim.Adadelta(net.parameters())
    criterion = nn.MSELoss(reduction="mean")
    start_epoch = 0
    epochs = 2000
    myprint("start training")

    for epoch in range(start_epoch, epochs):
        loss_clas_epoch = loss_rec_epoch = 0
        net.train()
        for step, one_data in enumerate(data_loader):
            one_data = one_data.float().cuda()
            x = one_data[:,1:]
            y = one_data[:,0]
            z = net.encode(x)
            y_new = net.clasy(z)
            x_new = net.decode(z)
            loss_rec = criterion(x, x_new)
            loss_clas = criterion(y,y_new)

            loss = loss_rec + loss_clas
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_rec_epoch += loss_rec.item()
            loss_clas_epoch += loss_clas.item()
        myprint(
            f"Epoch [{epoch}/{epochs}]\t Clas Loss: {loss_clas_epoch / len(data_loader)}\t Rec Loss: {loss_rec_epoch / len(data_loader)}"
        )
        
    y_pred = []
    for step, one_data in enumerate(data_loader):
        one_data = one_data.float().cuda()
        x = one_data[:,1:]
        y_prd = pd.DataFrame(net.predict(x).detach().cpu().numpy())
        y_pred.append(y_prd)
    y_pred = pd.concat(y_pred)
    y_pred_copy = y_pred
    myprint(str(y_pred))
    myprint(str(y_pred.shape))
    ran = list(np.arange(0.02,0.90,0.02))
    for threshold in ran:
      y_pred = y_pred_copy.copy()
      #y_pred.loc[y_pred >= threshold] = 1
      #y_pred.loc[y_pred < threshold] = 0
      myprint("(Training set) Threshold is ")
      myprint(str(threshold))
      myprint("-------------------------------------")
      y_pred.loc[y_pred[0] >= threshold,0] = 1
      y_pred.loc[y_pred[0] < threshold,0] = 0
      cm = confusion_matrix(y_train, y_pred)
      myprint(str(cm))
      myprint(str(classification_report(y_train, y_pred)))
      
      
    y_pred = []
    for step, one_data in enumerate(data_loader_test):
        one_data = one_data.float().cuda()
        x = one_data[:,1:]
        y_prd = pd.DataFrame(net.predict(x).detach().cpu().numpy())
        y_pred.append(y_prd)
    y_pred = pd.concat(y_pred)
    y_pred_copy = y_pred
    myprint(str(y_pred))
    myprint(str(y_pred.shape))
    ran = list(np.arange(0.02,0.970,0.02))
    for threshold in ran:
      y_pred = y_pred_copy.copy()
      #y_pred.loc[y_pred >= threshold] = 1
      #y_pred.loc[y_pred < threshold] = 0
      myprint("(Validation set) Threshold is ")
      myprint(str(threshold))
      myprint("-------------------------------------")
      y_pred.loc[y_pred[0] >= threshold,0] = 1
      y_pred.loc[y_pred[0] < threshold,0] = 0
      cm = confusion_matrix(y_val, y_pred)
      myprint(str(cm))
      myprint(str(classification_report(y_val, y_pred)))
        
        
    
