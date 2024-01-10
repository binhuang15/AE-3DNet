from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import os
from tqdm import trange
import random
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import time
from model import *
import argparse
import numpy as np


#set random seeds
def seed_torch(seed=23):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

#data Augumentation

trian_transforms_224 = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.Grayscale(1),
        transforms.ToTensor()
    ])

val_transforms_224 = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.Grayscale(1),
        transforms.ToTensor()
    ])


#image preprocessing
class Dataset_transform_224(Dataset):

    def __init__(self, file_dict, transform=None):
        self.file_dict = file_dict
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_dict)
        return self.filelength

    def __getitem__(self, idx):
        data2_path, frame_label = self.file_dict[idx]
        # print(data2_path)

        data2_list = os.listdir(str(data2_path))
        ID = data2_path.split('/')[-2]
        data2_list.sort(key=lambda x: int(x[:-4]))
        data2_list_num = data2_list.__len__()

        frame_data = []

        for k in range(data2_list_num):
            data4_name = data2_list[k]
            img_path = os.path.join(data2_path, data4_name)

            img = Image.open(str(img_path)).convert('L')
            if self.transform is not None:
                img = self.transform(img)
                img = img.numpy()
            frame_data.append(img)

        while len(frame_data) < 81:
            frame_data.append(img)

        frame_data = np.array(frame_data)
        frame_tensor = torch.from_numpy(frame_data)
        frame_tensor = frame_tensor.permute(1,0,2,3)

        return frame_tensor, int(frame_label), ID

def train_fn(model, optimizer, loss_fn, dataloader):
    # model training

    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for inputs, labels,ID in dataloader:

        torch.cuda.empty_cache()
        inputs = inputs.cuda()
        labels = labels.cuda()
        P = torch.sum(labels)
        alpha = 4*(labels.size(0))/(P+1)
        beta = (labels.size(0))/(labels.size(0)-P+1)
        weights1 = labels.clone()
        weights0 = 1 - labels.clone()
        weights = weights1 * alpha + weights0 * beta

        optimizer.zero_grad()

        outputs = model(inputs)
        probability = F.softmax(outputs, -1)
        loss = loss_fn(outputs, labels)
        all_loss = loss * weights
        all_loss = torch.mean(all_loss)
        loss = all_loss
        _, preds = torch.max(probability, dim=1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()* inputs.size(0)
        running_acc += (preds == labels).sum().item()

    epoch_loss = running_loss/ len(dataloader.dataset)
    epoch_acc = running_acc / len(dataloader.dataset)

    return epoch_loss, epoch_acc

def valid_fn(model, loss_fn, dataloader):
#model validation
    model.eval()
    running_loss = 0.0
    running_acc = 0.0

    for inputs, labels,ID in dataloader:

        inputs = inputs.cuda()
        labels = labels.cuda()

        with torch.no_grad():

            outputs = model(inputs)
            probability = F.softmax(outputs, -1)
            loss = loss_fn(outputs,labels)
            _, preds = torch.max(probability, dim=1)
            loss = torch.mean(loss)
            running_loss += loss.item()* inputs.size(0)
            running_acc += (preds == labels).sum().item()


    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_acc / len(dataloader.dataset)

    return epoch_loss, epoch_acc

parse = argparse.ArgumentParser(description='train_params')
parse.add_argument('--data_path', default="/home/zhi/PycharmProjects/Project_hou/HCC_HAIC/data_npy/HCC_HAIC_all_data_20s_seed23/", type=str, help='the path of training data')
parse.add_argument('--batchsize', '-bsz', default=2, type=int, help='the batchsize of trian')
parse.add_argument('--lr', '-lr', default=0.0001, type=float, help='learning rate')
parse.add_argument('--epochs', '-epochs', default=100, type=int, help='')
parse.add_argument('--model_save_name', '-ms', default="model_checkpoint", type=str, help='model_save_name')

#main function
if __name__ == '__main__':
    print('start in :', time.asctime())
    seed_torch(42)
    args = parse.parse_args()
    print(args)

    ## Five-fold cross validation
    for i in range(5):
        print(i)
        seed_torch(42)

        EPOCHS = args.epochs
        BATCH_SIZE = args.batchsize

        train_path = os.path.join(args.data_path, str(i), "train.npy")
        train_list = np.load(train_path)
        print(len(train_list.tolist()))

        train_dataset = Dataset_transform_224(train_list, transform=trian_transforms_224)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        val_path = os.path.join(args.data_path, str(i), "val.npy")
        val_list = np.load(val_path)

        val_dataset = Dataset_transform_224(val_list, transform=val_transforms_224)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = AE_3DNet(num_classes=2)

        model = model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        loss_fn = nn.CrossEntropyLoss(reduce=False, size_average=False).cuda()

        best_acc = 0
        train_loss_list = []
        val_loss_list = []
        with trange(EPOCHS) as t:

            for epoch in t:

                train_loss, train_acc = train_fn(model, optimizer, loss_fn, train_loader)
                train_loss_list.append(train_loss)
                val_loss, val_acc = valid_fn(model, loss_fn, val_loader)
                val_loss_list.append(val_loss)

                t.set_postfix(train_acc = train_acc,train_loss = train_loss,val_acc = val_acc,val_loss = val_loss,best_val_acc = best_acc)

                model_path = os.path.join("./" + args.model_save_name, str(i))
                if os.path.exists(model_path) is False:
                    os.makedirs(model_path)

                if epoch > 5 and val_acc >= best_acc:
                    best_acc = val_acc
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
                    torch.save(model.state_dict(), os.path.join(model_path, 'fold{}_epoch{}_val_loss{:.4f}_val_acc{:.4f}.pth'.format(i,epoch,val_loss,val_acc)))
        plt.figure()
        x = np.linspace(0,args.epochs,args.epochs)
        plt.plot(x,train_loss_list,color = 'red',label = 'train_loss')
        plt.plot(x, val_loss_list, color='blue',label = 'val_loss')
        plt.savefig(os.path.join(model_path, "img_" + str(i) + "_fold.png"))
    txr_path = model_path+'hyperpara.txt'
    with open(txr_path,'w') as writer:
        writer.write('lr:{}\n'.format(args.lr))
        writer.write('epochs:{}\n'.format(EPOCHS))
        writer.write('batchsize:{}\n'.format(BATCH_SIZE))
        writer.write('optimizer:{}\n'.format(optimizer))
        writer.write('loss_func:{}\n'.format(loss_fn))
        writer.write('model:{}\n'.format(model))
        writer.close()

    print('finished in :',time.asctime())


