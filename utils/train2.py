
from __future__ import print_function

import argparse
import os
import sys
pointnet= os.path.split(os.path.abspath(__file__))[0]
main_dir = "/".join(pointnet.split("/")[0:-1])
pointnet2_ops_lib_dir = main_dir+"/pointnet/" 

sys.path.insert(0,main_dir)
sys.path.insert(0,pointnet2_ops_lib_dir)
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetDenseCls, feature_transform_regularizer
from pointnet.model import PointNetDenseCls_contrast
from pointnet.loss import Contrast_loss_point_cloud
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from openTSNE import TSNE
import matplotlib.pyplot as plt

from pylab import cm

from sklearn.manifold import skearnTSNE



def show_embeddings(tsne_embs_i, lbls,title = "",highlight_lbls=None, imsize=8, cmap=plt.cm.tab20):


    labels = lbls.flatten()
    feat = np.zeros((tsne_embs_i.shape[1],tsne_embs_i.shape[2])).T
    
    for b in tsne_embs_i:
      feat= np.concatenate((feat, b.T), axis=0)

    feat= feat[tsne_embs_i.shape[2]: , :]
    number_of_labels = np.amax(labels) + 1
    selected = np.zeros((tsne_embs_i.shape[1],1)).T
    labels_s = []
    for i in range(number_of_labels):
      selected= np.concatenate((selected,feat[labels == i][0:100]), axis=0)
      labels_s= np.concatenate((labels_s,labels[labels == i][0:100]), axis=0)
    selected = selected[1:]

    tsne = TSNE(metric='cosine', n_jobs=-1)
    tsne_embs = tsne.fit(selected)

    fig,ax = plt.subplots(figsize=(imsize,imsize))

    # colors = cmap(np.array(labels_s))
    ax.scatter(tsne_embs[:,0], tsne_embs[:,1], c=labels_s, cmap=cmap, alpha=1 if highlight_lbls is None else 0.1)
    fig.savefig(title+'.png') 



def show_embedding_sklearn(tsne_embs_i, lbls,title = ""):

    labels = lbls.flatten()
    feat = np.zeros((tsne_embs_i.shape[1],tsne_embs_i.shape[2])).T
    
    for b in tsne_embs_i:
      feat= np.concatenate((feat, b.T), axis=0)

    feat= feat[tsne_embs_i.shape[2]: , :]
    number_of_labels = np.amax(labels) + 1
    selected = np.zeros((tsne_embs_i.shape[1],1)).T
    labels_s = []
    for i in range(number_of_labels):
      selected= np.concatenate((selected,feat[labels == i][0:100]), axis=0)
      labels_s= np.concatenate((labels_s,labels[labels == i][0:100]), axis=0)
    selected = selected[1:]

    tsne = sklearnTSNE(n_components=2, random_state=0)  # n_components means you mean to plot your dimensional data to 2D
    x_test_2d = tsne.fit_transform(selected)



    markers = ('s', 'd', 'o', '^', 'v', '8', 's', 'p', "_", '2')
    color_map = {0: 'red', 1: 'blue', 2: 'lightgreen', 3: 'purple', 4: 'cyan', 5: 'black', 6: 'yellow', 7: 'magenta',
            8: 'plum', 9: 'yellowgreen'}
    for idx, cl in enumerate(np.unique(y)):

        plt.scatter(x=x_test_2d[labels_s == cl, 0], y=x_test_2d[labels_s == cl, 1], c=color_map[idx], marker=markers[idx],
                label=cl)
    plt.xlabel('X in t-SNE')
    plt.ylabel('Y in t-SNE')
    plt.legend(loc='upper left')
    plt.title('t-SNE visualization of test data')
    random_str = str(random.randint(0,10))
    plt.savefig("/./content/shape"+random_str+"--"+str(title)+'.png')

def vis_point_cloud(points, target, title = 12345,relative = ''):
    # points = points.transpose(2, 1)
    points = points.T

    points = points.cpu().data.numpy()
    target = target.cpu().data.numpy()

    if relative == "":
      relative = points[:, 2]
    else:
      relative = relative.cpu().data.numpy()

    
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=cm.hsv(relative/max(relative)), marker='o')
    colmap = cm.ScalarMappable(cmap=cm.hsv)
    colmap.set_array(relative)
    cb = fig.colorbar(colmap)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # plt.show()
    random_str = str(random.randint(0,10))
    plt.savefig("/./content/"+random_str+"--"+str(title)+'.png')


class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_DataLoader: torch.utils.data.Dataset,
                 validation_DataLoader: torch.utils.data.Dataset = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 epochs: int = 100,
                 epoch: int = 0,
                 notebook: bool = False,
                 save_best_model : int = 1,
                 load_checkpoint : bool = False
                 ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.notebook = notebook
        self.save_best_model = save_best_model
        self.load_checkpoint = load_checkpoint
        
    
        self.training_loss = [0]
        self.validation_loss = [0]
        self.learning_rate = [0]
        self.last_model = pointnet + "/checkpoints/27_001.pth.tar"
        self.validation_acc = [0]
        self.training_acc = [0]

    def save_checkpoint(self,state,filename = "chechpoint.pth.tar"):
        print("**************saving model****************")
        print(pointnet)
        filename =pointnet+"/checkpoints/"+filename
        torch.save(state,filename)
        self.last_model = filename


    def load_from_checkpoint(self , checkpoint = "" ):
        print("++++++++++++++loading_model++++++++++++++++")
        if checkpoint == "":
            checkpoint =  torch.load(self.last_model)
        self.model.load_state_dict(checkpoint["state_dict"])
        # self.optimizer = 
        



    def run_trainer(self):


        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        progressbar = trange(self.epochs, desc='Progress')
        # writer = SummaryWriter("loss_lr_logs")
        if self.load_checkpoint == True:
            self.load_from_checkpoint(torch.load(self.last_model))


        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1  # epoch counter

            """Training block"""
            self._train()


            """Validation block"""
            if self.validation_DataLoader is not None:
                self._validate()
            

            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                if self.validation_DataLoader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    self.lr_scheduler.batch(self.validation_loss[i])  # learning rate scheduler step with validation loss
                else:
                    self.lr_scheduler.batch()  # learning rate scheduler step
            logs = {"train_loss":self.training_loss[-1],"val_loss":self.validation_loss[-1],"lr":self.learning_rate[-1]}
            logs_acc = {"training_acc":self.training_acc[-1],"val_acc":self.validation_acc[-1]}
            # writer.add_scalars("train/loss",logs, self.epoch)
            print("---------------------------------------------------------------------------------")
            print("epoch_num:",i,"\n")
            print("=>",logs,"\n","=>",logs_acc)
            print("---------------------------------------------------------------------------------")
        
            
            if self.epoch % 5 == 0:

                state = {'epoch': self.epoch,
                                'state_dict': self.model.state_dict(),
                                'optimizer': self.optimizer.state_dict()}
                self.save_checkpoint(state,filename= f"acc: {self.validation_acc[-1]:.4f} chechpoint.pth.tar")

            
        # writer.close()
        return self.training_loss, self.validation_loss, self.learning_rate

    def _train(self):



        self.model.train()  # train mode
        train_losses = []  # accumulate the losses here
        train_acc = []
        batch_iter = self.training_DataLoader
        print("tran_size ",len(self.training_DataLoader))
        indx_print = 0 
        for (x, y) in batch_iter:
            indx_print += 1

            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)



            self.optimizer.zero_grad()  # zerograd the parameters

            input = input.transpose(2, 1)
            out = self.model(input)  # one forward pass


            with torch.no_grad():
                if indx_print == 1 and self.epoch == 1 :
                    show_embedding_sklearn((out).cpu().detach().numpy(),target.cpu().detach().numpy(),title = "train_fisrt"+str(self.epoch)+"*")
                    normalize_vectors = F.normalize(out[0].T,p = 2,dim = 1)
                    dot_products = torch.matmul(normalize_vectors, normalize_vectors.T) 
                    vis_point_cloud(input[0], target[0], title = 123,relative = dot_products[0])
                    # show_embeddings((out).cpu().detach().numpy(),target.cpu().detach().numpy(),title = "train_fisrt"+str(self.epoch)+"*")
            # print(out.size())
            loss = self.criterion(out, target)  # calculate loss
            
            loss_value = loss.item()
            print("loss=>",loss_value)
            train_losses.append(loss_value)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters
            # with torch.no_grad():
            #     acc = (torch.argmax(out, dim=1) == target).float().mean()
            # train_acc.append(acc.item())



            # print(f'Training: (loss {loss_value:.4f})') 
        # print(out[0].size())
        # print(target[0].size())
        # print(out[0])
        # print(target[0])
        # print(type((out[0].T).cpu().detach().numpy()))
        # print(target[0].cpu().detach().numpy())
        with torch.no_grad():
            normalize_vectors = F.normalize(out[0].T,p = 2,dim = 1)
            dot_products = torch.matmul(normalize_vectors, normalize_vectors.T) 
            vis_point_cloud(input[0], target[0], title = self.epoch ,relative = dot_products[0])
            show_embeddings((out).cpu().detach().numpy(),target.cpu().detach().numpy(),title = "train"+str(self.epoch)+"*"+str(np.mean(train_losses)))
        self.training_loss.append(np.mean(train_losses))
        # self.training_acc.append(np.mean(train_acc))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])
        # batch_iter.close()

    def _validate(self):


        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        valid_acc    = []
        batch_iter = self.validation_DataLoader

        for (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)

            with torch.no_grad():
                input = input.transpose(2, 1)
                out = self.model(input)
                loss = self.criterion(out, target)
                loss_value = loss.item()
                acc = (torch.argmax(out, dim=1) == target).float().mean()
                valid_losses.append(loss_value)
                print("loss=>",loss_value)
                # valid_acc.append(acc.item())
                # print(f'Validation: (loss {loss_value:.4f})')
        with torch.no_grad():
            show_embeddings((out).cpu().detach().numpy(),target.cpu().detach().numpy(),title = "val"+str(self.epoch)+"*"+str(np.mean(valid_losses)))
        self.validation_loss.append(np.mean(valid_losses))
        # self.validation_acc.append(np.mean(valid_acc))

        # batch_iter.close()
















parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")



def show_embeddings(tsne_embs_i, lbls,title = "",highlight_lbls=None, imsize=8, cmap=plt.cm.tab20):


    labels = lbls.flatten()
    feat = np.zeros((tsne_embs_i.shape[1],tsne_embs_i.shape[2])).T
    
    for b in tsne_embs_i:
      feat= np.concatenate((feat, b.T), axis=0)

    feat= feat[tsne_embs_i.shape[2]: , :]
    number_of_labels = np.amax(labels) + 1
    selected = np.zeros((tsne_embs_i.shape[1],1)).T
    labels_s = []
    for i in range(number_of_labels):
      selected= np.concatenate((selected,feat[labels == i][0:100]), axis=0)
      labels_s= np.concatenate((labels_s,labels[labels == i][0:100]), axis=0)
    selected = selected[1:]

    tsne = TSNE(metric='cosine', n_jobs=-1)
    tsne_embs = tsne.fit(selected)

    fig,ax = plt.subplots(figsize=(imsize,imsize))

    # colors = cmap(np.array(labels_s))
    ax.scatter(tsne_embs[:,0], tsne_embs[:,1], c=labels_s, cmap=cmap, alpha=1 if highlight_lbls is None else 0.1)
    fig.savefig(title+'.png') 


if torch.cuda.is_available():
    device = torch.device('cuda')
        
else:
    torch.device('cpu')

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)


dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False,
    class_choice=[opt.class_choice])
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

test_dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False,
    class_choice=[opt.class_choice],
    split='test',
    data_augmentation=False)
testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = dataset.num_seg_classes
print('classes', num_classes)
try:
    os.makedirs(opt.outf)
except OSError:
    pass

blue = lambda x: '\033[94m' + x + '\033[0m'

classifier = PointNetDenseCls_contrast(k=num_classes, feature_transform=opt.feature_transform)


optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()
loss_func = Contrast_loss_point_cloud()
num_batch = len(dataset) / opt.batchSize

trainer = Trainer(model=classifier,
                    device=device,
                    criterion=loss_func,
                    optimizer=optimizer,
                    training_DataLoader= dataloader,
                    validation_DataLoader=testdataloader,
                    lr_scheduler=None,
                    epochs= opt.nepoch,
                    epoch=0,
                    notebook=True)

training_losses, validation_losses, lr_rates = trainer.run_trainer()