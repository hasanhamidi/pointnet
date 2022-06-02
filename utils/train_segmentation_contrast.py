from __future__ import print_function
import argparse
import os
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







opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)





num_batch = len(dataset) / opt.batchSize

for epoch in range(opt.nepoch):
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        points, target = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)

        loss = loss_func(features = pred,labels_all = target)
        print('train%f - epoch %d -%d' % (loss, epoch,i))


        loss.backward()
        optimizer.step()
        # pred_choice = pred.data.max(1)[1]
        # correct = pred_choice.eq(target.data).cpu().sum()
        # print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item()/float(opt.batchSize * 2500)))
        
        if i % 5 == 0:
          with torch.no_grad():
                j, data = next(enumerate(testdataloader, 0))
                points, target = data
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                classifier = classifier.eval()
                pred, _, _ = classifier(points)
                loss = loss_func(features = pred,labels_all = target)
                print('%s %f - epoch %d -%d' % ( blue('test'),loss, epoch,i))
                if i % 30 == 0:
                    show_embeddings((pred).cpu().detach().numpy(),target.cpu().detach().numpy(),title = "1train"+str(epoch)+"-"+str(i)+"-"+str(loss.item()))


    torch.save(classifier.state_dict(), '%s/seg_model_%s_%d.pth' % (opt.outf, opt.class_choice, epoch))

## benchmark mIOU
shape_ious = []
for i,data in tqdm(enumerate(testdataloader, 0)):
    points, target = data
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(2)[1]

    pred_np = pred_choice.cpu().data.numpy()
    target_np = target.cpu().data.numpy() - 1

    for shape_idx in range(target_np.shape[0]):
        parts = range(num_classes)#np.unique(target_np[shape_idx])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            if U == 0:
                iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))

print("mIOU for class {}: {}".format(opt.class_choice, np.mean(shape_ious)))
#################################################################################







# {'gpus': [0], 'optimizer': {'weight_decay': 0.0, 'lr': 0.001, 'lr_decay': 0.5, 'bn_momentum': 0.5, 'bnm_decay': 0.5,
#  'decay_step': 300000.0}, 'task_model': {'class': 'model_ssg.PointNet2SemSegSSG', 'name': 'sem-ssg'},
#   'model': {'use_xyz': True}, 'distrib_backend': 'dp', 'num_points': 4096, 'epochs': 50, 'batch_size': 24}



import os
import sys

pointnet2_dir = os.path.split(os.path.abspath(__file__))[0]
main_dir = "/".join(pointnet2_dir.split("/")[0:-1])
pointnet2_ops_lib_dir = main_dir+"/pointnet/" 

sys.path.insert(0,main_dir)
sys.path.insert(0,pointnet2_ops_lib_dir)

import hydra
import omegaconf
import torch
import numpy as np
import time
from tqdm import tqdm, trange
from data.Indoor3DSemSegLoader import fakeIndoor3DSemSeg,Indoor3DSemSeg
from torch.utils.data import DataLoader
from losses import Contrast_loss_point_cloud



# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

# from pytorch_lightning.loggers import TensorBoardLogger
# from surgeon_pytorch import Inspect,get_layers

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def hydra_params_to_dotdict(hparams):
    def _to_dot_dict(cfg):
        res = {}
        for k, v in cfg.items():
            if isinstance(v, omegaconf.DictConfig):
                res.update(
                    {k + "." + subk: subv for subk, subv in _to_dot_dict(v).items()}
                )
            elif isinstance(v, (str, int, float, bool)):
                res[k] = v

        return res

    return _to_dot_dict(hparams)


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
        self.last_model = pointnet2_dir + "/checkpoints/27_001.pth.tar"
        self.validation_acc = [0]
        self.training_acc = [0]

    def save_checkpoint(self,state,filename = "chechpoint.pth.tar"):
        print("**************saving model****************")
        print(pointnet2_dir)
        filename =pointnet2_dir+"/checkpoints/"+filename
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

                self.lr_scheduler.step()


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
                self.save_checkpoint(state,filename= f"acc: {self.validation_acc[-1]:.4f} {self.epoch:.4f} chechpoint.pth.tar")

            
        # writer.close()
        return self.training_loss, self.validation_loss, self.learning_rate

    def _train(self):



        self.model.train()  # train mode
        train_losses = []  # accumulate the losses here
        train_acc = []
        batch_iter = self.training_DataLoader

        for (x, y) in batch_iter:

            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)
            self.optimizer.zero_grad()  # zerograd the parameters
            out = self.model(input)  # one forward pass

            loss = self.criterion(out, target)  # calculate loss
            
            loss_value = loss.item()

            train_losses.append(loss_value)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters


        self.training_loss.append(np.mean(train_losses))

        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])


    def _validate(self):


        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        valid_acc    = []
        batch_iter = self.validation_DataLoader

        for (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)

            with torch.no_grad():
                out = self.model(input)
                loss = self.criterion(out, target)
                loss_value = loss.item()
                acc = (torch.argmax(out, dim=1) == target).float().mean()
                valid_losses.append(loss_value)


        self.validation_loss.append(np.mean(valid_losses))


def main(cfg):
##cuda 
    if torch.cuda.is_available():
         device = torch.device('cuda')
        
    else:
         torch.device('cpu')

##data loaders 

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
##model
    model = PointNetDenseCls_contrast(k=num_classes, feature_transform=opt.feature_transform).to(device)
##optimizers
    optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
##loss function
    criterion =  Contrast_loss_point_cloud()

 
    trainer = Trainer(model=model,
                    device=device,
                    criterion=criterion,
                    optimizer=optimizer,
                    training_DataLoader= dataloader,
                    validation_DataLoader= testdataloader,
                    lr_scheduler=scheduler,
                    epochs= 10,
                    epoch=0,
                    notebook=True)


    # start training

    training_losses, validation_losses, lr_rates = trainer.run_trainer()
    #test







if __name__ == "__main__":
    main()
