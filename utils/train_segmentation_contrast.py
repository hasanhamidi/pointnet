from __future__ import print_function

import argparse
import os
import sys
pointnet= os.path.split(os.path.abspath(__file__))[0]
main_dir = "/".join(pointnet.split("/")[0:-1])
pointnet2_ops_lib_dir = main_dir+"/pointnet_pyt/" 

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

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()
loss_func = Contrast_loss_point_cloud()
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