from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetDenseCls_simple_contrast, feature_transform_regularizer
from pointnet.model import PointNetDenseCls_contrast
from pointnet.loss import Contrast_loss_point_cloud
from tqdm import trange
from tqdm import tqdm
import numpy as np




class Trainer():
    def __init__(self,model,optimizer,loss_func1,loss_func2,epoch,
                schaduler,
                train_data_loader,validation_data_loader,
                num_classes,
                feature_transform = False) -> None:

        self.model = model
        self.optimizer = optimizer
        self.loss_func1 = loss_func1
        self.loss_func2 = loss_func2
        self.epoch = epoch
        self.batch_size = train_data_loader.batch_size
        self.schaduler = schaduler
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.num_classes = num_classes
        self.feature_transform = feature_transform
        self.blue= lambda x: '\033[94m' + x + '\033[0m'
        self.red = lambda x: '\033[91m' + x + '\033[0m'
        

    def train_one_epoch(self,epoch_number=0):

        loss_train = []
        batch_iter = tqdm(enumerate(self.train_data_loader), 'Training', total=len(self.train_data_loader),
                           position=0)
        for i, data in batch_iter:
                points, target = data
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                self.optimizer.zero_grad()
                classifier = self.model.train()
                pred, trans, trans_feat , contrast_features = classifier(points)

                target_contarst  =target
                target = target.view(-1, 1)[:, 0] - 1
                loss_cross_entorpy = self.loss_func1(pred, target)
                loss_contrast =      self.loss_func2(contrast_features, target_contarst)
                loss = loss_cross_entorpy + loss_contrast

                if self.feature_transform:
                    loss += feature_transform_regularizer(trans_feat) * 0.001
                loss.backward()
                self.optimizer.step()
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                batch_iter.set_description('[%d] train loss:      %.4f accuracy: %.4f' % (epoch_number, loss.item(), correct.item()/float(self.batch_size * 2500)))
                loss_train.append(loss.item())
        return np.mean(loss_train)
    def validation_one_epoch(self,epoch_number = 0):

        loss_val = []
        batch_iter = tqdm(enumerate(self.validation_data_loader), 'Validation', total=len(self.validation_data_loader),
                           position=0)
        for i, data in batch_iter:
                points, target = data
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                classifier = self.model.eval()
                pred, trans, trans_feat , contrast_features = classifier(points)

                pred = pred.view(-1, self.num_classes)
                target_contarst  =target
                target = target.view(-1, 1)[:, 0] - 1
                loss_cross_entorpy = self.loss_func1(pred, target)
                loss_contrast =      self.loss_func2(contrast_features, target_contarst)
                loss = loss_cross_entorpy + loss_contrast
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                batch_iter.set_description('[%d] validation loss: %.4f accuracy: %.4f' % (epoch_number, loss.item(), correct.item()/float(self.batch_size * 2500)))
                loss_val.append(loss.item())
        return np.mean(loss_val)
        
    def evaluate_miou(self):
        shape_ious = []
        batch_iter = tqdm(enumerate(self.validation_data_loader), 'Miou on val', total=len(self.validation_data_loader),
                           position=0)
        for i,data in batch_iter:
            points, target = data
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = self.model.eval()
            pred, _, _ ,_= classifier(points)
            pred_choice = pred.data.max(2)[1]

            pred_np = pred_choice.cpu().data.numpy()
            target_np = target.cpu().data.numpy() - 1

            for shape_idx in range(target_np.shape[0]):
                parts = range(self.num_classes)#np.unique(target_np[shape_idx])
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

        return np.mean(shape_ious)
        


    def train_one_epoch_just_contrast(self,epoch_number=0): 
        loss_train = []
        batch_iter = tqdm(enumerate(self.train_data_loader), 'Training', total=len(self.train_data_loader),
                           position=0)
        for i, data in batch_iter:
                points, target = data
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                self.optimizer.zero_grad()
                classifier = self.model.train()
                pred, trans, trans_feat , contrast_features = classifier(points)

                pred = pred.view(-1, self.num_classes)
                
                #print(pred.size(), target.size())
                # loss_cross_entorpy = self.loss_func1(pred, target)
                target_contast = target 
                target = target.view(-1, 1)[:, 0] - 1
                loss_contrast =      self.loss_func2(contrast_features, target_contast)
                # print(contrast_features)
                # print(target)
                loss = loss_contrast

                

                if self.feature_transform:
                    loss += feature_transform_regularizer(trans_feat) * 0.001
                loss.backward()
                self.optimizer.step()
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                batch_iter.set_description('[%d] train loss contrast:      %.4f accuracy: %.4f' % (epoch_number, loss.item(), correct.item()/float(self.batch_size * 2500)))
                loss_train.append(loss.item())
        return np.mean(loss_train)

    def validation_one_epoch_just_contrast(self,epoch_number = 0):

            loss_val = []
            batch_iter = tqdm(enumerate(self.validation_data_loader), 'Validation', total=len(self.validation_data_loader),
                            position=0)
            for i, data in batch_iter:
                    points, target = data
                    points = points.transpose(2, 1)
                    points, target = points.cuda(), target.cuda()
                    classifier = self.model.eval()
                    pred, trans, trans_feat , contrast_features = classifier(points)

                    pred = pred.view(-1, self.num_classes)
                    target_contast = target 
                    target = target.view(-1, 1)[:, 0] - 1
                    loss_contrast =      self.loss_func2(contrast_features, target_contast)
                    loss = loss_contrast
                    pred_choice = pred.data.max(1)[1]
                    correct = pred_choice.eq(target.data).cpu().sum()
                    batch_iter.set_description('[%d] validation loss contrast: %.4f accuracy: %.4f' % (epoch_number, loss.item(), correct.item()/float(self.batch_size * 2500)))
                    loss_val.append(loss.item())
            return np.mean(loss_val)
    def train(self):
        # for epoch_idx in range(self.epoch * 2):
        #     self.schaduler.step()
        #     loss_train = self.train_one_epoch_just_contrast(epoch_number=epoch_idx)
        #     loss_validation = self.validation_one_epoch_just_contrast(epoch_number=epoch_idx)
        #     miou = self.evaluate_miou()
        #     print(self.red('Mean loss  and acc for epoch-[%d]\ntrain loss:      %.4f \nvalidation loss: %.4f \nMiou:            %.4f' % (epoch_idx, loss_train, loss_validation,miou)))
        #     print("------------------------------------------------------------------------------------------------")
        for epoch_idx in range(self.epoch * 2):
            self.schaduler.step()
            loss_train = self.train_one_epoch(epoch_number=epoch_idx)
            loss_validation = self.train_one_epoch(epoch_number=epoch_idx)
            miou = self.evaluate_miou()
            print(self.red('Mean loss  and acc for epoch-[%d]\ntrain loss:      %.4f \nvalidation loss: %.4f \nMiou:            %.4f' % (epoch_idx, loss_train, loss_validation,miou)))
            print("------------------------------------------------------------------------------------------------")

if __name__ == '__main__':   

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
    parser.add_argument('--class_choice', type=str, default='Car', help="class_choice")
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform") 
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
    classifier = PointNetDenseCls_simple_contrast(k=num_classes, feature_transform=opt.feature_transform)
    optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    schaduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    classifier.cuda()
    loss_contrast = Contrast_loss_point_cloud()
    trainer = Trainer(model =classifier,
                    optimizer = optimizer,
                    loss_func1 = torch.nn.CrossEntropyLoss(),
                    loss_func2 = loss_contrast ,
                    epoch = opt.nepoch,
                    schaduler = schaduler,
                    train_data_loader = dataloader,
                    validation_data_loader = testdataloader,
                    num_classes = num_classes)
    trainer.train()
