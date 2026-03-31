import os
import time
import torch
import torch.nn as nn
from torch import optim
import numpy as np

from models import WEANet
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.losses import DualObjectiveLoss
from utils.metrics import metric
from data_provider.data_factory import data_provider


class Exp_Main(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.criterion = DualObjectiveLoss(self.args).to(self.device)

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
        return device

    def _build_model(self):
        model = WEANet.WEANet(self.args)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                pred, _ = self.model(batch_x)
                loss = criterion(pred, batch_y)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                pred, block1_features = self.model(batch_x)

                loss_total, loss_task, loss_recon = self.criterion(pred, batch_y, block1_features, batch_x)
                train_loss.append(loss_total.item())

                loss_total.backward()
                model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, self.criterion)
            test_loss = self.vali(test_data, test_loader, self.criterion)

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        self.model.eval()
        preds = []
        trues = []

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                pred, _ = self.model(batch_x)

                preds.append(pred.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        mae, mse, rmse, mape, mspe = metric(preds, trues)

        return mae, mse, rmse, mape, mspe