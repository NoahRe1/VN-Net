import os
import time

import numpy as np
import torch
import torch.nn as nn
from lib import utils

from datasets.dataloader import load_dataset

from models import VNNET

from models.loss import masked_mae_loss, masked_mse_loss, masked_mape_loss
from tqdm import tqdm
from pathlib import Path
torch.set_num_threads(4)

import wandb


def exists(val):
    return val is not None

class Supervisor:
    def __init__(self, use_wandb, **kwargs):
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self.label_id=self._data_kwargs['label_id']
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')
        self.gpu_id = kwargs.get('gpu')
        self._device = torch.device("cuda:{}".format(self.gpu_id) if torch.cuda.is_available() else "cpu") 
        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)

        # logging
        self._experiment_name = self._train_kwargs.get('experiment_name')
        self._log_dir = self._get_log_dir(self, kwargs)
        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)
        
        # wandb
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.init(project="vnnet", name=self._experiment_name)
            wandb.config.update(kwargs)
        
        self.region=self._data_kwargs.get('region')
        self._data = load_dataset(**self._data_kwargs)
        self._model_name = self._model_kwargs.get('model_name')
        
        if self._model_kwargs['graph_encoder']['type'].lower() in ['degcn']:
            self.ode = True
        else:
            self.ode = False
        
        lonlat=np.load('data_process/lonlat_{}.npy'.format(self.region),allow_pickle=True)

        kernel_generator = utils.KernelGenerator(lonlat,k_neighbors=4)
        self.sparse_idx = torch.from_numpy(kernel_generator.sparse_idx).long().to(self._device)

        self.neighbors_idx = torch.from_numpy(kernel_generator.nbhd_idx).long().to(self._device)

        if self._model_name.lower() in ['vnnet']:
            model = VNNET(
                sparse_idx=self.sparse_idx, 
                logger=self._logger,
                region=self.region,
                **self._model_kwargs
                )
        else:
            print('The method is not provided.')
            exit()
        
        final_model=model
        for p in final_model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
        
        self.standard_scaler = self._data['scaler']

        self.num_nodes = int(self._model_kwargs.get('node_num', 1))
        self.input_dim = int(self._model_kwargs.get('input_dim', 1))
        self.seq_len = int(self._model_kwargs.get('seq_len'))  # for the encoder
        self.output_dim = int(self._model_kwargs.get('output_dim', 1))
        self.use_curriculum_learning = bool(
            self._model_kwargs.get('use_curriculum_learning', False)
            )
        self.horizon = int(self._model_kwargs.get('horizon', 1))  # for the decoder

        # setup model
        self.model = final_model.to(self._device)
        self._logger.info("Model created")
        
        # resume
        self._epoch_num = self._train_kwargs.get('epoch', 0)
        if self._epoch_num > 0:
            self.load_model(self._epoch_num)

    @staticmethod
    def _get_log_dir(self, kwargs):
        log_dir = Path(kwargs['train'].get('log_dir'))/self._experiment_name
        print(log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def save_model(self, epoch):
        model_path = Path(self._log_dir)/'saved_model'
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        config = dict(self._kwargs)
        config['model_state_dict'] = self.model.state_dict()
        config['epoch'] = epoch
        torch.save(config, model_path/('epo%d.tar' % epoch))
        self._logger.info("Saved model at {}".format(epoch))
        return 'models/epo%d.tar' % epoch

    def load_model(self, epoch_num):
        model_path = Path(self._log_dir)/'saved_model'
        assert os.path.exists(model_path/('epo%d.tar' % epoch_num)), 'Weights at epoch %d not found' % epoch_num
        checkpoint = torch.load(model_path/('epo%d.tar' % epoch_num), map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self._logger.info("Loaded model at {}".format(epoch_num))
            
    def updateNFE(self,flag):
        lay_num = self._model_kwargs.get('layer_num')
        for i in range(lay_num):
            if flag ==0:
                self.model.fm.update(self.model.graph_encoder.conv[i]._gate.getNFE())
                self.model.graph_encoder.conv[i]._gate.resetNFE()
                self.model.fm.update(self.model.graph_encoder.conv[i]._update.getNFE())
                self.model.graph_encoder.conv[i]._update.resetNFE()
                
            elif flag == 1:
                self.model.bm.update(self.model.graph_encoder.conv[i]._gate.getNFE())
                self.model.graph_encoder.conv[i]._gate.resetNFE()
                self.model.bm.update(self.model.graph_encoder.conv[i]._update.getNFE())
                self.model.graph_encoder.conv[i]._update.resetNFE()

    def train(self, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(**kwargs)

    def evaluate(self, dataset, epoch_num, load_model=False, steps=None):

        if load_model == True:
            self.load_model(epoch_num)

        with torch.no_grad():
            self.model = self.model.eval()
            val_iterator = self._data['{}_loader'.format(dataset)]
            losses = []
            y_truths = []
            y_preds = []

            MAE_metric = masked_mae_loss
            MSE_metric = masked_mse_loss
            MAPE_metric = masked_mape_loss
            for _, (x, y, image) in enumerate(val_iterator):
                x, y, image = self._prepare_data(x, y, image)
                output = self.model(x,image=image)

                loss, y_true, y_pred = self._compute_loss(y, output)
                
                y_truths.append(y_true.cpu())
                y_preds.append(y_pred.cpu())
                
                if self.ode:
                    self.updateNFE(0)

            y_preds = torch.cat(y_preds, dim=0)
            y_truths = torch.cat(y_truths, dim=0)

            loss_mae = MAE_metric(y_preds, y_truths).item()
            loss_mse = MSE_metric(y_preds, y_truths).item()
            loss_mape = MAPE_metric(y_preds, y_truths).item()
            dict_out = {'prediction': y_preds, 'truth': y_truths}
            dict_metrics = {}
            if exists(steps):
                for step in steps:
                    assert(step <= y_preds.shape[1]), ('the largest step is should smaller than prediction horizon!!!')
                    y_p = y_preds[:, step-1, :,:]
                    y_t = y_truths[:, step-1, :,:]
                    dict_metrics['mae_{}'.format(step)] = MAE_metric(y_p, y_t).item()
                    dict_metrics['rmse_{}'.format(step)] = MSE_metric(y_p, y_t).sqrt().item()
                    dict_metrics['mape_{}'.format(step)] = MAPE_metric(y_p, y_t).item()

            return loss_mae, loss_mse, loss_mape, dict_out, dict_metrics

    def _train(self, base_lr,
               steps, patience=50, epochs=100, lr_decay_ratio=0.1, log_every=1, save_model=1,
               test_every_n_epochs=10, epsilon=1e-8, **kwargs):
        # steps is used in learning rate - will see if need to use it?
        min_val_loss = float('inf')
        wait = 0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=base_lr, eps=epsilon)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps,
                                                            gamma=lr_decay_ratio,)

        self._logger.info('Start training ...')
        # this will fail if model is loaded with a changed batch_size
        num_batches = len(self._data['train_loader'])
        self._logger.info("num_batches:{}".format(num_batches))

        best_epoch=0
        batches_seen = num_batches * self._epoch_num
        for epo in range(self._epoch_num, epochs):
            
            epoch_num = epo + 1
            self.model = self.model.train()

            train_iterator = self._data['train_loader']
            losses = []

            start_time = time.time()
            progress_bar =  tqdm(train_iterator,unit="batch")

            for _, (x, y, image) in enumerate(progress_bar): 
                
                optimizer.zero_grad()

                x, y, image = self._prepare_data(x, y, image)
                output = self.model(x,labels=y,batches_seen=batches_seen,image=image)

                loss, y_true, y_pred = self._compute_loss(y, output)
                
                progress_bar.set_postfix(training_loss=loss.item())
                self._logger.debug(loss.item())

                losses.append(loss.item())

                batches_seen += 1

                if self.ode:
                    self.updateNFE(0)

                loss.backward()
                # gradient clipping - this does it in place
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                optimizer.step()
                
                if self.ode:
                    self.updateNFE(1)
            
            self._logger.info("epoch complete")
            lr_scheduler.step()
            self._logger.info("evaluating now!")

    
            val_loss, val_loss_mse, val_loss_mape, _, __ = self.evaluate(dataset='val', epoch_num=epoch_num)

            end_time = time.time()

            log_dict = {}
            if (epoch_num % log_every) == 0:
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, pde_loss: {:.4f},lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                           np.mean(losses), val_loss, pde_loss,lr_scheduler.get_last_lr()[0],
                                           (end_time - start_time))
                self._logger.info(message)
                log_dict.update({
                    "epoch": epoch_num,
                    "train_mae": np.mean(losses),
                    "val_mae": val_loss,
                    "val_mse": val_loss_mse,
                    "val_mape": val_loss_mape,
                    "lr": lr_scheduler.get_last_lr()[0]
                    })

            if (epoch_num % test_every_n_epochs) == 0:
                test_loss, test_loss_mse, test_loss_mape, _, __ = self.evaluate(dataset='test', epoch_num=epoch_num)
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, test_mae: {:.4f} ,pde_loss: {:.4f} ,test_mse: {:.4f},test_mape: {:.4f},  lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                           np.mean(losses), test_loss,pde_loss,test_loss_mse,test_loss_mape, lr_scheduler.get_last_lr()[0],
                                           (end_time - start_time))
                self._logger.info(message)
                log_dict.update({
                    "test_mae": test_loss,
                    "test_mse": test_loss_mse,
                    "test_mape": test_loss_mape
                    })
            
            if self.use_wandb:
                wandb.log(log_dict)

            if val_loss < min_val_loss:
                wait = 0
                if save_model:
                    best_epoch=epoch_num
                    model_file_name = self.save_model(epoch_num)
                    self._logger.info(
                        'Val loss decrease from {:.4f} to {:.4f}, '
                        'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss

            elif val_loss >= min_val_loss:
                wait += 1
                if wait == patience:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_num)
                    break

    def _prepare_data(self, x, y, image):
        x, y , image = self._get_x_y(x, y, image)
        return x.to(self._device), y.to(self._device), image.to(self._device)

    def _get_x_y(self, x, y, image):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                 y shape (horizon, batch_size, num_sensor, input_dim)
        """
        self._logger.debug("X: {}".format(x.size()))
        self._logger.debug("y: {}".format(y.size()))

        x = x.float()
        y = y.float()
        image = image.float()
        return x, y, image

    
    def _compute_loss(self, y_true, y_predicted):
        out_dim=-1
        y_true[...,out_dim] = self.standard_scaler[self.label_id].inverse_transform(y_true[...,out_dim])
        y_predicted[...,out_dim] = self.standard_scaler[self.label_id].inverse_transform(y_predicted[...,out_dim])

        return masked_mae_loss(y_predicted, y_true), y_true, y_predicted
    
    def _test_final_n_epoch(self, n=5, steps=[1]):
        model_path = Path(self._log_dir)/'saved_model'
        model_list = os.listdir(model_path)
        import re

        epoch_list = []
        for filename in model_list:
            epoch_list.append(int(re.search(r'\d+', filename).group()))

        epoch_list = np.sort(epoch_list)[-n:]
        for i in range(n):
            epoch_num = epoch_list[i]
            mean_score, _,mean_loss_mse, mean_loss_mape, _, dict_metrics = self.evaluate('test', epoch_num, load_model=True, steps=steps)
            message = "Loaded the {}-th epoch.".format(epoch_num) + \
                " MAE : {}".format(mean_score), "RMSE : {}".format(np.sqrt(mean_loss_mse)), "MAPE : {}".format(mean_loss_mape)
            self._logger.info(message)
            message = "Metrics in different steps: {}".format(dict_metrics)
            self._logger.info(message)
            self._logger.handlers.clear()