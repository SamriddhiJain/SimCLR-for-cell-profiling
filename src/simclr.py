import torch
from models.resnet_simclr import ResNetSimCLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from loss.nt_xent import NTXentLoss
from feature_eval.random_forest_classifier import RFClassifier

import os
import shutil
import sys
import getpass
from tqdm import tqdm
from datetime import datetime

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False

import numpy as np

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, dataset, config, eval_dataset=None):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter(log_dir=os.path.join(self.config["run_dir"],
                                                         getpass.getuser() + '_' + datetime.now().strftime("%d%B%Y_%H-%M-%S")))
        self.dataset = dataset
        self.eval_dataset = eval_dataset
        self.nt_xent_criterion = NTXentLoss(self.device, config['batch_size'], **config['loss'])
        self.model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def _step(self, model, xis, xjs, n_iter):

        # get the representations and the projections
        ris, zis = model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.nt_xent_criterion(zis, zjs)
        return loss

    def _save_config_file(self):
        if not os.path.exists(self.model_checkpoints_folder):
            os.makedirs(self.model_checkpoints_folder)
            shutil.copy('./config.yaml', os.path.join(self.model_checkpoints_folder, 'config.yaml'))

    def train(self):

        train_loader, valid_loader = self.dataset.get_data_loaders()

        model = ResNetSimCLR(**self.config["model"]).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), 3e-4, weight_decay=eval(self.config['weight_decay']))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)

        start_epoch = 1

        # load full previous states to continue a training
        if self.config["continue_training_from"] != "None":
            model, optimizer, scheduler, start_epoch = self._load_previous_state(model, optimizer)

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(model, optimizer,
                                              opt_level='O2',
                                              keep_batchnorm_fp32=True)

        # save config file
        self._save_config_file()

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in tqdm(range(start_epoch, self.config['epochs']+1)):
            for ((xis, xjs), _) in tqdm(train_loader):
                optimizer.zero_grad()

                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs, n_iter)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    #self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    print(f"train_loss {loss.item()}, iter {n_iter}")

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader)
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    self._save_checkpoint(epoch_counter, model, optimizer, scheduler, update_best=True)

                #self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                print(f"validation_loss {valid_loss}, iter {valid_n_iter}")
                valid_n_iter += 1

            # train linear model if requested
            if("eval_classifier_n_epoch" in self.config.keys() and
                epoch_counter % self.config['eval_classifier_n_epoch'] == 0):
                self._eval_classifier(model)

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
            #self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)
            # save checkpoints
            if ("checkpoint_every_n_epochs" in self.config.keys() and
                    epoch_counter % self.config['checkpoint_every_n_epochs'] == 0):
                self._save_checkpoint(epoch_counter, model, optimizer, scheduler)

            self._save_checkpoint(epoch_counter, model, optimizer, scheduler, update_latest=True)
            print(f"epoch {epoch_counter} finished")

    def _validate(self, model, valid_loader):

        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0
            for ((xis, xjs), _) in valid_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs, counter)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        model.train()
        return valid_loss

    def _eval_classifier(self, model):
        # validation steps
        with torch.no_grad():
            model.eval()

            classifier = RFClassifier(model, self.device)
            train_original, validate_original = self.eval_dataset.get_data_loaders()
            print("RF classifier training started.")
            classifier.train(train_original)
            print("Training classifier done.")
            score_eval = classifier.test(validate_original)
            print(f"Classifier accuracy {score_eval}")

        model.train()

    def _load_previous_state(self, model, optimizer):
        try:
            checkpoints_folder = os.path.join(self.config['run_dir'],
                                              self.config['continue_training_from'],
                                              "checkpoints")
            checkpoint = torch.load(os.path.join(checkpoints_folder, 'model_latest.pth'))

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler = checkpoint['scheduler']
            epoch = checkpoint['epoch']
            print("Loaded from checkpoint successfully.")
        except FileNotFoundError:
            Exception("Previous state checkpoint not found.")

        return model, optimizer, scheduler, epoch + 1

    def _save_checkpoint(self, epoch, model, optimizer, scheduler, update_latest=False, update_best=False):
        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler': scheduler}
        if update_latest:
            torch.save(checkpoint, os.path.join(self.model_checkpoints_folder, 'model_latest.pth'))
        elif update_best:
            torch.save(checkpoint, os.path.join(self.model_checkpoints_folder, 'model_best.pth'))
        else:
            torch.save(checkpoint, os.path.join(self.model_checkpoints_folder, f'model_epoch_{epoch}.pth'))
