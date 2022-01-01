from __future__ import absolute_import, division, print_function

from util.validation import *
from util.logger import *
from tqdm import tqdm
from tqdm import trange
import matplotlib.pyplot as plt
import torch.nn.functional as F
from cv2 import imread

TQDM_COLS = 80

def soft_label_cross_entropy(pred, soft_label, pixel_weights=None):
    N, C, H, W = pred.shape
    loss = -soft_label.float()*F.log_softmax(pred, dim=1)
    if pixel_weights is None:
        return torch.mean(torch.sum(loss, dim=1))
    return torch.mean(pixel_weights*torch.sum(loss, dim=1))

def cross_entropy2d(input, target):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()

    # input: (n*h*w, c)
    input = input.transpose(1, 2).transpose(2, 3).contiguous()
    input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    input = input.view(-1, c)

    # target: (n*h*w,)
    mask = target >= 0.0
    target = target[mask]

    func_loss = torch.nn.CrossEntropyLoss(ignore_index=255)
    loss = func_loss(input, target)

    return loss


class Trainer(object):

    def __init__(self, classifier_model, generator_model, discriminator_model, optimizer_cl, optimizer_gn, optimizer_d, logger, num_epochs, train_loader,
                 target_train_loader, test_loader=None, epoch=0, log_batch_stride=30, check_point_epoch_stride=10, scheduler=None):
        """
        :param model: A network model to train.
        :param optimizer: A optimizer.
        :param logger: The logger for writing results to Tensorboard.
        :param num_epochs: iteration count.
        :param train_loader: pytorch's DataLoader
        :param test_loader: pytorch's DataLoader
        :param epoch: the start epoch number.
        :param log_batch_stride: it determines the step to write log in the batch loop.
        :param check_point_epoch_stride: it determines the step to save a model in the epoch loop.
        :param scheduler: optimizer scheduler for adjusting learning rate.
        """
        self.cuda = torch.cuda.is_available()
        self.classifier_model = classifier_model
        self.generator_model = generator_model
        self.discriminator_model = discriminator_model
        self.optim_cl = optimizer_cl
        self.optim_gn = optimizer_gn
        self.optim_d = optimizer_d
        self.logger = logger
        self.train_loader = train_loader
        self.target_train_loader = target_train_loader
        self.test_loader = test_loader
        self.num_epoches = num_epochs
        self.check_point_step = check_point_epoch_stride
        self.log_batch_stride = log_batch_stride
        self.scheduler = scheduler

        self.epoch = epoch

    def train(self):
        if not next(self.classifier_model.parameters()).is_cuda and self.cuda:
            raise ValueError("A model should be set via .cuda() before constructing optimizer.")

        for epoch in range(self.epoch, self.num_epoches):
            self.epoch = epoch

            # train
            self._train_epoch()

            # step forward to reduce the learning rate in the optimizer.
            if self.scheduler:
                self.scheduler.step()

            # model checkpoints
            if epoch%self.check_point_step == 0:
                self.logger.save_model_and_optimizer(self.classifier_model,
                                                     self.optim_cl,
                                                     'epoch_{}'.format(epoch))



    def evaluate(self):
        num_batches = len(self.test_loader)
        self.classifier_model.eval()

        with torch.no_grad():
            for n_batch, (sample_batched) in tqdm(enumerate(self.test_loader),
                                total=num_batches,
                                leave=False,
                                desc="Valid epoch={}".format(self.epoch),
                                ncols=TQDM_COLS):
                self._eval_batch(sample_batched, n_batch, num_batches)

    def _train_epoch(self):
        num_batches = len(self.train_loader)
        num_batches_target = len(self.target_train_loader)

        if self.test_loader:
            dataloader_iterator = iter(self.test_loader)

        # target img'lar da dönecek
        # TODO: HATA VAR MI DİYE BİR TEST ET
        for n_batch, ((sample_batched), sample_batched_target) in tqdm(enumerate(zip(self.train_loader, self.target_train_loader))):
            self.classifier_model.train()
            self.generator_model.train()
            self.discriminator_model.train()
            data = sample_batched[0]
            target = sample_batched[1].long()
            # BU BİZİM TARGET IMG OLACAK
            city_img = sample_batched_target

            if self.cuda:
                data, target = data.cuda(), target.cuda()

            self.optim_cl.zero_grad()

            # [pool5_out, out]
            classifier_output = self.classifier_model(data)
            encoder_output = classifier_output[0]
            decoder_output = classifier_output[1]
            loss = cross_entropy2d(decoder_output, target)
            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')

            loss.backward()
            self.optim_cl.step()

            """src_soft_label = F.softmax(decoder_output, dim=1).detach()
            src_soft_label[src_soft_label>0.9] = 0.9"""

            # generator part
            classifier_output_target = self.classifier_model(city_img)
            encoder_output_target = classifier_output_target[0]
            target_prediction = self.generator_model(encoder_output_target)
            target_soft_label = F.softmax(target_prediction, dim=1)
            
            tgt_soft_label = target_soft_label.detach()
            tgt_soft_label[tgt_soft_label>0.9] = 0.9

            # TODO: Discriminator gelecek buraya
            # SOURCE IMG ICINDE LOSS EKLE BURAYA SONRA TRG_LOSS + SRC_LOSS TOPLA
            target_prediction = self.discriminator_model(target_prediction)
            loss_adv_tgt = 0.001*soft_label_cross_entropy(target_prediction, torch.cat((tgt_soft_label, torch.zeros_like(tgt_soft_label)), dim=1))

            source_prediction = self.generator_model(encoder_output)
            source_soft_label = F.softmax(source_prediction, dim=1)
            
            src_soft_label = source_soft_label.detach()
            src_soft_label[src_soft_label>0.9] = 0.9

            source_prediction = self.discriminator_model(source_prediction)
            loss_adv_src = 0.001*soft_label_cross_entropy(source_prediction, torch.cat((src_soft_label, torch.zeros_like(src_soft_label)), dim=1))

            # LsGAN + LtGAN
            loss_adv = loss_adv_tgt + loss_adv_src
            loss_adv.backward()

            self.optim_cl.step()
            self.optim_gn.step()
            self.optim_d.zero_grad()

            if n_batch%self.log_batch_stride != 0:
                continue

            self.logger.store_checkpoint_var('img_width', data.shape[3])
            self.logger.store_checkpoint_var('img_height', data.shape[2])

            self.classifier_model.img_width = data.shape[3]
            self.classifier_model.img_height = data.shape[2]

            #write logs to Tensorboard.
            lbl_pred = decoder_output.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu().numpy()
            acc, acc_cls, mean_iou, fwavacc = \
                label_accuracy_score(lbl_true, lbl_pred, n_class=decoder_output.shape[1])

            self.logger.log_train(loss, 'loss', self.epoch, n_batch, num_batches)
            self.logger.log_train(acc, 'acc', self.epoch, n_batch, num_batches)
            self.logger.log_train(acc_cls, 'acc_cls', self.epoch, n_batch, num_batches)
            self.logger.log_train(mean_iou, 'mean_iou', self.epoch, n_batch, num_batches)
            self.logger.log_train(fwavacc, 'fwavacc', self.epoch, n_batch, num_batches)

            #write result images when starting epoch.
            if n_batch == 0:
                log_img = self.logger.concatenate_images([lbl_pred, lbl_true], input_axis='byx')
                log_img = self.logger.concatenate_images([log_img, data.cpu().numpy()[:, :, :, :]])
                self.logger.log_images_train(log_img, self.epoch, n_batch, num_batches,
                                             nrows=data.shape[0])

            #if the trainer has the test loader, it evaluates the model using the test data.
            if self.test_loader:
                self.classifier_model.eval()
                with torch.no_grad():
                    try:
                        sample_batched = next(dataloader_iterator)
                    except StopIteration:
                        dataloader_iterator = iter(self.test_loader)
                        sample_batched = next(dataloader_iterator)

                    self._eval_batch(sample_batched, n_batch, num_batches)
            
            torch.cuda.empty_cache()


    def _eval_batch(self, sample_batched, n_batch, num_batches):
        data = sample_batched['image']
        target = sample_batched['annotation']

        if self.cuda:
            data, target = data.cuda(), target.cuda()
        torch.cuda.empty_cache()

        classifier_output = self.classifier_model(data)
        decoder_output = classifier_output[1]

        loss = cross_entropy2d(decoder_output, target)
        loss_data = loss.data.item()
        if np.isnan(loss_data):
            raise ValueError('loss is nan while training')

        lbl_pred = decoder_output.data.max(1)[1].cpu().numpy()[:, :, :]
        lbl_true = target.data.cpu().numpy()
        acc, acc_cls, mean_iou, fwavacc = \
            label_accuracy_score(lbl_true, lbl_pred, n_class=decoder_output.shape[1])

        self.logger.log_test(loss, 'loss', self.epoch, n_batch, num_batches)
        self.logger.log_test(acc, 'acc', self.epoch, n_batch, num_batches)
        self.logger.log_test(acc_cls, 'acc_cls', self.epoch, n_batch, num_batches)
        self.logger.log_test(mean_iou, 'mean_iou', self.epoch, n_batch, num_batches)
        self.logger.log_test(fwavacc, 'fwavacc', self.epoch, n_batch, num_batches)

        if n_batch == 0:
            log_img = self.logger.concatenate_images([lbl_pred, lbl_true], input_axis='byx')
            log_img = self.logger.concatenate_images([log_img, data.cpu().numpy()[:, :, :, :]])
            self.logger.log_images_test(log_img, self.epoch, n_batch, num_batches,
                                        nrows=data.shape[0])

    def _write_img(self, score, target, input_img, n_batch, num_batches):
        lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
        lbl_true = target.data.cpu().numpy()

        log_img = self.logger.concatenate_images([lbl_pred, lbl_true], input_axis='byx')
        log_img = self.logger.concatenate_images([log_img, input_img.cpu().numpy()[:, :, :, :]])
        self.logger.log_images(log_img, self.epoch, n_batch, num_batches, nrows=log_img.shape[0])