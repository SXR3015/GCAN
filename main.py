import os
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from opts import parse_opts
from model import generate_model
from dataset import TrainSet, ValidSet
from utils import Logger, OsJoin
from train import train_epoch
from validation import val_epoch
from test import test_epoch
from dataset import TestSet
from tensorboardX import SummaryWriter
from muse_maskgit_pytorch.vqgan_vae import VQGanVAE
from models.my_model  import generator, Transformer_dirsc
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:6144"
torch.autograd.set_detect_anomaly(True)
def run(fold_id, opt):
    if opt.root_path != '':
        result_path = OsJoin(opt.root_path, opt.result_path)
        event_path = OsJoin(opt.root_path, opt.event_path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
    opt.arch ='{}-{}'.format(opt.model_name,opt.model_depth)
    #print(opt)

    print('-'*50, 'RUN FOLD %s'%str(fold_id), '-'*50)

    model, parameters = generate_model(opt)
    # print(model)
    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()

    if not opt.no_train:
        training_data = TrainSet(fold_id=fold_id)
        train_loader = DataLoader(training_data, batch_size=opt.batch_size, shuffle=True,
                                  num_workers=opt.n_threads, pin_memory=True)
        if opt.pretrain_path:
            log_path = OsJoin(result_path, opt.data_type, opt.model_name + '_' + str(opt.model_depth) + '_pretrain',
                              'logs_fold%s' % str(fold_id))
            event_path = OsJoin(event_path, opt.data_type, opt.model_name + '_' + str(opt.model_depth) + '_pretrain',
                                'logs_fold%s' % str(fold_id))
        elif not opt.pretrain_path:
            if opt.mode_net == 'pretrained classifier':
                log_path = OsJoin(result_path, opt.data_type, opt.mode_net, opt.model_name + '_' + str(opt.model_depth),
                                  'logs_%s_fold%s_%s_epoch%d' % (opt.category, str(fold_id), opt.features, opt.n_epochs))
                event_path = OsJoin(event_path, opt.data_type, opt.mode_net, opt.model_name + '_' + str(opt.model_depth),
                                     'logs_%s_fold%s_%s_epoch%d' % (opt.category, str(fold_id), opt.features, opt.n_epochs))
            elif opt.mode_net == 'image_generator':
                log_path = OsJoin(result_path, opt.data_type,opt.mode_net,opt.model_name + '_' + str(opt.model_depth),
                                  'logs_%s_fold%s_%s_epoch%d' % (opt.category, str(fold_id), opt.features, opt.n_epochs))
                event_path = OsJoin(event_path, opt.data_type, opt.mode_net,
                                     'logs_%s_fold%s_%s_epoch%d' % (opt.category, str(fold_id), opt.features, opt.n_epochs))
            elif opt.mode_net == 'region-specific':
                log_path = OsJoin(result_path, opt.data_type,opt.mode_net,opt.model_name + '_' + str(opt.model_depth),
                                  'logs_%s_fold%s_%s_epoch%d' % (opt.category, str(fold_id), opt.features, opt.n_epochs))
                event_path = OsJoin(event_path, opt.data_type, opt.mode_net,
                                     'logs_%s_fold%s_%s_epoch%d' % (opt.category, str(fold_id), opt.features, opt.n_epochs))
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        if opt.mode_net == 'pretrained classifier' or opt.mode_net == 'region-specific':
            train_logger = Logger(
                OsJoin(log_path,'train.log'),
                # ['epoch','loss','acc','lr',])
            ['epoch', 'loss', 'acc','lr'])
        elif opt.mode_net == 'image_generator':
            train_logger = Logger(
                OsJoin(log_path,'train.log'),
                # ['epoch','loss','acc','lr',])
            ['epoch', 'loss_G', 'loss_D','acc','lr'])
        if opt.mode_net == 'pretrained classifier' or opt.mode_net == 'region-specific':
            train_batch_logger = Logger(
                OsJoin(log_path, 'train_batch.log'),
                ['epoch','batch','iter','loss_G', 'acc','lr'])
        elif opt.mode_net == 'image_generator':
            train_batch_logger = Logger(
                OsJoin(log_path, 'train_batch.log'),
                ['epoch','batch','iter','loss_G', 'loss_D','acc','lr'])
        if opt.train_pretrain != ' ':
            params = [
                {'params': filter(lambda p: p.requires_grad, parameters['base_parameters']), 'lr': opt.learning_rate*0.001},
                {'params': filter(lambda p: p.requires_grad, parameters['new_parameters']), 'lr': opt.learning_rate}
            ]
        else:
            params = [{'params': filter(lambda p: p.requires_grad, parameters), 'lr': opt.learning_rate}]


        # vae = VQGanVAE(dim=256, opt=opt)
        # all_parameters = set(vae.parameters())
        # discr_parameters = set(vae.discr.parameters())
        # vae_parameters = all_parameters - discr_parameters
        if opt.mode_net == 'pretrained classifier' or opt.mode_net == 'region-specific':
            generator_parameters = model.parameters()
            discriminator_parameters = model.parameters()
        elif opt.mode_net == 'image_generator':
            generator_parameters = model.module.generator.parameters()
            discriminator_parameters = model.module.discriminator.parameters()
        if opt.train_pretrain != ' ':
            params_G = [
                {'params': filter(lambda p: p.requires_grad, generator_parameters['base_parameters']), 'lr': opt.learning_rate*0.001},
                {'params': filter(lambda p: p.requires_grad, generator_parameters['new_parameters']), 'lr': opt.learning_rate}
            ]
        else:
            params_G = [{'params': filter(lambda p: p.requires_grad, generator_parameters), 'lr': opt.learning_rate}]
        if opt.train_pretrain != ' ':
            params_D = [
                {'params': filter(lambda p: p.requires_grad, discriminator_parameters['base_parameters']), 'lr': opt.learning_rate*0.001},
                {'params': filter(lambda p: p.requires_grad, discriminator_parameters['new_parameters']), 'lr': opt.learning_rate}
            ]
        else:
            params_D = [{'params': filter(lambda p: p.requires_grad, discriminator_parameters), 'lr': opt.learning_rate}]
        # DIscriminator = Transformer_dirsc(opt)
        optimizer_G = torch.optim.Adam(params_G, lr=opt.learning_rate)
        optimizer_D = torch.optim.Adam(params_D, lr=opt.learning_rate)
        #optimizer = optim.SGD(params, momentum=0.9)
        scheduler_G = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_G, mode='min',
                                                   factor=opt.lr_decay_factor, patience =opt.lr_patience)
        scheduler_D = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_D, mode='min',
                                                   factor=opt.lr_decay_factor, patience =opt.lr_patience)
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min',
        #                                            factor=opt.lr_decay_factor, patience =opt.lr_patience)
    if not opt.no_val:
        validation_data = ValidSet(fold_id=fold_id)
        val_loader = DataLoader(validation_data, batch_size = opt.batch_size, shuffle = False,
                                                    num_workers = opt.n_threads, pin_memory=True)
        #val_logger =  Logger(OsJoin(log_path,'val.log'),['epoch','loss','acc','recall', 'Precision', 'f1', 'sensitivity', 'specificity '])
        if opt.mode_net == 'pretrained classifier' or opt.mode_net == 'region-specific':
            # val_logger = Logger(OsJoin(log_path, 'val.log'),
            #                     ['epoch', 'loss_G', 'acc'])
            val_logger = Logger(OsJoin(log_path, 'val.log'),
                                ['epoch', 'loss', 'acc', 'recall', 'precision', 'f1', 'sensitivity', 'specificity'])
        elif opt.mode_net == 'image_generator':
            # val_logger = Logger(OsJoin(log_path, 'val.log'),
            #                     ['epoch', 'loss_G','loss_D', 'acc'])
            val_logger = Logger(OsJoin(log_path, 'val.log'),
                        ['epoch', 'loss_G','loss_D', 'acc', 'recall', 'precision', 'f1', 'sensitivity', 'specificity'])
    # if opt.pretrain_path:
    #     print('loading checkpoint{}'.format(opt.pretrain_path))
    #     checkpoint = torch.load(opt.pretrain_path)
    #     assert opt.arch==checkpoint['arch']
    #
    #     opt.begin_epoch = checkpoint['epoch']
    #     model.load_state_dict(checkpoint['state_dict'])
    #     if not opt.no_train:
    #         optimizer.load_state_dict(checkpoint['optimizer'])

    #print('run')
    writer = SummaryWriter(logdir=event_path)
    for i in range(opt.begin_epoch, opt.n_epochs+1):
        torch.cuda.empty_cache()
        if not opt.no_train:
            train_epoch(i, fold_id, train_loader, model, criterion, opt,
                        train_logger, train_batch_logger, writer,optimizer_G, optimizer_D)
        if not opt.no_val:
            validation_loss_G, validation_loss_D= val_epoch(i,val_loader, model, criterion, opt, val_logger, writer,optimizer_G, optimizer_D)
        if not opt.no_train and not opt.no_val:
            # scheduler.step(validation_loss)
            scheduler_G.step(validation_loss_G)
            scheduler_D.step(validation_loss_D)
            lr_G = optimizer_G.param_groups[0]["lr"]
            writer.add_scalar('lr', lr_G, i)
    writer.close()
    test_data = TestSet()
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = opt.batch_size, shuffle=False,
                                                            num_workers = 0, pin_memory=True)
    if opt.mode_net == 'pretrained classifier' or opt.mode_net == 'region-specific':
        # val_logger = Logger(OsJoin(log_path, 'val.log'),
        #                     ['epoch', 'loss_G', 'acc'])
        test_logger = Logger(OsJoin(log_path, 'test.log'),
                            ['epoch', 'loss', 'acc', 'recall', 'precision', 'f1', 'sensitivity', 'specificity'])
    elif opt.mode_net == 'image_generator':
        # val_logger = Logger(OsJoin(log_path, 'val.log'),
        #                     ['epoch', 'loss_G','loss_D', 'acc'])
        test_logger = Logger(OsJoin(log_path, 'test.log'),
                            ['epoch', 'loss_G', 'loss_D', 'acc', 'recall', 'precision', 'f1', 'sensitivity',
                             'specificity'])
    test_epoch(1, test_loader, model, writer, fold_id, criterion, opt, test_logger)
    print('-'*47, 'FOLD %s FINISHED'%str(fold_id), '-'*48)


# if opt.test:
#     test_data = TestSet()
#     test_loader = torch.utils.data.DataLoader(test_data, batch_size = opt.batch_size, shuffle=False,
#                                                         num_workers = opt.n_threads, pin_memory=True)
#     test.test(test_loader, model, opt, test_data.label)

if __name__ == '__main__':
    opt = parse_opts()
    # 交叉验证
    for fold_id in range(1, opt.n_fold + 1):
        run(fold_id, opt)