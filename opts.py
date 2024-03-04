import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_path', default=r'E:\sxr\phd\LM in MRI\network_generation', type=str, help='Root directory path of code')
    parser.add_argument(
        '--pretrained_vae', default=r'E:\sxr\phd\LM in MRI\code\results\ALL\VQGAN\weights_MCI_SCD_fold1_ALFF_DFC_FA_FC_epoch300'
                                    r'\resnet10_weights_fold1_epoch218.pth', type=str, help='pretrained weight of vae')
    parser.add_argument(
        '--mask_option', default=r'DMN_FPN_OCN', type=str, help='DMN, FPN or OCN')
    parser.add_argument(
        '--pretrained', default=r'Y', type=str, help='DMN, FPN or OCN')
    parser.add_argument(
        '--dataset', default=r'adni', type=str, help='dataset: adni or clinical')
    parser.add_argument(
        '--data_root_path', default=r'D:\sxr_bak\images\train', type=str, help='Root directory path of data')
    parser.add_argument(
        '--mode_net', default=r'region-specific', type=str, help='project mode: pretrained classifier, image_generator, or region-specific')
    parser.add_argument(
        '--pretrain_path', default=r'', type=str, help='path of pretrained classifier weight')
    parser.add_argument(
        '--result_path', default='results', type=str, help='Result directory path')
    parser.add_argument(
        '--event_path', default='events', type=str, help='Result directory path')
    parser.add_argument(
        '--resume_path', default=r'E:\sxr\phd\LM in MRI\code\results\ALL\resnet10\weights_HC_MCI_SCD_fold1_ALFF_DFC_FA_FC_epoch100'
                                 r'\resnet10_weights_fold1_epoch100.pth',
        type=str, help='Saved model (.pth) of previous training'
    )
    parser.add_argument(
        '--fold_id', default='2', type=str, help='Different data type directory')
    parser.add_argument(
        '--pretrain_explain_path', default='None', type=str, help='Pretrained model (.pth)')
    #E:\sxr\phd\code_multimodal_git\multi-scale-curriculum-master\MDD_sMRI_classification_PyTorch-master\dl_test_MDD/pretrain/resnet_50.pth
    parser.add_argument(
        '--train_pretrain', default=' ', type=str, help='wheather train the pretrained model or not')
    #False
    parser.add_argument(
        '--data_type', default='DFC_CLINICAL', type=str, help='FC or JPEG')
    parser.add_argument(
        '--category', default='HC_SCD', type=str, help='Different data type directory')
    parser.add_argument(
        '--features', default='ALFF_DFC_FA_FC', type=str, help='Different data type directory')
    parser.add_argument(
        '--n_classes', default=2, type=int, help='Number of classes (an: 2, tri: 3)')
    parser.add_argument(
        '--n_fold', default=5, type=int, help='Number of cross validation fold')
    parser.add_argument(
        '--model_name', default='resnet', type=str, help='(resnet | preresnet | wideresnet | resnext | densenet | simpleCNN')
    parser.add_argument(
        '--model_depth', default=10, type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101); densenet (121, 169, 201, 264); simpleCNN(8)')
    parser.add_argument(
        '--new_layer_names',
        # default=['upsample1', 'cmp_layer3', 'upsample2', 'cmp_layer2', 'upsample3', 'cmp_layer1', 'upsample4', 'cmp_conv1', 'conv_seg'],
        default=['fc'], type=list, help='New layer except for backbone')
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=160, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument(
        '--resnet_shortcut', default='B', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--wide_resnet_k', default=2, type=int, help='Wide resnet k')
    parser.add_argument(
        '--resnext_cardinality', default=32, type=int, help='ResNeXt cardinality')
    parser.add_argument(
        '--manual_seed', default=1680, type=int, help='Manually set random seed')#1024
    parser.add_argument(
        '--learning_rate', default=1e-4, type=float, help= 'Initial learning rate')#学习率
    parser.add_argument(
        '--lr_decay_factor', default=0.02, type=float,
        help=' Factor by which the learning rate will be reduced. new_lr = lr * factor')
    parser.add_argument(
        '--weight_decay', default=1e-4, type=float, help='Weight Decay')
    parser.add_argument(
        '--lr_patience', default=10, type=int, help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument(
        '--batch_size', default=16, type=int, help='Batch Size')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')
    parser.add_argument('--n_views', default=512, type=int, metavar='N',
                        help='Number of views for contrastive learning training.')
    parser.add_argument(
        '--n_epochs', default=60, type=int, help='Number of total epochs to run')
    parser.add_argument(
        '--n_epochs_pretrain', default=60, type=int, help='Number of total epochs to run')
    # parser.add_argument(
    #     '--n_epochs_pretrain', default=20, type=int, help='Number of total epochs to run')
    parser.add_argument(
        '--save_weight', default=True, type=int, help='wheather save the Trained model or not.')
    parser.add_argument(
        '--checkpoint', default=20, type=int, help='Trained model is saved at every this epochs.')
    parser.add_argument(
        '--begin_epoch', default=1, type=int, help= 'Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')
    parser.add_argument(
        '--n_val_samples', default=4, type=int, help='Number of validation samples for each activity')
    parser.add_argument(
        '--sample_size1_dti', default=90, type=int, help='width of inputs')#73 256
    parser.add_argument(
        '--sample_size2_dti', default=90, type=int, help='Height of inputs')  # 61 256
    parser.add_argument(
        '--sample_duration_dti', default=1, type=int, help='Temporal duration of inputs')
    parser.add_argument(
        '--sample_size1_fmri', default=61, type=int, help='width of inputs')
    parser.add_argument(
        '--sample_size2_fmri', default=73, type=int, help='Height of inputs')  # 61 256
    parser.add_argument(
        '--sample_duration_fmri', default=61, type=int, help='Temporal duration of inputs')
    parser.add_argument(
        '--sample_size1_fc', default=164, type=int, help='width of inputs')
    # 73 256
    parser.add_argument(
        '--sample_size2_fc', default=164, type=int, help='Height of inputs')  # 61 256
    parser.add_argument(
        '--sample_duration_dfc', default=120, type=int, help='Temporal duration of inputs')
    parser.add_argument(
        '--sample_duration_zfc', default=1, type=int, help='Temporal duration of inputs')
    #61 46
    parser.add_argument(
        '--num_of_feature', default=6, type=int, help='Temporal duration of inputs')
    parser.add_argument(
        '--weight_ce', default=0.8, type=int, help='weight of Cross-Entropy')
    parser.add_argument(
        '--weight_cl_fl', default=0.1, type=int, help='weight of contrastive learning loss between functional and structure')
    parser.add_argument(
        '--weight_cl_fc', default=0.1, type=int,
        help='weight of contrastive learning loss between functional connectivity and dynamic functional connectivity')
    parser.add_argument(
        '--seq_len', default=512, type=int,
        help='image size of transformer')
    parser.add_argument(
        '--patch_size', default=32, type=int,
        help='patch size of transformer')
    parser.add_argument(
        '--dim', default=512, type=int,
        help='Last dimension of output tensor after linear transformation in transformer')
    parser.add_argument(
        '--depth', default=4, type=int,
        help='Number of Transformer blocks.')
    parser.add_argument(
        '--classify_depth', default=1, type=int,
        help='Number of Transformer blocks.')
    parser.add_argument(
        '--heads', default=8, type=int,
        help='Number of heads in Multi-head Attention layer.')
    parser.add_argument(
        '--mlp_dim', default=32, type=int,
        help='Dimension of the MLP (FeedForward) layer.')
    parser.add_argument(
        '--mode', default='score', type=str,
        help='Mode (score | feature). score outputs class scores. '
             'feature outputs features (after global average pooling).')
    # parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    # parser.add_argument(
    #   '--dampening', default=0.9, type=float, help='dampening of SGD')
    # parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
    # parser.set_defaults(nesterov=False)
    # parser.add_argument( '--optimizer', default='sgd', type=str, help='Currently only support SGD')
    '''
        parser.add_argument(
            '--n_finetune_classes', default=400, type=int,
            help='Number of classes for fine-tuning. n_classes is set to the number when pretraining.')
    '''
    parser.add_argument(
        '--ft_begin_index', default=0, type=int, help='Begin block index of fine-tuning')
    parser.add_argument(
        '--no_train', action='store_true', help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument(
        '--no_val', action='store_true', help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)
    parser.add_argument(
        '--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=True)
    parser.add_argument(
        '--test_subset', default='test', type=str, help='Used subset in test (val | test)')

    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument(
        '--n_threads', default=4, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument(
        '--cross_entropy_weights_CN_MCI_SMC', default=[0.5, 0.5,0.5], type=int, help='weights of classes in cross entropy')
    parser.add_argument(
        '--cross_entropy_weights_CN_MCI', default=[0.6, 0.4], type=int, help='weights of classes in cross entropy')
    parser.add_argument(
        '--cross_entropy_weights_CN_SMC', default=None, type=int, help='weights of classes in cross entropy')
    parser.add_argument(
        '--cross_entropy_weights_MCI_SMC', default=None, type=int, help='weights of classes in cross entropy')
    args = parser.parse_args()

    return args
