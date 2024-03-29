import torch
from torch import nn
from models import resnet, pre_resnet, wide_resnet, densenet, simpleCNN
from models import my_model
def generate_model(opt):
    assert opt.mode in ['score', 'feature']
    if opt.mode == 'score':
        last_fc = True
    elif opt.mode == 'feature':
        last_fc = False

    assert opt.model_name in ['resnet', 'preresnet', 'wideresnet', 'resnext', 'densenet', 'simpleCNN']

    if opt.model_name == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        if opt.model_depth == 10:
            # model = resnet.resnet10(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
            #                         sample_size1=opt.sample_size1_fmri, sample_size2=opt.sample_size2_fmri,
            #                         sample_duration=opt.sample_duration_fmri, opt=opt, last_fc=last_fc)

            model = my_model.main(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut, opt=opt, last_fc=last_fc, dim=256)
        elif opt.model_depth == 18:
            model = resnet.resnet18(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                    sample_size1=opt.sample_size1_fmri, sample_size2=opt.sample_size2_fmri,
                                    sample_duration=opt.sample_duration_fmri, opt=opt, last_fc=last_fc)
        elif opt.model_depth == 34:
            model = resnet.resnet34(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                    sample_size1=opt.sample_size1_fmri, sample_size2=opt.sample_size2_fmri,
                                    sample_duration=opt.sample_duration_fmri, opt=opt, last_fc=last_fc)

        elif opt.model_depth == 50:
            model = resnet.resnet50(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                    sample_size1=opt.sample_size1_fmri, sample_size2=opt.sample_size2_fmri,
                                    sample_duration=opt.sample_duration_fmri, opt=opt, last_fc=last_fc)
        elif opt.model_depth == 101:
            model = resnet.resnet101(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                    sample_size1=opt.sample_size1_fmri, sample_size2=opt.sample_size2_fmri,
                                    sample_duration=opt.sample_duration_fmri, opt=opt, last_fc=last_fc)

        elif opt.model_depth == 152:
            model = resnet.resnet152(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                     sample_size1=opt.sample_size1, sample_size2=opt.sample_size2,
                                     sample_duration=opt.sample_duration, last_fc=last_fc)
        elif opt.model_depth == 200:
            model = resnet.resnet200(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                     sample_size1=opt.sample_size1, sample_size2=opt.sample_size2,
                                     sample_duration=opt.sample_duration, last_fc=last_fc)
    elif opt.model_name == 'wideresnet':
        assert opt.model_depth in [50]

        if opt.model_depth == 50:
            model = wide_resnet.resnet50(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                         sample_size1=opt.sample_size1, sample_size2=opt.sample_size2,
                                         sample_duration=opt.sample_duration, last_fc=last_fc)
            '''
    elif opt.model_name == 'resnext':
        assert opt.model_depth in [50, 101, 152]

        if opt.model_depth == 50:
            model = resnext.resnet50(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut, cardinality=opt.resnext_cardinality,
                                     sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                     last_fc=last_fc)
        elif opt.model_depth == 101:
            model = resnext.resnet101(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut, cardinality=opt.resnext_cardinality,
                                      sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                      last_fc=last_fc)
        elif opt.model_depth == 152:
            model = resnext.resnet152(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut, cardinality=opt.resnext_cardinality,
                                      sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                      last_fc=last_fc)
         '''
    elif opt.model_name == 'preresnet':
        assert opt.model_depth in [18, 34, 50, 101, 152, 200]

        if opt.model_depth == 18:
            model = pre_resnet.resnet18(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                        sample_size1=opt.sample_size1,  sample_size2=opt.sample_size2,
                                        sample_duration=opt.sample_duration, last_fc=last_fc)
        elif opt.model_depth == 34:
            model = pre_resnet.resnet34(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                        sample_size1=opt.sample_size1, sample_size2=opt.sample_size2,
                                        sample_duration=opt.sample_duration, last_fc=last_fc, opt=opt)
        elif opt.model_depth == 50:
            model = pre_resnet.resnet50(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                        sample_size1=opt.sample_size1, sample_size2=opt.sample_size2,
                                        sample_duration=opt.sample_duration, last_fc=last_fc)
        elif opt.model_depth == 101:
            model = pre_resnet.resnet101(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                         sample_size1=opt.sample_size1, sample_size2=opt.sample_size2,
                                         sample_duration=opt.sample_duration, last_fc=last_fc)
        elif opt.model_depth == 152:
            model = pre_resnet.resnet152(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                         sample_size1=opt.sample_size1, sample_size2=opt.sample_size2,
                                         sample_duration=opt.sample_duration, last_fc=last_fc)
        elif opt.model_depth == 200:
            model = pre_resnet.resnet200(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                         sample_size1=opt.sample_size1, sample_size2=opt.sample_size2,
                                         sample_duration=opt.sample_duration, last_fc=last_fc)
    elif opt.model_name == 'densenet':
        assert opt.model_depth in [121, 169, 201, 264]

        if opt.model_depth == 121:
            model = densenet.densenet121(num_classes=opt.n_classes, sample_size1=opt.sample_size1,
                                         sample_size2=opt.sample_size2, sample_duration=opt.sample_duration,
                                         last_fc=last_fc)
        elif opt.model_depth == 169:
            model = densenet.densenet169(num_classes=opt.n_classes,sample_size1=opt.sample_size1,
                                         sample_size2=opt.sample_size2, sample_duration=opt.sample_duration,
                                         last_fc=last_fc)
        elif opt.model_depth == 201:
            model = densenet.densenet201(num_classes=opt.n_classes,sample_size1=opt.sample_size1,
                                         sample_size2=opt.sample_size2, sample_duration=opt.sample_duration,
                                         last_fc=last_fc)
        elif opt.model_depth == 264:
            model = densenet.densenet264(num_classes=opt.n_classes,sample_size1=opt.sample_size1,
                                         sample_size2=opt.sample_size2, sample_duration=opt.sample_duration,
                                         last_fc=last_fc)

    elif opt.model_name == 'simpleCNN':
        assert opt.model_depth in [8]

        if opt.model_depth == 8:
            model = simpleCNN._3DCNN()

    if not opt.no_cuda:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)
        net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()

    # load pretrain
    if opt.pretrain_path != '':
        print('loading pretrained model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path)
        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}

        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)

        new_parameters = []
        for pname, p in model.named_parameters():
            for layer_name in opt.new_layer_names:
                if pname.find(layer_name) >= 0:
                    new_parameters.append(p)
                    break


        new_parameters_id = list(map(id, new_parameters))
        if not opt.train_pretrain:
            for p in model.parameters():
                # freeze the pretrained parameters that not in new_parameters
                if id(p) not in new_parameters_id:
                    p.requires_grad = False

        base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))

        parameters = {'base_parameters': base_parameters,
                      'new_parameters': new_parameters}

        return model, parameters

    return model, model.parameters()