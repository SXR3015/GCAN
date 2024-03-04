import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = ['ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnet200']


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, sample_size1, sample_size2, sample_duration, opt, shortcut_type='B', num_classes=400, last_fc=True):
        self.last_fc = last_fc
        self.opt=opt
        self.inplanes = 64
        self.batch_size=opt.batch_size

        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(1, 2, 2),
                               padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)
        self.criterion = torch.nn.CrossEntropyLoss().cuda()# import crossentropy directly cause error
        # last_duration = math.ceil(sample_duration / 16)
        # last_size1 = math.ceil(sample_size1 / 32)
        # last_size2 = math.ceil(sample_size2 / 32)
        # self.avgpool = nn.AvgPool3d((last_duration, last_size1, last_size2), stride=1)
        self.avgpool_fmri = nn.AvgPool3d((math.ceil(opt.sample_duration_fmri / 16),
                                          math.ceil(opt.sample_size2_fmri/ 32),math.ceil(opt.sample_size1_fmri/ 32)), stride=1)
        self.avgpool_dti = nn.AvgPool3d((math.ceil(opt.sample_size1_dti/ 16),
                                          math.ceil(opt.sample_duration_dti/ 32),math.ceil(opt.sample_size2_dti/ 32)), stride=1)
        self.avgpool_dfc = nn.AvgPool3d((math.ceil(opt.sample_duration_dfc / 16),
                                          math.ceil(opt.sample_size1_fc/ 32),math.ceil(opt.sample_size2_fc/ 32)), stride=1)
        self.avgpool_dfc_half = nn.AvgPool3d((math.ceil(opt.sample_duration_dfc / 32),
                                          math.ceil(opt.sample_size1_fc/ 32),math.ceil(opt.sample_size2_fc/ 32)), stride=1)
        self.avgpool_dfc_quarter = nn.AvgPool3d((math.ceil(opt.sample_duration_dfc / 64),
                                          math.ceil(opt.sample_size1_fc/ 32),math.ceil(opt.sample_size2_fc/ 32)), stride=1)
        self.avgpool_zfc = nn.AvgPool3d((math.ceil(opt.sample_size2_fc/ 16),
                                          math.ceil(opt.sample_size1_fc/ 32), 1), stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.opt = opt
        self.weight_ce = opt.weight_ce
        self.weight_cl_fl = opt.weight_cl_fl
        self.weight_cl_fc = opt.weight_cl_fc
        self.num_classes = num_classes
        self.n_views = opt.n_views
        self.temperature = opt.temperature
        self.w = nn.Parameter(torch.ones(3))
        self.conv2D_dfc = nn.Conv1d(in_channels=7, out_channels=1, kernel_size=7)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
# FA and FC contrastive learning
    def cnn_backbone(self, x , avgpool ):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = avgpool(x)
        return x

    def info_nce_loss(self, features):
        batch_size=features[0].shape[0]
        labels = torch.cat([torch.arange(batch_size) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()
#        features = torch.cat(features, dim=1)
        features_1 = F.normalize(features[0], dim=1)
        features_2 = F.normalize(features[1], dim=1)
        similarity_matrix = torch.matmul(features_1.reshape(batch_size*self.n_views, 1),
                                         features_2.reshape(batch_size*self.n_views, 1).T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        #similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        #similarity_matrix = similarity_matrix * (~mask)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        logits = logits / self.temperature

        return logits, labels

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def avg_choose(self, x):
        shape_res_H = x.shape[2]
        shape_res_W = x.shape[3]
        shape_res_T = x.shape[4]
        if shape_res_H == self.opt.sample_size1_fmri and shape_res_W == self.opt.sample_size2_fmri:
            avgpool = self.avgpool_fmri
        elif shape_res_H == self.opt.sample_size1_dti and shape_res_T == self.opt.sample_size2_dti:
            avgpool = self.avgpool_dti
        elif shape_res_T == self.opt.sample_duration_zfc:
            avgpool = self.avgpool_zfc
        elif shape_res_H == self.opt.sample_size1_fc and shape_res_T == self.opt.sample_duration_dfc:
            avgpool = self.avgpool_dfc
        else:
            avgpool = self.avgpool_fmri
        return avgpool
    def dfc_pyramid(self, x):
        shape_res_H = x.shape[2]
        shape_res_W = x.shape[3]
        shape_res_T = x.shape[4]
        x_full = self.cnn_backbone(x, self.avgpool_dfc)
        x_half_1 = self.cnn_backbone(x[:, :, :, 0:round(shape_res_T / 2)], self.avgpool_dfc_half)
        x_half_2 = self.cnn_backbone(x[:, :, :, round(shape_res_T / 2):], self.avgpool_dfc_half)
        x_quarter_1 = self.cnn_backbone(x[:, :, :, 0:round(shape_res_T / 4)], self.avgpool_dfc_quarter)
        x_quarter_2 = self.cnn_backbone(x[:, :, :, round(shape_res_T / 4): round(shape_res_T / 2)],
                                        self.avgpool_dfc_quarter)
        x_quarter_3 = self.cnn_backbone(x[:, :, :, round(shape_res_T / 2): round(shape_res_T * 3 / 4)],
                                        self.avgpool_dfc_quarter)
        x_quarter_4 = self.cnn_backbone(x[:, :, :, round(shape_res_T * 3 / 4): round(shape_res_T)],
                                        self.avgpool_dfc_quarter)
        dfc_fea = torch.cat([x_full, x_half_1, x_half_2, x_quarter_1, x_quarter_2, x_quarter_3, x_quarter_4], dim=2)
        x = self.conv1D_dfc(dfc_fea)
    def forward(self, x):
        x_res = x.copy()
        fea_arr_fc = []
        fea_arr_local_dti = []
        # w1 = torch.exp(self.w[0])/torch.sum(torch.exp(self.w))
        # w2 = torch.exp(self.w[1])/torch.sum(torch.exp(self.w))
        # w3 = torch.exp(self.w[2]) / torch.sum(torch.exp(self.w))
        # x_array = torch.zeros(x_res[0][0].shape[0], self.num_classes).cuda()
        x_array_list = list()
        for x in x_res[0]:
            x = torch.where(torch.isinf(x), torch.full_like(x, 0), x)
            shape_res_H = x.shape[2]
            shape_res_W = x.shape[3]
            shape_res_T = x.shape[4]
            #if shape_res_H==164 and shape_res_W==164 and shape_res_T==165:

            # x = self.conv1(x)
            # x = self.bn1(x)
            # x = self.relu(x)
            # x = self.maxpool(x)
            #
            # x = self.layer1(x)
            # x = self.layer2(x)
            # x = self.layer3(x)
            # x = self.layer4(x)
            # x = avgpool(x)
            if x.shape[3] == self.opt.sample_duration_dfc and x.shape[2] == self.opt.sample_size1_fc:
                self.dfc_pyramid(x)
            else:
              avgpool =self.avg_choose(x)
              x = self.cnn_backbone(x, avgpool)
            if shape_res_H == self.opt.sample_size1_fc:
                fea_arr_fc.append(x)
            if shape_res_H == self.opt.sample_size1_fmri or shape_res_H == self.opt.sample_size1_dti:
                fea_arr_local_dti.append(x)
            x = x.view(x.size(0), -1)
            x_array_list.append(x)
            # if self.last_fc:
            #     x = self.fc(x)
        #x_multi_tesor_list=torch.Tensor(np.array(x_array_list))
        x_multi_add = torch.zeros(x_array_list[0].shape).cuda()
        x_multi_multiply = torch.ones(x_array_list[0].shape).cuda()
        for tensor in x_array_list:
            x_multi_add = torch.add(x_multi_add, tensor)
            x_multi_multiply = torch.multiply(x_multi_multiply,tensor)
            x_mid_multiply = torch.multiply(x_multi_add, x_multi_multiply)
            x_multi = torch.add(x_multi_add, x_mid_multiply)
        # x_multi = torch.sum(torch.Tensor(x_array_list), 1)
        if self.last_fc:
            x = self.fc(x_multi)
            #x_array = x + x_array
        if len(fea_arr_fc) > 1 and len(fea_arr_local_dti) > 1:
            fea_arr_local_dti_fusion=[torch.multiply(fea_arr_local_dti[0]*fea_arr_local_dti[0])
                                      , torch.add(fea_arr_local_dti[0]+fea_arr_local_dti[0])]
            logits_fc, labels_fc = self.info_nce_loss(fea_arr_fc)
            logits_local_dti, labels_local_dti = self.info_nce_loss(fea_arr_local_dti_fusion)
#            loss_cl = nn.CrossEntropyLoss(logits.type(torch.FloatTensor), labels.type(torch.FloatTensor))
            loss_cl_fc = self.criterion(logits_fc, labels_fc)
            loss_cl_local_dti = self.criterion(logits_local_dti, labels_local_dti)
            #loss_ce = nn.CrossEntropyLoss(x/3, x_res[1])
            loss_ce= self.criterion(x, x_res[1])
            #w1*loss_cl_fc_dti +w2*loss_cl_local_dti+w3*
            loss = self.weight_ce*loss_ce/(self.weight_cl_fc*loss_cl_fc+self.weight_cl_fl*loss_cl_local_dti+self.weight_ce*loss_ce)
            #loss = self.weight_cl_fc * loss_cl_fc + self.weight_cl_fl * loss_cl_local_dti
            return loss, x
        else:
            loss = nn.CrossEntropyLoss(x, x_res[1])
            return loss, x
            # i = i+1




def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(ft_begin_index))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def resnet10(**kwargs):
    """Constructs a ResNet-10 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model

def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model
