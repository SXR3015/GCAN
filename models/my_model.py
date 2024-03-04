import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
from utils import OsJoin
from einops import rearrange
from functools import partial
from opts import parse_opts
from muse_maskgit_pytorch import VQGanVAETrainer, MaskGitTransformer,MaskGit
from muse_maskgit_pytorch.vqgan_vae import VQGanVAE
from collections import OrderedDict
from einops.layers.torch import Rearrange
from torch.autograd import grad as torch_grad
from torchvision.utils import make_grid, save_image
__all__ = ['ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnet200']
opt = parse_opts()

def safe_div(numer, denom, eps = 1e-8):
    return numer / denom.clamp(min = eps)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def grad_layer_wrt_loss(loss, layer):
    return torch_grad(
        outputs = loss,
        inputs = layer,
        grad_outputs = torch.ones_like(loss),
        retain_graph = True,
        allow_unused=True
    )[0].detach()

def gradient_penalty(images, output, weight = 10):
    batch_size = images.shape[0]

    gradients = torch_grad(
        outputs = output,
        inputs = images,
        grad_outputs = torch.ones(output.size(), device = images.device),
        create_graph = True,
        retain_graph = True,
        only_inputs = True
    )[0]

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * ((gradients.norm(2, dim = 1) - 1) ** 2).mean()
def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log
def log(t, eps=1e-10):
        return torch.log(t + eps)
def hinge_gen_loss(fake):
    return -fake.mean()
def bce_discr_loss(fake, real):
        return (-log(1 - torch.sigmoid(fake)) - log(torch.sigmoid(real))).mean()

def bce_discr_loss_label(fake, real):
    return (-log(1-torch.sigmoid(fake)) * log(1-torch.sigmoid(real))).mean()
def noop(*args, **kwargs):
    pass
def exists(val):
    return val is not None
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

class avgpool_choose(nn.Module):

    def __init__(self, opt):
        super(avgpool_choose, self).__init__()
        self.opt = opt
        self.avgpool_fmri = nn.AvgPool3d((math.ceil(160/ 16),
                                          math.ceil(160/ 32),1), stride=1)
        self.avgpool_dti = nn.AvgPool3d((math.ceil(opt.sample_size1_dti/ 16),
                                          math.ceil(opt.sample_size2_dti/ 32),math.ceil(opt.sample_duration_dti/ 32)), stride=1)
        # self.avgpool_dfc = nn.AvgPool3d((math.ceil(opt.sample_duration_dfc / 17),
        #                                   math.ceil(opt.sample_size1_fc/ 32),math.ceil(opt.sample_size2_fc/ 32)), stride=1)
        self.avgpool_dfc = nn.AvgPool3d((math.ceil(opt.sample_duration_dfc / 11),
                                          math.ceil(opt.sample_size1_fc/ 32),math.ceil(opt.sample_size2_fc/ 48)), stride=1)
        # self.avgpool_dfc = nn.AvgPool3d((math.ceil(1/12),
        #                                   math.ceil(opt.sample_size1_fc/ 32),math.ceil(opt.sample_size2_fc/ 48)), stride=1)
        self.avgpool_dfc_half = nn.AvgPool3d((math.ceil(opt.sample_duration_dfc / 16),
                                          math.ceil(opt.sample_size1_fc/ 32),math.ceil(opt.sample_size2_fc/ 64)), stride=1)
        self.avgpool_dfc_quarter = nn.AvgPool3d((math.ceil(opt.sample_duration_dfc / 16),
                                          math.ceil(opt.sample_size1_fc/ 32),math.ceil(opt.sample_size2_fc/ 128)), stride=1)
        self.avgpool_zfc = nn.AvgPool3d((math.ceil(opt.sample_size2_fc/ 16),
                                          math.ceil(opt.sample_size1_fc/ 32), 1), stride=1)
    def forward(self, x):
        shape_res_H = x.shape[2]
        shape_res_W = x.shape[3]
        shape_res_T = x.shape[4]
        if shape_res_H == self.opt.sample_size1_fmri and shape_res_W == self.opt.sample_size2_fmri:
            avgpool = self.avgpool_fmri
        elif shape_res_H == self.opt.sample_size1_dti and shape_res_T == self.opt.sample_duration_dti:
            avgpool = self.avgpool_dti
        elif shape_res_H == self.opt.sample_size1_fc and shape_res_T == self.opt.sample_duration_zfc:
            avgpool = self.avgpool_zfc
        elif shape_res_H == self.opt.sample_size1_fc and shape_res_T == self.opt.sample_duration_dfc:
            avgpool = self.avgpool_dfc
        elif shape_res_H == self.opt.sample_size1_fc and shape_res_T == round(self.opt.sample_duration_dfc/2):
            avgpool = self.avgpool_dfc_half
        elif  shape_res_H == self.opt.sample_size1_fc and shape_res_T == round(self.opt.sample_duration_dfc/4):
            avgpool = self.avgpool_dfc_quarter
        else:
            avgpool = self.avgpool_fmri
        return avgpool

# class text_generate(nn.Module):
#
#     def __init__(self, opt):
#         super(avgpool_choose, self).__init__()
#         self.opt=opt
#     def forward(self,x):
#         if opt.category == 'MCI_SCD':
#             for sam
class ResNet(nn.Module):

    def __init__(self, block, layers, opt, shortcut_type='B', num_classes=400, t_stride=2):
        # self.last_fc = last_fc
        super(ResNet, self).__init__()
        self.opt = opt
        self.inplanes = 64
        self.batch_size = opt.batch_size
        super(ResNet, self).__init__()
        self.t_stride = t_stride
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(1, 2, self.t_stride),
                               padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 512, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)
        self.pool_choose = avgpool_choose(opt)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
    def forward(self, x):
        avgpool = self.pool_choose(x)
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
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.inverse_to_patch_embedding = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim),
            nn.LayerNorm(patch_dim),
            Rearrange(' b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=image_height // patch_height, w=image_width // patch_width,
                      p1=patch_height, p2=patch_width)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n)]
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.inverse_to_patch_embedding(x)
        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        #
        # x = self.to_latent(x)
        return x

class ViT_discr(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.inverse_to_patch_embedding = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim),
            nn.LayerNorm(patch_dim),
            Rearrange(' b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=image_height // patch_height, w=image_width // patch_width,
                      p1=patch_height, p2=patch_width)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n)]
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.inverse_to_patch_embedding(x)
        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        #
        # x = self.to_latent(x)
        return x
def posemb_sincos_1d(patches, temperature=10000, dtype=torch.float32):
        _, n, dim, device, dtype = *patches.shape, patches.device, patches.dtype

        n = torch.arange(n, device=device)
        assert (dim % 2) == 0, 'feature dimension must be multiple of 2 for sincos emb'
        omega = torch.arange(dim // 2, device=device) / (dim // 2 - 1)
        omega = 1. / (temperature ** omega)

        n = n.flatten()[:, None] * omega[None, :]
        pe = torch.cat((n.sin(), n.cos()), dim=1)
        return pe.type(dtype)
class Transformer_classify(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dim_head=64):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.dim = dim
        self.heads = heads
        self.ff = FeedForward(dim, mlp_dim)
        self.att = Attention(dim, heads = heads, dim_head = dim_head),
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
class ViT_classify(nn.Module):
    def __init__(self, seq_len, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=1, dim_head=16):
        super().__init__()
        self.conv_fusion = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=1)
        self.bn_fusion = nn.BatchNorm1d(1)
        assert seq_len % patch_size == 0

        num_patches = seq_len // patch_size
        patch_dim = channels * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer_classify(dim, depth, heads, mlp_dim, dim_head)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )



    def forward(self, series):
        *_, n, dtype = *series.shape, series.dtype
        x = self.to_patch_embedding(series)
        pe = posemb_sincos_1d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe
        x_multi = x
        # for modal in range(1, series.shape[1]):
        #     x_modal = self.to_patch_embedding(series[:, modal:modal + 1, :])
        #     pe = posemb_sincos_1d(x_modal)
        #     x_modal = rearrange(x_modal, 'b ... d -> b (...) d') + pe
        #     x_multi = torch.cat([x_multi, x_modal], dim=1)
        x = self.transformer(x_multi)
        x = x.mean(dim=1)
        # x = torch.unsqueeze(x, dim=1)
        # x = self.conv_fusion(x)
        # x = self.bn_fusion(x)
        # x = torch.squeeze(x, dim=1)
        # self.linear_head(x)
        x = self.to_latent(x)
        return self.linear_head(x)

class classifier(nn.Module):
    def  __init__(self, block, layers, opt, shortcut_type='B', num_classes=400, last_fc=True, dim=None):
        super(classifier, self).__init__()
        self.avgpool_choose = avgpool_choose(opt)
        self.opt = opt
        self.ResNet = ResNet(block, layers, opt, shortcut_type=shortcut_type, num_classes=num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.fc = nn.Linear(512 * block.expansion, opt.n_classes)
        self.vit = ViT_classify(opt.seq_len, opt.patch_size, opt.n_classes, opt.dim, opt.classify_depth, opt.heads, opt.mlp_dim)
    def forward(self, x):

        if opt.mode_net == 'pretrained classifier':
            x_res = x.copy()
            x = self.ResNet(x_res[0][1].unsqueeze(4))
            x = self.vit(x.squeeze().unsqueeze(1))
            # x = self.fc(x)
            # x = self.fc(x.squeeze())
            loss_ce = self.criterion(x, x_res[1])
            return loss_ce, x
        elif opt.mode_net == 'image_generator':
            x = self.ResNet(x)
            x = self.vit(x.squeeze().unsqueeze(1))
            # x = self.fc(x)
            # x = self.fc(x.squeeze())
            return x
        else:
            x = self.ResNet(x)
            x = self.fc(x.squeeze())
            return x
            # torch.cuda.empty_cache()
            # shape_res_W = x.shape[3]



class Transformer_encoder(nn.Module):
    def  __init__(self, opt, dim=256):
        super(Transformer_encoder, self).__init__()
        self.avgpool_choose = avgpool_choose(opt)
        self.opt = opt
        self.dim = dim
        # self.codebook_size = codebook_size
        self.ViT = ViT(image_size=160, patch_size=opt.patch_size, num_classes=opt.n_classes, dim=opt.dim, depth=opt.depth,
                       heads=opt.heads, mlp_dim=opt.mlp_dim)
        self.vae = VQGanVAE(dim=self.dim, opt=opt)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.apply_grad_penalty_every =4
        self.grad_accum_every = 1
        self.use_ema = True
    def forward(self, x):

            # torch.cuda.empty_cache()
            # shape_res_W = x.shape[3]
            # x_squeeze = x_res[0][1].squeeze()
        # loss_dirsc, x = self.vae(x_res[0][1], return_discr_loss= True, return_recons = True)
        # loss_auto, x = self.vae(x_res[0][1], return_loss=True, return_recons=True)
        x_encode = self.ViT(x)
        # with torch.no_grad():
        #     checkpoint = torch.load(opt.resume_path)
        #     opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
        #     assert opt.arch == checkpoint['arch']
        #     new_state_dict = OrderedDict()
        #     for k,v in checkpoint['state_dict'].items():
        #         name=k[7:]
        #         new_state_dict[name]=v
        #     self.classifier.load_state_dict(new_state_dict)
        #     x = self.classifier(x.unsqueeze(4))
        # loss_ce = self.criterion(x, x_res[1])
        return x_encode
class Transformer_decoder(nn.Module):
    def  __init__(self, opt):
        super(Transformer_decoder, self).__init__()
        self.avgpool_choose = avgpool_choose(opt)
        self.opt = opt
        # self.codebook_size = codebook_size
        self.ViT = ViT(image_size=160, patch_size=opt.patch_size, num_classes=opt.n_classes, dim=opt.dim, depth=opt.depth,
                       heads=opt.heads, mlp_dim=opt.mlp_dim)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.apply_grad_penalty_every =4
        self.grad_accum_every = 1
        self.use_ema = True
    def last_dec_layer(self):
        return self.ViT.transformer[-1].weight
    def forward(self, x):
        # x_res = x.copy()
            # torch.cuda.empty_cache()
            # shape_res_W = x.shape[3]
            # x_squeeze = x_res[0][1].squeeze()
        # loss_dirsc, x = self.vae(x_res[0][1], return_discr_loss= True, return_recons = True)
        # loss_auto, x = self.vae(x_res[0][1], return_loss=True, return_recons=True)
        x_decode = self.ViT(x)
        # with torch.no_grad():
        #     checkpoint = torch.load(opt.resume_path)
        #     opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
        #     assert opt.arch == checkpoint['arch']
        #     new_state_dict = OrderedDict()
        #     for k,v in checkpoint['state_dict'].items():
        #         name=k[7:]
        #         new_state_dict[name]=v
        #     self.classifier.load_state_dict(new_state_dict)
        #     x = self.classifier(x.unsqueeze(4))
        # loss_ce = self.criterion(x, x_res[1])
        return x_decode
class Transformer_dirsc(nn.Module):
    def  __init__(self, opt):
        super(Transformer_dirsc, self).__init__()
        self.avgpool_choose = avgpool_choose(opt)
        self.opt = opt
        self.classifier = classifier(BasicBlock, [1, 1, 1, 1], opt)
        # self.codebook_size = codebook_size
        self.ViT_discr = ViT_discr(image_size=160, patch_size=opt.patch_size, num_classes=opt.n_classes, dim=opt.dim, depth=8,
                       heads=opt.heads, mlp_dim=opt.mlp_dim)
        self.ViT_discr_label = ViT_discr(image_size=160, patch_size=opt.patch_size, num_classes=opt.n_classes, dim=opt.dim, depth=8,
                       heads=opt.heads, mlp_dim=opt.mlp_dim)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.apply_grad_penalty_every =4
        self.grad_accum_every = 1
        self.use_ema = True
    def forward(self, x):
        try :
            x_shape = x.shape
            x_discr = self.ViT_discr(x)
            return x_discr
        except AttributeError:
            x_source = self.ViT_discr_label(x[0])
            x_target = self.ViT_discr_label(x[1])
            return x_source,x_target


# class dc_Generator(nn.Module):
#     def __init__(self):
#         super(dc_Generator, self).__init__()
#
#         self.init_size = opt.img_size // 4
#         self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))
#
#         self.conv_blocks = nn.Sequential(
#             nn.BatchNorm2d(128),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 128, 3, stride=1, padding=1),
#             nn.BatchNorm2d(128, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 64, 3, stride=1, padding=1),
#             nn.BatchNorm2d(64, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
#             nn.Tanh(),
#         )
#
#     def forward(self, z):
#         out = self.l1(z)
#         out = out.view(out.shape[0], 128, self.init_size, self.init_size)
#         img = self.conv_blocks(out)
#         return img
#
#
# class dc_Discriminator(nn.Module):
#     def __init__(self):
#         super(dc_Discriminator, self).__init__()
#
#         def discriminator_block(in_filters, out_filters, bn=True):
#             block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
#             if bn:
#                 block.append(nn.BatchNorm2d(out_filters, 0.8))
#             return block
#
#         self.model = nn.Sequential(
#             *discriminator_block(opt.channels, 16, bn=False),
#             *discriminator_block(16, 32),
#             *discriminator_block(32, 64),
#             # *discriminator_block(64, 96),
#             *discriminator_block(64, 128),
#             *discriminator_block(128, 256),
#             *discriminator_block(128, 512),
#             # *discriminator_block(256, 512),
#             # *discriminator_block(512, 1024),
#             # *discriminator_block(256, 768),
#         )
#
#         # The height and width of downsampled image
#         ds_size = opt.img_size // 2 ** 4
#         # self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
#         self.adv_layer = nn.Sequential(nn.Linear(256 * ds_size ** 2, 1), nn.Sigmoid())
#
#     def forward(self, img):
#         out = self.model(img)
#         out = out.view(out.shape[0], -1)
#         validity = self.adv_layer(out)
#
#         return validity
class generator(nn.Module):
    def __init__(self, opt):
        super(generator, self).__init__()
        self.Transformer_encode = Transformer_encoder(opt=opt)
        self.Transformer_decode = Transformer_decoder(opt=opt)
        self.label_generation = Transformer_encoder(opt=opt)
    def forward(self, noise, target_fc):
        en_fc = self.Transformer_encode(noise)
        de_fc = self.Transformer_decode(en_fc)
        label_prompt = self.label_generation(target_fc)
        return de_fc, label_prompt
class image_generator(nn.Module):
    def __init__(self, opt):
        super(image_generator, self).__init__()
        self.classifier = classifier(BasicBlock, [1, 1, 1, 1], opt)
        self.Transformer_decode = Transformer_decoder(opt=opt)
        self.Transformer_encode = Transformer_encoder(opt=opt)
        self.Transformer_discriminator = Transformer_dirsc(opt=opt)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.opt = opt
        self.vae = VQGanVAE(dim=256, opt=opt)
        self.generator = generator(opt)
        self.discriminator = Transformer_dirsc(opt=opt)

        # self.generator.apply(weights_init_normal)
        # self.discriminator.apply(weights_init_normal)
        # self.optimizer_G = optimizer_G
        # self.optimizer_D = optimizer_D

    def vgg(self):
        vgg = torchvision.models.vgg16(pretrained = True)
        vgg.classifier = nn.Sequential(*vgg.classifier[:-2])
        _vgg = vgg.cuda()
        return _vgg
    # def matrix_viarance(self, ex_connectivity, in_connectivity):
    #     matrix_variance = list()
    #     for i in range(len(ex_connectivity)):
    #         net_ex_T = torch.transpose(ex_connectivity[i],2,3)
    #         net_ex = ex_connectivity[i]
    #         ex_sum_matrix = torch.matmul(net_ex,net_ex_T)/ex_connectivity[i].shape[3]
    #         net_in_T = torch.transpose(in_connectivity[i], 2, 3)
    #         net_in = in_connectivity[i]
    #         net_in_martrix =  torch.matmul(net_in,net_in_T)/in_connectivity[i].shape[3]
    #         matrix_variance_tmp = net_in_martrix- ex_sum_matrix
    #         matrix_variance.append(matrix_variance_tmp)
    #     return matrix_variance
    # def viarance_compute(self, matrix_variance_gen, matrix_variance_target):
    #     loss = list()
    #     for i in range(len(matrix_variance_gen)):
    #         loss_tmp = F.mse_loss(matrix_variance_gen[i], matrix_variance_target[i])
    #         loss.append(loss_tmp)
    #     loss = sum(loss)
    #     return loss

    def generate_target_label(self, labels):
        if opt.n_classes == 2:
           labels[labels==1] = 2
           labels[labels == 0] = 1
           labels[labels == 2] = 0
        elif opt.n_classes == 3:
           for  i in range(labels.shape[0]):
                  _, label_i = torch.transpose(labels[i].unsqueeze(1), 1,0).topk(k=1, dim=1, largest=True)
                  if label_i == 0 :
                      labels[i] = torch.tensor([0.0, 1.0, 0.0], dtype =torch.float32)
                  elif label_i == 1 :
                      labels[i] = torch.tensor([0.0, 0.0, 1.0], dtype =torch.float32)
                  elif label_i == 2 :
                      labels[i] = torch.tensor([1.0, 0.0, 0.0], dtype =torch.float32)
               
        return labels
    def perceptual_loss_viarance(self, matrix_variance_gen, matrix_variance_target):
        loss = list()
        for i in range(len(matrix_variance_gen)):
            img_vgg_input = torch.repeat_interleave(matrix_variance_gen[i], 3, dim=3)
            img_vgg_input = torch.repeat_interleave(img_vgg_input, 3, dim=2)
            fmap_vgg_input = torch.repeat_interleave(matrix_variance_target[i],3, dim=3)
            fmap_vgg_input = torch.repeat_interleave(fmap_vgg_input, 3, dim=2)
            if img_vgg_input.shape[1] == 1:
                # handle grayscale for vgg
                img_vgg_input, fmap_vgg_input = map(lambda t: repeat(t, 'b 1 ... -> b c ...', c=3),
                                                    (img_vgg_input, fmap_vgg_input))
            vgg_ = self.vgg()
            img_vgg_feats = vgg_(img_vgg_input)
            recon_vgg_feats = vgg_(fmap_vgg_input)
            perceptual_loss = F.mse_loss(img_vgg_feats, recon_vgg_feats)
            loss.append(perceptual_loss)
        loss = sum(loss)
        return loss


    # def in_ex_connectivity_patch(self, array):
    #     network_region_wise = [16,32,33,30,21,34] #region wise of the networks
    #     ex_connectivity = list()
    #     in_connectivity = list()
    #     ex_conectivity_cellebrum = array[:,:,:network_region_wise[0],network_region_wise[0]:]
    #     in_conectivity_cellebrum = array[:,:,:network_region_wise[0],:network_region_wise[0]]
    #     ex_connectivity.append(ex_conectivity_cellebrum)
    #     in_connectivity.append(in_conectivity_cellebrum)
    #     for i in range(1,len(network_region_wise)):
    #         if i<len(network_region_wise)-1:
    #             start_index = sum(network_region_wise[:i])
    #             ex_conectivity_tmp = torch.cat((array[:,:,start_index :start_index+network_region_wise[i],:start_index],
    #                                            array[:, :,start_index:start_index+network_region_wise[i],start_index+network_region_wise[i]:]),dim=3)
    #             in__conectivity_tmp = array[:,:,start_index :start_index+network_region_wise[i],start_index:start_index+network_region_wise[i]]
    #             ex_connectivity.append(ex_conectivity_tmp)
    #             in_connectivity.append(in__conectivity_tmp)
    #         else:
    #             start_index = sum(network_region_wise[:i])
    #             ex_conectivity_SMA = array[:, :, start_index:,:start_index]
    #             in_conectivity_SMA = array[:, :, start_index:, start_index:]
    #             ex_connectivity.append(ex_conectivity_SMA)
    #             in_connectivity.append(in_conectivity_SMA)
    #     return ex_connectivity, in_connectivity
    #
    # def compute_viarance_main(self, gen, target):
    #     gen_connectivity_ex, gen_connectivity_in = self.in_ex_connectivity_patch(gen)
    #     target_connectivity_ex, target_connectivity_in = self.in_ex_connectivity_patch(target)
    #     gen_viarance_matrix = self.matrix_viarance(gen_connectivity_ex, gen_connectivity_in)
    #     target_viarance_matrix = self.matrix_viarance(target_connectivity_ex, target_connectivity_in)
    #     loss_viarance = self.viarance_compute(gen_viarance_matrix,target_viarance_matrix )
    #     loss_percep = self.perceptual_loss_viarance(gen_viarance_matrix,target_viarance_matrix)
    #     return loss_viarance + loss_percep
    # def exchange_perceptual_loss(self, gen, target):
    #     mask_inter = torch.zeros((gen.shape[0],1,160, 160))
    #     mask_externel = torch.ones((gen.shape[0], 1, 160, 160))
    #     mask_inter[:, :, 0:16, 0:16] = 1
    #     mask_externel[:, :, 0:16, 0:16] = 0
    #     mask_inter[:, :, 17:49, 17:49] = 1
    #     mask_externel[:, :, 17:49, 17:49] = 0
    #     mask_inter[:, :, 50:83, 50:83] = 1
    #     mask_externel[:, :, 50:83, 50:83] = 0
    #     mask_inter[:, :, 84:104, 84:104] = 1
    #     mask_externel[:, :, 84:104, 84:104] = 0
    #     mask_inter[:, :, 105:126, 105:126] = 1
    #     mask_externel[:, :, 105:126, 105:126] = 0
    #     mask_inter[:, :, 126:, 126:] = 1
    #     mask_externel[:, :, 105:126, 105:126] = 0
    #     mask_inter[:, :, 126:, 126:] = 1
    #     gen_inter = mask_inter.cuda() * gen
    #     gen_externel = mask_externel.cuda()* gen
    #     target_inter = mask_inter.cuda() * target
    #     target_externel = mask_externel.cuda()* target
    #     gen_in_tar_ex = gen_inter + target_externel
    #     gen_ex_tar_in = gen_externel + target_inter
    #     loss_1, loss_discr_1 = self.loss_function(gen_in_tar_ex, gen_ex_tar_in)
    #     loss_2, loss_discr_2 = self.loss_function(gen_ex_tar_in, gen_in_tar_ex)
    #     loss = loss_1 + loss_2
    #     loss_discr = loss_discr_1+loss_discr_2
    #     return loss, loss_discr
    #
    # def strength_consistency(self,gen, target):
    #     new_gen = torch.zeros((gen.shape[0],1,160, 160))
    #     new_target = torch.zeros((gen.shape[0],1,160, 160))
    #     new_gen[:,:,0:16,0:16]=gen[:,:,0:16,0:16]#cellebrum
    #     new_gen[:, :, 17:49, 17:49] = gen[:, :, 17:49, 17:49]#CON
    #     new_gen[:, :, 50:83, 50:83] = gen[:, :, 50:83, 50:83]  # DMN
    #     new_gen[:, :, 84:104, 84:104] = gen[:, :, 84:104, 84:104]  # OCN
    #     new_gen[:, :, 105:126, 105:126] = gen[:, :, 105:126, 105:126]  # OCN
    #     new_gen[:, :, 126:, 126:] = gen[:, :, 126:, 126:]  # SMA
    #     new_target[:,:,0:16,0:16]=target[:,:,0:16,0:16]#cellebrum
    #     new_target[:, :, 17:49, 17:49] = target[:, :, 17:49, 17:49]#CON
    #     new_target[:, :, 50:83, 50:83] = target[:, :, 50:83, 50:83]  # DMN
    #     new_target[:, :, 84:104, 84:104] = target[:, :, 84:104, 84:104]  # OCN
    #     new_target[:, :, 105:126, 105:126] = target[:, :, 105:126, 105:126]  # OCN
    #     new_target[:, :, 126:, 126:] = target[:, :, 126:, 126:]  # SMA
    #     return new_gen, new_target
    def loss_function(self, gen, target):
        # gen.detach_()
        # target.requires_grad_()
        img_vgg_input = target
        fmap_vgg_input = gen
        fmap_discr_logits, img_discr_logits = map(self.discriminator, (gen, target))
        # gp = gradient_penalty(target, img_discr_logits)
        discr_loss = bce_discr_loss(fmap_discr_logits, img_discr_logits)
        # discr_loss = discr_loss
        gen_loss = hinge_gen_loss(self.discriminator(gen))

        if target.shape[1] == 1:
            # handle grayscale for vgg
            img_vgg_input, fmap_vgg_input = map(lambda t: repeat(t, 'b 1 ... -> b c ...', c=3),
                                                (img_vgg_input, fmap_vgg_input))
        vgg_ = self.vgg()
        img_vgg_feats = vgg_(img_vgg_input)
        recon_vgg_feats = vgg_(fmap_vgg_input)
        perceptual_loss = F.mse_loss(img_vgg_feats, recon_vgg_feats)
        recon_loss = F.mse_loss(gen, target)
        loss = recon_loss + gen_loss + perceptual_loss
        return loss,discr_loss
    def forward(self, x):
        # self.optimizer_D.zero_grad()
        # self.optimizer_G.zero_grad()
        x_res = x
        # x_res_1 = x
        # adversarial_loss = torch.nn.BCELoss()
        # cuda = torch.cuda.is_available()
        # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        # valid = Tensor(x_res[0].shape[0], 1).fill_(1.0).requires_grad_(False)
        # fake = Tensor(x_res[0].shape[0], 1).fill_(0.0).requires_grad_(False)
        # z = Tensor(np.random.normal(0, 1, (x_res[0].shape[0], opt.latent_dim)))
        # gen_fc = self.generator(z)
        # loss = adversarial_loss(self.discriminator(gen_fc), valid)
        # real_loss = adversarial_loss(self.discriminator(x_res[1]), valid)
        # fake_loss = adversarial_loss(self.discriminator(gen_fc.detach()), fake)
        # discr_loss = (real_loss + fake_loss) / 2
        # if 'DMN' in self.opt.mask_option:
        #     img_new=x_res[0][0]
        #     # img_new[:,:,51:84, 51:84] = 0
        # if 'OCN' in self.opt.mask_option:
        #     img_new = x_res[0][0]
        #     # img_new[:,:,18:52, 18:52] = 0
        # if 'FPN' in self.opt.mask_option:
        #     img_new = x_res[0][0]
            # img_new[:,:,110:131, 110:131] = 0
        gen_fc,label_prompt = self.generator(x_res[0][0], x_res[2])
        #

        # loss, gen_fc= self.vae(x_res, return_loss =True, return_recons =True)
        # gen_fc = self.vae.decode(self.vae.encode(x_res[0]))
        loss, discr_loss = self.loss_function(gen_fc, x_res[0][1])
        n_epochs = opt.n_epochs_pretrain
        try:
            resume_path = OsJoin(opt.root_path, opt.result_path, opt.data_type, 'pretrained classifier',
                                 opt.model_name + str(opt.model_depth),
                                 'weights_%s_fold%s_%s_epoch%d' % (opt.category, 1, opt.features, n_epochs),
                                 '{}{}_weights_fold{}_epoch{}.pth'.format(opt.model_name, opt.model_depth, 1,
                                                                          n_epochs))
            checkpoint = torch.load(resume_path)
            opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
            assert opt.arch == checkpoint['arch']
            new_state_dict = OrderedDict()
            for k,v in checkpoint['state_dict'].items():
                    name = k[7:]
                    new_state_dict[name]=v
            self.classifier.load_state_dict(new_state_dict)
        except FileNotFoundError or IOError:
            n_epochs = n_epochs - 10
            resume_path = OsJoin(opt.root_path, opt.result_path, opt.data_type, opt.mode_net,
                                 opt.model_name + str(opt.model_depth),
                                 'weights_%s_fold%s_%s_epoch%d' % (opt.category, 1, opt.features, n_epochs),
                                 '{}{}_weights_fold{}_epoch{}.pth'.format(opt.model_name, opt.model_depth, 1,
                                                                          n_epochs))
            checkpoint = torch.load(resume_path)
            opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
            assert opt.arch == checkpoint['arch']
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:]
                new_state_dict[name] = v
            self.classifier.load_state_dict(new_state_dict)
        # checkpoint = torch.load(resume_path)
        # opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
        # assert opt.arch == checkpoint['arch']
        # self.classifier.load_state_dict(resume_path)
        source_cognitive_feature, target_cognitive_feature = self.discriminator([gen_fc, (gen_fc + label_prompt)])
        loss_dirsc_total = bce_discr_loss_label(source_cognitive_feature, target_cognitive_feature)
        with torch.no_grad():
            label_gen = self.classifier((gen_fc+label_prompt).unsqueeze(4))
            target_label = self.generate_target_label(x[1])
            label_source = self.classifier(source_cognitive_feature.unsqueeze(4))
            label_traget_gen = self.classifier(target_cognitive_feature.unsqueeze(4))
        loss_ce = self.criterion(label_gen, target_label)
        loss_ce_dirsc = self.criterion(label_traget_gen, target_label)
        loss_dirsc_label = self.criterion(label_source, x[1])
        discr_loss =  discr_loss + loss_dirsc_total + loss_dirsc_label + loss_ce_dirsc
        loss = loss + loss_ce
        # gen_fc=self.vae.decode(self.vae.encode(x_res[0]))
        # discr_loss = self.vae(x_res, return_discr_loss =True)
        # discr_loss.backward(retain_graph=True)
        # in_loss,in_loss_D = self.loss_function(gen_fc,x_res[0][0])
        # loss_viarance = self.compute_viarance_main(gen_fc,x_res[0][0])
        # exchange_loss, exchange_loss_discr = self.exchange_perceptual_loss(gen_fc,x_res[0][0])
        # loss = loss + loss_viarance + exchange_loss
        # discr_loss = discr_loss + exchange_loss_discr
        # with torch.no_grad():
        #     checkpoint = torch.load(opt.resume_path)
        #     opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
        #     assert opt.arch == checkpoint['arch']
        #     new_state_dict = OrderedDict()
        #     for k,v in checkpoint['state_dict'].items():
        #         name = k[7:]
        #         new_state_dict[name]=v
        #     self.classifier.load_state_dict(new_state_dict)
        #     gen_data=self.classifier(gen_fc.unsqueeze(4))
        # true_data = self.classifier(x_res[0][0].unsqueeze(4))
        # _,True_label = true_data.topk(k=1, dim=1, largest=True)
        # global_loss, global_loss_discr= self.loss_function(gen_fc,x_res[0][0])
        # net_gen, net_target = self.strength_consistency(gen_fc,x_res[0][0])
        # intra_net_loss, intra_net_discr= self.loss_function(net_gen.cuda(),net_target.cuda())
        # loss = global_loss
        # discr_loss = global_loss_discr
        # loss, gen_fc=self.vae(x_res[0][0], return_loss =True, return_recons=True)
        # discr_loss = self.vae(x_res[0][0], return_discr_loss=True)
        # loss = self.criterion(gen_data, true_data.cuda())
        # step = 0 ;
        # gen_fc.detach_()
        # x_res[0][1].requires_grad_()
        # img_vgg_input = x_res[0][1]
        # fmap_vgg_input = gen_fc
        # fmap_discr_logits, img_discr_logits = map(self.Transformer_discriminator, (gen_fc, x_res[0][1]))
        # gp = gradient_penalty(x_res[0][1], img_discr_logits)
        # discr_loss = bce_discr_loss(fmap_discr_logits, img_discr_logits)
        # discr_loss = discr_loss+gp
        # gen_loss = hinge_gen_loss(self.Transformer_discriminator(gen_fc))
        #
        # if x_res[0][1].shape[1] == 1:
        #     # handle grayscale for vgg
        #     img_vgg_input, fmap_vgg_input = map(lambda t: repeat(t, 'b 1 ... -> b c ...', c=3),
        #                                         (img_vgg_input, fmap_vgg_input))
        # vgg_ = self.vgg()
        # img_vgg_feats = vgg_(img_vgg_input)
        # recon_vgg_feats = vgg_(fmap_vgg_input)
        # perceptual_loss = F.mse_loss(img_vgg_feats, recon_vgg_feats)
        # last_dec_layer = self.Transformer_decode.last_dec_layer()
        #
        # norm_grad_wrt_gen_loss = grad_layer_wrt_loss(gen_loss, last_dec_layer).norm(p = 2)
        # norm_grad_wrt_perceptual_loss = grad_layer_wrt_loss(perceptual_loss, last_dec_layer).norm(p = 2)
        # adaptive_weight = safe_div(norm_grad_wrt_perceptual_loss, norm_grad_wrt_gen_loss)
        # adaptive_weight.clamp_(max = 1e4)
        #
        # recon_loss = F.mse_loss(gen_fc, x_res[0][1])
        # loss = recon_loss + gen_loss + perceptual_loss
        # while (sum(True_label != generate_label)>2 and step<100):
        # print('generating same label FC')
        # gen_fc.detach_()
        # img_vgg_input = x_res[0][1]
        # # loss_cls = self.criterion(gen_data ,true_data.cuda())
        # # loss_cls.backward(retain_graph=True)
        # fmap_vgg_input = gen_fc
        # fmap_discr_logits, img_discr_logits = map(self.Transformer_discriminator, (gen_fc, x_res[0][1]))
        # discr_loss = bce_discr_loss(fmap_discr_logits, img_discr_logits)
        # gen_loss = hinge_gen_loss(self.Transformer_discriminator(gen_fc))
        # if x_res[0][1].shape[1] == 1:
        #     # handle grayscale for vgg
        #     img_vgg_input, fmap_vgg_input = map(lambda t: repeat(t, 'b 1 ... -> b c ...', c=3),
        #                                         (img_vgg_input, fmap_vgg_input))
        # vgg_ =self.vgg()
        # img_vgg_feats = vgg_(img_vgg_input)
        # recon_vgg_feats = vgg_(fmap_vgg_input)
        # perceptual_loss = F.mse_loss(img_vgg_feats, recon_vgg_feats)
        # recon_loss = F.mse_loss(gen_fc, x_res[0][1])
        # loss = recon_loss + gen_loss + perceptual_loss
        # #In summary, the code is preparing the optimizer for a new iteration of training by resetting the gradients to zero
        # loss.backward()
        # self.optimizer_G.step()
        # loss_cls.backward()
        #
        # discr_loss.backward()
        # self.optimizer_D.step()
        # gen_fc = self.generator(x_res[0][1])
        # with torch.no_grad():
        #     gen_data = self.classifier(gen_fc.unsqueeze(4))
        #     _,generate_label = gen_data.topk(k=1, dim=1, largest=True)
        # step+=1
        return loss, discr_loss, gen_fc,label_gen, gen_fc+label_prompt



class text_image_generator(nn.Module):
    def  __init__(self, block, layers,opt, shortcut_type='B', num_classes=400, last_fc=True, dim=256):
        super(text_image_generator, self).__init__()
        self.avgpool_choose = avgpool_choose(opt)
        self.opt = opt
        self.dim = dim
        self.criterion = torch.nn.CrossEntropyLoss()
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.vae = VQGanVAE(dim=self.dim)
        self.vae_pretrian_path = opt.pretrained_vae
        self.classifier = classifier(block, layers, opt, shortcut_type=shortcut_type, num_classes=num_classes)
        self.transformer =  MaskGitTransformer(
                num_tokens = 65536,       # must be same as codebook size above
                seq_len = 256,            # must be equivalent to fmap_size ** 2 in vae
                dim = 512,                # model dimension
                depth = 8,                # depth
                dim_head = 64,            # attention head dimension
                heads = 8,                # attention heads,
                ff_mult = 4,              # feedforward expansion factor
                t5_name = 't5-small' )    # name of your T5
        self.base_maskgit = MaskGit(
            vae=self.vae,  # vqgan vae
            transformer=self.transformer,  # transformer
            image_size=160,  # image size
            cond_drop_prob=0.25,  # conditional dropout, for classifier free guidance
        ).cuda()

    def forward(self, x):
        checkpoint = torch.load(opt.pretrained_vae)
        opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
        assert opt.arch == checkpoint['arch']
        # self.vae.load_state_dict(checkpoint['state_dict'])
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k
            if 'ResNet' in name or 'classifier' in name:
                continue
            name = k[11:]
            new_state_dict[name] = v
        self.vae.load_state_dict(new_state_dict)
        x_res = x.copy()
        # for x in x_res[0]:
            # torch.cuda.empty_cache()
            # shape_res_W = x.shape[3]
        images = x_res[0][1]
        texts = x_res[2]
        loss = self.base_maskgit (images, texts=texts)
        if opt.category == 'MCI_SCD':
            texts = ['MCI to SCD in two class situation',
                     'SCD to MCI in two class situation']
            target_label =[[0,1],[1,0]]
        elif opt.category == 'HC_SCD':
            texts = ['HC to SCD in two class situation',
                     'SCD to HC in two class situation']
            target_label = [[0, 1], [1, 0]]
        elif opt.category == 'HC_MCI':
            texts = ['HC to MCI in two class situation',
                     'MCI to HC in two class situation']
            target_label = [[0, 1], [1, 0]]
        elif opt.category == 'HC_MCI_SCD':
            texts = ['HC to SCD in two class situation',
                     'SCD to MCI in two class situation',
                     'MCI to HC in two class situation']
            target_label = [[0,0,1], [0,1, 0], [1,0,0]]
        images = self.base_maskgit.generate(texts=texts,
                                       cond_scale=3.)
        with torch.no_grad():
            checkpoint = torch.load(opt.resume_path)
            opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
            assert opt.arch == checkpoint['arch']
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:]
                new_state_dict[name] = v
            self.classifier.load_state_dict(new_state_dict)
            x = self.classifier(images.unsqueeze(4))
        loss_ce = self.criterion(x, torch.FloatTensor(target_label).cuda())

        return loss_ce+loss,x,images

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

def main(**kwargs):
    if opt.mode_net == 'pretrained classifier':
        model = classifier(BasicBlock, [1, 1, 1, 1], **kwargs)
    elif opt.mode_net == 'image_generator':
        model = image_generator(opt)
    elif opt.mode_net == 'text-image generator':
        model = text_image_generator(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model
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
