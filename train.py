import torch
from torch.autograd import Variable
import os
import io
from PIL import Image
import numpy as np
from utils import OsJoin
import time
import matplotlib.pyplot as plt
from models.my_model  import generator, Transformer_dirsc
from utils import AverageMeter,calculate_accuracy,generate_target_label,generate_neurodegeneration

def plt_image(array):
    figure = plt.figure()
    plt.imshow(array)
    plt.axis('off')
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # buf.seek(0)
    # image = Image.open(buf)
    plt.close(figure)
    return figure
def normalize(array):
    max_ = np.max(np.max(array))
    min_ = np.min(np.min(array))
    return (array-max_)/(max_-min_)
def train_epoch(epoch, fold_id, data_loader, model, criterion,
                opt, epoch_logger, batch_logger, writer,optimizer_G, optimizer_D):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses= AverageMeter()
    losses_discr = AverageMeter()
    accuracies = AverageMeter()
    writer = writer
    end_time = time.time()
    labels_arr = torch.empty(4).cuda()
    pred_arr = torch.empty(4, 1).cuda()
    SFC_gen_total =torch.zeros((1,160,160))
    gen_target_total = torch.zeros((1, 160, 160))
    if opt.n_classes == 3:
        labels_total = torch.zeros((1,3))
    elif opt.n_classes == 2:
        labels_total = torch.zeros((1,2))
    # optimizer_G.zero_grad()
    # optimizer_D.zero_grad()
    writer_index = np.random.randint(1,len(data_loader),size=1)
    for i ,(inputs,labels, target_FC) in enumerate(data_loader):
        # if i > 50:
        #     continue
        torch.cuda.empty_cache()
        data_time.update(time.time()-end_time)
        # labels = list(map(int, labels))
        # inputs= (torch.unsqueeze(input,1) for input in inputs)
        # inputs = ( input.permute(0,3,1,2) for input in inputs)
        #inputs = torch.unsqueeze(inputs,1)  #在 1 的位置加一个维度
        # labels = labels.type(torch.FloatTensor).cuda()
        # inputs_noise = (input.type(torch.FloatTensor) for input in inputs[0][0])
        # inputs_target = (input.type(torch.FloatTensor) for input in inputs[0][1])
       # inputs = (input.type(torch.Float64) for input in inputs)
       #  if not opt.no_cuda:
       #      labels = torch.LongTensor(labels).cuda(non_blocking = True)
       #  inputs = (Variable(input) for input in inputs)
        #inputs = Variable(inputs)
        if opt.n_classes == 3:
            labels = labels.repeat(1,inputs[0].shape[1]).view(-1,3)
        else:
            labels = labels.repeat(1, inputs[0].shape[1]).view(-1, 2)
        target_FC = target_FC.unsqueeze(1).repeat(1,inputs[0].shape[1],1,1).view(-1, inputs[0].shape[2], inputs[0].shape[3])
        inputs[0] = inputs[0].view(-1,inputs[0].shape[2], inputs[0].shape[3])
        inputs[1] = inputs[1].view(-1, inputs[1].shape[2], inputs[1].shape[3])
        labels = labels.type(torch.FloatTensor)
        target_FC = target_FC.type(torch.FloatTensor).unsqueeze(1)
        inputs_noise = inputs[0].type(torch.FloatTensor).unsqueeze(1)
        inputs_target = inputs[1].type(torch.FloatTensor).unsqueeze(1)
        inputs = [inputs_noise, inputs_target]
        # labels = Variable(labels)
        # outputs_add=torch.zeros(inputs[0].shape[0], 3, opt.n_classes).cuda()
        # outputs_mutiply = torch.ones(inputs[0].shape[0], opt.n_classes).cuda()
        # outputs_array = torch.zeros(opt.num_of_feature,inputs[0].shape[0], opt.n_classes)
        # inputs_fmri=(,inputs[5])
        # inputs_fc=(inputs[1],inputs[3])
        # inputs_dti=(inputs[2],inputs[4])
        # i=0
    #     features_dict = ['DFC','FC']
    #     features_select = opt.features.split('_')
    # #    indexs = []
    #     indexs = (features_select.index(feature) for feature in features_dict)
    #     inputs_1 = (inputs[index] for index in indexs)
        #inputs_1=[inputs[0], inputs[2], inputs[3]]
#        inputs_2 = [inputs[5], inputs[3], inputs[4]]
#         inputs = [list(inputs)]
        if opt.mode_net == 'pretrained classifier' or opt.mode_net == 'region-specific':
           loss, outputs = model([inputs,labels])
        elif opt.mode_net == 'image_generator' or opt.mode_net == 'text-image generator':
            loss, loss_discr, SFC_gen, outputs, gen_target = model([inputs,labels,target_FC])
            SFC_gen_total = torch.concatenate([SFC_gen_total.to(SFC_gen.device),SFC_gen.squeeze()],dim=0)
            gen_target_total = torch.concatenate([gen_target_total.to(gen_target), gen_target.squeeze()], dim=0)
            labels_total = torch.concatenate([labels_total.to(labels), labels], dim=0)
            if len(SFC_gen.squeeze().shape) == 3:
               gen_smaple = SFC_gen.squeeze()[0, :, :,...].cpu().detach().numpy()
            else:
                gen_smaple = SFC_gen.squeeze().cpu().detach().numpy()
            # gen_smaple = np.transpose(gen_smaple, (1, 2, 0))
            if len(SFC_gen.squeeze().shape)  == 3:
               target_smaple = inputs[1].squeeze()[0, :, :,...].cpu().detach().numpy()
            else:
                 target_smaple = inputs[1].squeeze().cpu().detach().numpy()
            if len(SFC_gen.squeeze().shape) == 3:
                noise_smaple = inputs_noise.squeeze()[0, :, :, ...].cpu().detach().numpy()
            else:
                noise_smaple = inputs_noise.squeeze().cpu().detach().numpy()
            if len(SFC_gen.squeeze().shape) == 3:
                gen_target_smaple = gen_target.squeeze()[0, :, :, ...].cpu().detach().numpy()
                sub_sample = normalize(gen_target_smaple) - normalize(gen_smaple)
            else:
                gen_target_smaple = gen_target_smaple.squeeze().cpu().detach().numpy()
                sub_sample = normalize(gen_target_smaple) - normalize(gen_smaple)
            # gen_smaple = SFC_gen.squeeze()[0,:,:,...].cpu().detach().numpy()
            # gen_smaple = np.transpose(gen_smaple, (1, 2, 0))
            # target_smaple = inputs[0][0].squeeze()[0, :, :,...].cpu().detach().numpy()
            # target_smaple = np.transpose(target_smaple, (1,2,0))
            # if 'DMN' in opt.mask_option:
            #     target_smaple[51:84, 51:84] = 0
            # if 'OCN' in opt.mask_option:
            #     target_smaple[18:52, 18:52] = 0
            # if 'FPN' in opt.mask_option:
            #     target_smaple[110:131, 110:131] = 0
            # image = Image.fromarray(gen_smaple)
            result_path = OsJoin(opt.root_path, opt.result_path)
            save_path = OsJoin(result_path, opt.data_type, opt.mode_net,'gen images')
            if opt.mode_net == 'text-image generator':
                save_path = OsJoin(result_path, opt.data_type, 'total', 'gen images')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if epoch %50 ==0:
                save_name = OsJoin(save_path,'train_epoch%d_batch%d_gen.png' % (epoch,i+1))
                plt.imshow(gen_smaple)
                plt.axis('off')
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.savefig(save_name)
                plt.close()
                save_name = OsJoin(save_path,'train_epoch%d_batch%d_gen_target.png' % (epoch,i+1))
                plt.imshow(gen_target_smaple)
                plt.axis('off')
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.savefig(save_name)
                plt.close()
                save_name = OsJoin(save_path,'train_epoch%d_batch%d_target.png' % (epoch,i+1))
                plt.imshow(target_smaple)
                plt.axis('off')
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.savefig(save_name)
                plt.close()
                save_name = OsJoin(save_path,'train_epoch%d_batch%d_noise.png' % (epoch,i+1))
                plt.imshow(noise_smaple)
                plt.axis('off')
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.savefig(save_name)
                plt.close()
                save_name = OsJoin(save_path,'train_epoch%d_batch%d_sub.png' % (epoch,i+1))
                plt.imshow(sub_sample)
                plt.axis('off')
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.savefig(save_name)
                plt.close()
                # generate_neurodegeneration(SFC_gen, gen_target, labels, opt, epoch, i, mode='train')
                # HC_dedeneration,MCI = generate_neurodegeneration(SFC_gen,gen_target,labels,opt)

        # if len(outputs) == 1:
        #     loss = criterion(outputs, labels)
        # else:
        #     loss_cl=criterion(outputs[0],outputs[1])
        #     loss_ce=criterion(outputs[2],labels)
        #     loss=loss_cl+loss_ce
       # inputs=[inputs_1, inputs_2]
       #  outputs = torch.zeros(opt.num_of_feature, 3).cuda()
       #  for input in inputs:
       #      output_tmp = outputs
       #      outputs = outputs + output_tmp
        # for input in inputs_fmri:
        #     output_tmp=model(input)
        #     outputs_add=outputs_add+output_tmp
        #     outputs_mutiply=outputs_mutiply *output_tmp
        #     outputs_array[i,:,:]=output_tmp
        #     i=i+1
        # for input in inputs_fc:
        #         output_tmp = model(input)
        #         outputs_add = outputs_add + output_tmp
        #         outputs_mutiply = outputs_mutiply * output_tmp
        #         outputs_array[i, :, :] = output_tmp
        #         i = i + 1
        # for input in inputs_dti:
        #         output_tmp = model(input)
        #         outputs_add = outputs_add + output_tmp
        #         outputs_mutiply = outputs_mutiply * output_tmp
        #         outputs_array[i, :, :] = output_tmp
        #         i = i + 1

            #outputs = model(inputs)
#        outputs_list=torch.from_numpy(np.array(outputs_list)).cuda()
        #loss = criterion(outputs,labels)
        if opt.mode_net == 'pretrained classifier' or opt.mode_net == 'region-specific':
            acc = calculate_accuracy(outputs, labels)
        elif opt.mode_net == 'image_generator' :
            acc = calculate_accuracy(outputs, generate_target_label(labels,opt))
            # acc = 1
        elif opt.mode_net == 'text-image generator':
            if opt.category == 'MCI_SCD':
                target_label = [[0, 1], [1, 0]]
            elif opt.category == 'HC_SCD':
                target_label = [[0, 1], [1, 0]]
            elif opt.category == 'HC_MCI':
                target_label = [[0, 1], [1, 0]]
            elif opt.category == 'HC_MCI_SCD':
                target_label = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
            acc = calculate_accuracy(outputs,torch.FloatTensor(target_label).cuda())
        losses.update(loss.data,inputs[0].size(0))
        # if opt.mode_net == 'image_generator':

        accuracies.update(acc, inputs[0].size(0))
        if opt.mode_net == 'pretrained classifier' or opt.mode_net == 'region-specific':
            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()
            batch_logger.log({
                'epoch': epoch,
                'batch': i + 1,
                'iter': (epoch - 1) * len(data_loader) + (i - 1),
                'loss_G': round(losses.avg.item(), 4),
                # 'loss_D': round(losses_discr.avg.item(), 4),
                'acc': round(accuracies.val.item(), 4),
                'lr': optimizer_G.param_groups[0]['lr']
            })
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'ACC {acc.val:.4f} ({acc.avg:.4f})\t'
            .format(
                epoch, i + 1, len(data_loader), batch_time=batch_time,
                data_time=data_time, loss=losses,acc=accuracies))
        elif opt.mode_net == 'image_generator':
            losses_discr.update(loss_discr.data, inputs[0].size(0))
            if i%2==0:
                optimizer_G.zero_grad()
                loss.backward(retain_graph=True)
                optimizer_G.step()
                # optimizer_D.zero_grad()
            else:
                optimizer_D.zero_grad()
                loss_discr.backward(retain_graph=True)
                # if epoch % 3 == 0:
                optimizer_D.step()
            batch_logger.log({
                'epoch': epoch,
                'batch': i + 1,
                'iter': (epoch - 1) * len(data_loader) + (i - 1),
                'loss_G': round(losses.avg.item(), 4),
                'loss_D': round(losses_discr.avg.item(), 4),
                'acc': round(accuracies.val.item(), 4),
                'lr': optimizer_G.param_groups[0]['lr']
            })
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss_discr {loss_discr.val:.4f} ({loss_discr.avg:.4f})\t'.format(
                epoch, i + 1, len(data_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, loss_discr=losses_discr))
            if i % writer_index == 0:
                writer.add_scalar('train/loss_G', losses.avg, i + (epoch - 1) * len(data_loader))
                writer.add_scalar('train/loss_D', losses_discr.avg, i + (epoch - 1) * len(data_loader))
                # fig, ax
                # figure = plt.figure()
                # plt.imshow(gen_smaple)
                # plt.axis('off')
                # plt.gca().xaxis.set_major_locator(plt.NullLocator())
                # plt.gca().yaxis.set_major_locator(plt.NullLocator())
                writer.add_figure('train/gen_image', plt_image(gen_smaple), i + (epoch - 1) * len(data_loader))
                plt.close()
                writer.add_figure('train/gen_target_image', plt_image(gen_target_smaple), i + (epoch - 1) * len(data_loader))
                plt.close()
                writer.add_figure('train/sub_image', plt_image(sub_sample), i + (epoch - 1) * len(data_loader))
                plt.close()
                writer.add_figure('train/noise_image', plt_image(noise_smaple), i + epoch)
                plt.close()
                writer.add_figure('train/target_image', plt_image(target_smaple), i + (epoch - 1) * len(data_loader))
                plt.close()
        # batch_time.update(time.time()-end_time)
        end_time = time.time()



        # _, pred = outputs.topk(k=1, dim=1, largest=True)
        # pred_arr = torch.cat([pred_arr, pred], dim=0)
        # _, labels_ = labels.topk(k=1, dim=1, largest=True)
        # labels_arr = torch.cat([labels_arr, labels_.cuda().squeeze()], dim=0)
    # print('prediction :', end=' ')
    # for i in range(4, len(pred_arr)):
    #     print('%d\t'%(pred_arr[i]), end='')
    # print('\nlabel    :', end=' ')
    # for i in range(4, len(labels_arr)):
    #     print('%d\t'%(labels_arr[i]), end='')
    # print('\n')
    # labels_arr = torch.empty(4).cuda()
    # pred_arr = torch.empty(4, 1).cuda()
    generate_neurodegeneration(SFC_gen_total[1:,...], gen_target_total[1:,...], labels_total[1:,...], opt, epoch, mode='train')
    if opt.mode_net == 'pretrained classifier':
        epoch_logger.log({
            'epoch': epoch,
            'loss': round(losses.avg.item(), 4),
            # 'loss_D': round(losses_discr.avg.item(), 4),
            'acc': round(accuracies.avg.item(), 4),
            'lr': optimizer_G.param_groups[0]['lr']
        })
    elif opt.mode_net == 'image_generator':
        epoch_logger.log({
            'epoch': epoch,
            'loss_G': round(losses.avg.item(), 4),
            'loss_D': round(losses_discr.avg.item(), 4),
            'acc': round(accuracies.avg.item(), 4),
            'lr': optimizer_G.param_groups[0]['lr']
        })
    # writer.add_scalar('train/loss_G', losses.avg.cpu().detach().numpy(), epoch)
    # writer.add_scalar('train/loss_D', losses_discr.avg.cpu().detach().numpy(), epoch)
    # writer.add_image('train/noise_image', noise_smaple[0,...].squeeze().cpu().detach().numpy(), epoch)
    # writer.add_image('train/target_image', target_smaple[0,...].squeeze().cpu().detach().numpy(), epoch)
    # writer.add_image('train/gen_image', SFC_gen[0,...].squeeze().cpu().detach().numpy(), epoch)
    # writer.close()
    # writer.add_scalar('train/accuracy', accuracies.avg, epoch)
    if opt.mode_net =="pretrained classifier" or opt.mode_net == 'region-specific':
        checkpoint =20
    elif  opt.mode_net == 'image_generator':
        checkpoint = 100
    if opt.save_weight:
        if epoch % checkpoint == 0:
            if opt.mode_net == 'pretrained classifier' or opt.mode_net == 'region-specific':
                save_dir = OsJoin(opt.result_path, opt.data_type, opt.mode_net, opt.model_name + str(opt.model_depth),
                                 'weights_%s_fold%s_%s_epoch%d' % (opt.category, str(fold_id), opt.features, opt.n_epochs))
            elif opt.mode_net == 'image_generator':
                save_dir =OsJoin(opt.result_path, opt.data_type, opt.mode_net, opt.model_name + str(opt.model_depth),
                                 'weights_%s_fold%s_%s_epoch%d' % (opt.category, str(fold_id), opt.features, opt.n_epochs))
            elif opt.mode_net == 'text-image generator':
                save_dir =OsJoin(opt.result_path, opt.data_type, 'total', opt.model_name + str(opt.model_depth),
                                 'weights_%s_fold%s_%s_epoch%d' % (opt.category, str(fold_id), opt.features, opt.n_epochs))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = OsJoin(save_dir,
                        '{}{}_weights_fold{}_epoch{}.pth'.format(opt.model_name, opt.model_depth, fold_id, epoch))
            states = {
                'fold': fold_id,
                'epoch': epoch,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer_G.state_dict(),
            }
            torch.save(states, save_path)



