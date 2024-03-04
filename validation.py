import torch
from torch.autograd import Variable
import time
import os
from utils import OsJoin
import matplotlib.pyplot as plt
from utils import AverageMeter, calculate_accuracy, calculate_recall,generate_target_label,generate_neurodegeneration
import numpy as np

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
def val_epoch(epoch,data_loader,model,criterion,opt,logger, writer,optimizer_G, optimizer_D):
    print('validation at epoch {}'.format(epoch) )
    model.eval()
    if opt.mode_net == 'image_generator':
       model.train()

    batch_time =AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_discr = AverageMeter()
    accuracies = AverageMeter()
    recalls = AverageMeter()
    precisions = AverageMeter()
    f1s = AverageMeter()
    sensitivitys = AverageMeter()
    specificitys = AverageMeter()
    writer = writer
    SFC_gen_total =torch.zeros((1,160,160))
    gen_target_total = torch.zeros((1, 160, 160))
    if opt.n_classes == 3:
        labels_total = torch.zeros((1,3))
    elif opt.n_classes == 2:
        labels_total = torch.zeros((1,2))
    end_time = time.time()

    # labels_arr = torch.empty(4).cuda()
    # pred_arr = torch.empty(4, 1).cuda()
    writer_index = np.random.randint(1,len(data_loader),size=1)
    for i, (inputs,labels,target_FC) in enumerate(data_loader):
        # if i > 50:
        #     continue
        data_time.update(time.time() - end_time)
        # labels = list(map(int,labels))
        # inputs= (torch.unsqueeze(input,1) for input in inputs)
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
        # inputs = ( input.permute(0,3,1,2) for input in inputs)
        #inputs = inputs.type(torch.FloatTensor)
        # labels = labels.type(torch.FloatTensor).cuda()
        # if not opt.no_cuda:
        #     labels = torch.LongTensor(labels).cuda(non_blocking=True)#async to non_blocking
        # with torch.no_grad():
            #inputs = Variable(inputs)
        # inputs = (Variable(input) for input in inputs)
        # labels = Variable(labels)
        # inputs = list(inputs)
        #outputs_add = torch.zeros(inputs[0].shape[0], opt.n_classes).cuda()
        #outputs_array = torch.zeros(opt.num_of_feature, inputs[0].shape[0], opt.n_classes)
        # inputs_1 = [inputs[0], inputs[1], inputs[2]]
        # inputs_2 = [inputs[5], inputs[3], inputs[4]]
        # inputs = [inputs_1, inputs_2]
        #inputs_1 = [inputs[0], inputs[2], inputs[3]]
        # features_dict = ['ALFF','DFC', 'FA', 'FC']
        # features_select = opt.features.split('_')
        # indexs = (features_dict.index(feature) for feature in features_select)
        # inputs_1 = (inputs[index] for index in indexs)
        # inputs = [list(inputs)]
        if opt.mode_net == 'pretrained classifier' or opt.mode_net == 'region-specific':
            loss, outputs = model([inputs,labels])
        elif opt.mode_net == 'image_generator' or opt.mode_net == 'text-image generator' :
            loss, loss_discr, SFC_gen, outputs,gen_target = model([inputs,labels,target_FC])
            SFC_gen_total = torch.concatenate([SFC_gen_total.to(SFC_gen.device),SFC_gen.squeeze()],dim=0)
            gen_target_total = torch.concatenate([gen_target_total.to(gen_target), gen_target.squeeze()], dim=0)
            labels_total = torch.concatenate([labels_total.to(labels), labels], dim=0)
            if len(SFC_gen.squeeze().shape) == 3:
               gen_smaple = SFC_gen.squeeze()[0, :, :,...].cpu().detach().numpy()
            else:
                gen_smaple = SFC_gen.squeeze().cpu().detach().numpy()
            # gen_smaple = np.transpose(gen_smaple, (1, 2, 0))
            if len(SFC_gen.squeeze().shape)  == 3:
               target_smaple = inputs_target.squeeze()[0, :, :,...].cpu().detach().numpy()
            else:
                 target_smaple = inputs_target.squeeze().cpu().detach().numpy()
            if len(SFC_gen.squeeze().shape) == 3:
                noise_smaple = inputs_noise.squeeze()[0, :, :, ...].cpu().detach().numpy()
            else:
                noise_smaple = inputs_noise.squeeze().cpu().detach().numpy()
            if len(SFC_gen.squeeze().shape) == 3:
                gen_target_smaple = gen_target.squeeze()[0, :, :, ...].cpu().detach().numpy()
                sub_sample = normalize(gen_target_smaple) - normalize(gen_smaple)
            else:
                gen_target_smaple = gen_target.squeeze().cpu().detach().numpy()
                sub_sample = normalize(gen_target_smaple) - normalize(gen_smaple)
            # target_smaple = np.transpose(target_smaple, (1, 2, 0))
            # if 'DMN' in opt.mask_option:
            #     target_smaple[51:84, 51:84] = 0
            # if 'OCN' in opt.mask_option:
            #     target_smaple[ 18:52, 18:52] = 0
            # if 'FPN' in opt.mask_option:
            #     target_smaple[110:131, 110:131] = 0
            # image = Image.fromarray(gen_smaple)
            result_path = OsJoin(opt.root_path, opt.result_path)
            save_path = OsJoin(result_path, opt.data_type, opt.mode_net, 'gen images')
            if opt.mode_net == 'text-image generator':
                save_path = OsJoin(result_path, opt.data_type, 'total', 'gen images')

            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if epoch %50 ==0:
                save_name = OsJoin(save_path, 'validation_epoch%d_batch%d_gen.png' % (epoch, i + 1))
                plt.imshow(gen_smaple)
                plt.axis('off')
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.savefig(save_name)
                plt.close()
                save_name = OsJoin(save_path, 'validation_epoch%d_batch%d_gen_target.png' % (epoch, i + 1))
                plt.imshow(gen_target_smaple)
                plt.axis('off')
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.savefig(save_name)
                plt.close()
                save_name = OsJoin(save_path, 'validation_epoch%d_batch%d_target.png' % (epoch, i + 1))
                plt.imshow(target_smaple)
                plt.axis('off')
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.savefig(save_name)
                plt.close()
                save_name = OsJoin(save_path,'validation_epoch%d_batch%d_noise.png' % (epoch,i+1))
                plt.imshow(noise_smaple)
                plt.axis('off')
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.savefig(save_name)
                plt.close()
                save_name = OsJoin(save_path,'validation_epoch%d_batch%sub.png' % (epoch,i+1))
                plt.imshow(sub_sample)
                plt.axis('off')
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.savefig(save_name)
                plt.close()
            # outputs = torch.zeros(opt.num_of_feature, 3).cuda()
            # for input in inputs:
            #     output_tmp = model(input)
            #     outputs = outputs + output_tmp
            # inputs_fmri = (inputs[0], inputs[5])
            # inputs_fc = (inputs[1], inputs[3])
            # inputs_dti = (inputs[2], inputs[4])
            # i = 0
            # for input in inputs_fmri:
            #     output_tmp = model(input)
            #     outputs_add = outputs_add + output_tmp
            #     outputs_array[i, :, :] = output_tmp
            #     i = i + 1
            # for input in inputs_fc:
            #     output_tmp = model(input)
            #     outputs_add = outputs_add + output_tmp
            #     outputs_array[i, :, :] = output_tmp
            #     i = i + 1
            # for input in inputs_dti:
            #     output_tmp = model(input)
            #     outputs_add = outputs_add + output_tmp
            #     outputs_array[i, :, :] = output_tmp
            #     i = i + 1
            #outputs_mutiply = torch.ones(inputs[0].shape[0], opt.n_classes).cuda()
            #outputs = model(inputs)
            #loss = criterion(outputs, labels)
        if opt.mode_net == 'pretrained classifier' or opt.mode_net == 'region-specific':
            acc = calculate_accuracy(outputs, labels)
            recall, precision, f1, sensitivity, specificity = calculate_recall(outputs,
                                                                               labels, opt)
        # elif opt.mode_net == 'text-image generator':
        elif opt.mode_net == 'image_generator':
            acc = calculate_accuracy(outputs, generate_target_label(labels,opt))
            recall, precision, f1, sensitivity, specificity = calculate_recall(outputs,
                                                                               generate_target_label(labels, opt),
                                                                               opt)
        # elif opt.mode_net == 'text-image generator':
        #     if opt.category == 'MCI_SCD':
        #         target_label = [[0, 1], [1, 0]]
        #     elif opt.category == 'HC_SCD':
        #         target_label = [[0, 1], [1, 0]]
        #     elif opt.category == 'HC_MCI':
        #         target_label = [[0, 1], [1, 0]]
        #     elif opt.category == 'HC_MCI_SCD':
        #         target_label = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
        #     acc = calculate_accuracy(outputs, torch.FloatTensor(target_label).cuda())
        #     recall, precision, f1, sensitivity, specificity = calculate_recall(outputs, torch.FloatTensor(target_label).cuda(), opt)

        losses.update(loss.data, inputs[0].size(0))
        if opt.mode_net == 'image_generator':
            losses_discr.update(loss_discr.data, inputs[0].size(0))
        accuracies.update(acc, inputs[0].size(0))
        recalls.update(recall, inputs[0].size(0))
        precisions.update(precision, inputs[0].size(0))
        f1s.update(f1, inputs[0].size(0))
        sensitivitys.update(sensitivity, inputs[0].size(0))
        specificitys.update(specificity, inputs[0].size(0))
#        aucs.update(auc, inputs[0][0].size(0))
        if opt.mode_net == 'image_generator':
            if i % 2 == 0:
                optimizer_G.zero_grad()
                loss.backward(retain_graph=True)
                optimizer_G.step()
            else:
                optimizer_D.zero_grad()
                loss_discr.backward(retain_graph=True)
                # if epoch % 3 == 0:
                optimizer_D.step()
        batch_time.update(time.time() - end_time)
        end_time = time.time()
# need to check the data,the data of loss is the format of tensor array
        if opt.mode_net == 'pretrained classifier' or opt.mode_net == 'region-specific':
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  # 'Loss_discr {loss_discr.val[0]:.4f} ({loss_discr.avg[0]:.4f})\t'
                  'Acc {acc.val:.4f} ({acc.avg:.4f})\t'
                   'Recall {recall.val:.4f} ({recall.avg:.4f})\t'
                  'Precision {precision.val:.4f} ({precision.avg:.4f})\t\n'
                  'f1 {f1.val:.4f} ({f1.avg:.4f})\t'
                  'sensitivity {sensitivity.val:.4f} ({sensitivity.avg:.4f})\t'
                  'specificity {specificity.val:.4f} ({specificity.avg:.4f})\t'
            .format(
                epoch,
                i + 1,
                len(data_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                acc=accuracies,
                recall=recalls,
                precision=precisions,
                f1=f1s,
                sensitivity=sensitivitys,
                specificity=specificitys
            ))
        elif opt.mode_net == 'text-image generator' or opt.mode_net == 'image_generator':
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss_discr {loss_discr.val:.4f} ({loss_discr.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})\t'
                   'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                  'Precision {precision.val:.3f} ({precision.avg:.3f})\t\n'
                  'f1 {f1.val:.3f} ({f1.avg:.3f})\t'
                  'sensitivity {sensitivity.val:.3f} ({sensitivity.avg:.3f})\t'
                  'specificity {specificity.val:.3f} ({specificity.avg:.3f})\t'
            .format(
                epoch,
                i + 1,
                len(data_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                loss_discr=losses_discr,
                acc=accuracies,
                recall=recalls,
                precision=precisions,
                f1=f1s,
                sensitivity=sensitivitys,
                specificity=specificitys
            ))
            if i %writer_index ==0:
                writer.add_scalar('val/loss_G', losses.avg.cpu(),  i+(epoch-1)*len(data_loader))
                # print(i+(epoch-1)*len(data_loader))
                writer.add_scalar('val/loss_D', losses_discr.avg, i+(epoch-1)*len(data_loader))
                writer.add_figure('val/gen_image', plt_image(gen_smaple), i+(epoch-1)*len(data_loader))
                plt.close()
                writer.add_figure('val/gen_target_image', plt_image(gen_target_smaple), i+(epoch-1)*len(data_loader))
                plt.close()
                writer.add_figure('val/sub_image', plt_image(sub_sample), i+(epoch-1)*len(data_loader))
                plt.close()
                writer.add_figure('val/noise_image', plt_image(noise_smaple), i+epoch)
                plt.close()
                writer.add_figure('val/target_image', plt_image(target_smaple), i+(epoch-1)*len(data_loader))
                plt.close()
        # _, pred = outputs.topk(k=1, dim=1, largest=True)
        # print('prediction :', end=' ')
        # for i in range(0, len(pred)):
        #    print('%d\t' % (pred[i]), end='')
        # print('\nlabel    :', end=' ')
        # for i in range(0, len(labels)):
        #    print('%d\t' % (labels[i]), end='')
        # print('\n')
    #     pred_arr = torch.cat([pred_arr, pred], dim=0)
    #     _, labels_ = labels.topk(k=1, dim=1, largest=True)
    #     labels_arr = torch.cat([labels_arr, labels_.cuda().squeeze()], dim=0)
    # print('prediction :', end=' ')
    # for i in range(4, len(pred_arr)):
    #     print('%d\t'%(pred_arr[i]), end='')
    # print('\nlabel    :', end=' ')
    # for i in range(4, len(labels_arr)):
    #     print('%d\t'%(labels_arr[i]), end='')
    # print('\n')
    #
    # labels_arr = torch.empty(4).cuda()
    # pred_arr = torch.empty(4, 1).cuda()
    generate_neurodegeneration(SFC_gen_total[1:,...], gen_target_total[1:,...], labels_total[1:,...], opt, epoch, mode='validation')
    if opt.mode_net == 'pretrained classifier' or opt.mode_net == 'region-specific':
        logger.log({'epoch': epoch, 'loss': round(losses.avg.item(), 4),
                    'acc': round(accuracies.avg.item(), 4), 'recall': round(recalls.avg.item(), 4),
                   'precision': round(precisions.avg.item(), 4),  'f1': round(f1s.avg.item(), 4),
                   'sensitivity': round(sensitivitys.avg.item(), 4), 'specificity': round(specificitys.avg.item(), 4)
                    } )
    elif opt.mode_net == 'image_generator':
        logger.log({'epoch': epoch, 'loss_G': round(losses.avg.item(), 4), 'loss_D': round(losses_discr.avg.item(), 4),
                    'acc': round(accuracies.avg.item(), 4), 'recall': round(recalls.avg.item(), 4),
                   'precision': round(precisions.avg.item(), 4),  'f1': round(f1s.avg.item(), 4),
                   'sensitivity': round(sensitivitys.avg.item(), 4), 'specificity': round(specificitys.avg.item(), 4)
                    } )
    return losses.avg,losses_discr.avg
