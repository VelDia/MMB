#!/usr/bin/env python3

'''
Title: YOLOrs: Object Detection in Multimodal RemoteSensing Imagery
Authors: Manish Sharma, Mayur Dhanaraj, Srivallabha Karnam, Dimitris G. Chachlakis, Raymond Ptucha, Panos P. Markopoulos, Eli Saber
Date: 
URL: 
'''

### Classes in dataset ###
# 0: car, 1: pickup, 2: campingcar, 3: truck, 4: other, 5: tractor, 
# 6: boat, 7: van, 8: plane, 9: motorcycle, 10: bus

### Load helper scripts ###
import os
if os.getcwd().endswith('/bash/train'):
    os.chdir(os.getcwd()[:-11])
exec(open('helper.py').read())

### Input important parameters
parser = argparse.ArgumentParser()
parser.add_argument('--mods', type=str, default=['RGB','I','RGBI_0','RGBI_5','RGBI_12','RGBI_19','RGBI_28','RGBI_37'],nargs='+', help='model names')
parser.add_argument('--batch_size', type=int, default=5, help='size of each image batch')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--Alr_type', type=str, default='rop', help='adaptive learning rate type')
parser.add_argument('--Alr_coef', type=float, default=0.1, help='adaptive learning rate coefficient')
parser.add_argument('--reg', type=float, default=0.001, help='regularizer')
parser.add_argument('--n_classes', type=int, default=4, help='number of classes')
parser.add_argument('--aug', type=int, default=1, help='online image augmentation')
parser.add_argument('--pre', type=int, default=0, help='use pre-trained/checkpoint weights as initialization')
parser.add_argument('--folds', type=int, default=10, help='number of cross-validation folds')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--cpt_int', type=int, default=[100], nargs='+', help='interval between saving model weights')
parser.add_argument('--grad_accum', type=int, default=2, help='number of gradient accums before step')
parser.add_argument('--eval_int', type=int, default=1, help='interval evaluations on validation set')
parser.add_argument('--fl', type=str, default='3_1_10', help='Focal loss gamma value, obj weight and no-obj weight, if gamma = 0 then std BCELoss')
parser.add_argument('--act_fn', type=str, default='leaky_0.1', help='activation function for feature extraction layers')
parser.add_argument('--data_config', type=str, default='config/custom.data', help='path to data config file')
parser.add_argument('--model_def', type=str, default='config/yolors.cfg', help='path to model definition file')
parser.add_argument('--pre_path', type=str, default='weights/yolors.weights', help='if specified starts from pre-trained/checkpoint weights')
parser.add_argument('--multiscale_training', default=False, help='allow for multi-scale training')
parser.add_argument('--n_cpu', type=int, default=1, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=512, help='size of each image dimension')
parser.add_argument('--test', type=int, default=0, help='size of each image dimension')
parser.add_argument('--fusion_type', type=str, default='conc', help='Select fusion type: [conc, xpro, add, sub]')
parser.add_argument('--bias', type=str, default='center', help='Bias RGB(left) or I(right) or balanced(center)')
parser.add_argument('--nA', type=int, default=1, help='number of anchors per detection head')
opt = parser.parse_args()
print('\nModels: {}'.format(opt.mods))
# opt.pre_path='checkpoints/{}_b=5(2)_lr=0.001_Alr=(rop_g=0.1)_reg=0.001_c=8_aug=1_pre=0_fl=(3_1_10)_act=(leaky_0.1)_nA=1_cv=({}-10)_E=(100-100).pth'

########################Loop over models###############################
for name in opt.mods:
    exec(open('models.py').read())  ### load models script
    cif = 'RGBI'.index(name.split('_')[0][0]), 'RGBI'.index(name.split('_')[0][-1]) + 1 ### channels to use
    if name not in ['RGB','I','RGBI_0']:
        mod_name = (name.split('_')[0] + '_' + opt.fusion_type + 
                    ('({})'.format(opt.bias[0]) if opt.fusion_type in ['conc','xpro'] else '') + '_' + name.split('_')[-1])
    else:
        mod_name = name
        
    if opt.pre_path.endswith('.pth'):
        pre_epoch = int(opt.pre_path.split('E=(')[1].split('-')[0])
        lr_str = opt.pre_path.split('_Alr=(')[0].split('=')[-1]
        pre_str = 0
    else:
        pre_epoch = 0
        lr_str = str(opt.lr)
        pre_str = opt.pre
    
    name_str = ('{}_b={}({})_lr={}_Alr=({}_g={})_reg={}_c={}_aug={}_pre={}_fl=({})_act=({})_nA={}_cv=({})'
                .format(mod_name,opt.batch_size,opt.grad_accum,lr_str,opt.Alr_type,
                        opt.Alr_coef,opt.reg,opt.n_classes,opt.aug,pre_str,
                        opt.fl,opt.act_fn,opt.nA,opt.folds))
    
    if opt.pre_path.endswith('.pth'):
        pre_format = opt.pre_path
        name_str = '{}_E=({})'.format(name_str,int(opt.pre_path.split('E=(')[1].split('-')[1].split(')')[0]))
    else:
        name_str = '{}_E=({})'.format(name_str,opt.epochs)
    try:
        wb = load_workbook('results/{}.xlsx'.format(name_str))
        df,dft0,dft1,dft2 = [wb[s].values for s in wb.sheetnames if 'Train' in s]
        df,dft0,dft1,dft2 = [pd.DataFrame(s,columns=next(s)[0:]) for s in [df,dft0,dft1,dft2]]
        df,dft0,dft1,dft2 = [s.fillna(value=pd.np.nan) for s in [df,dft0,dft1,dft2]]
        dfv0,dfv1,dfv2 = [wb[s].values for s in wb.sheetnames if '_Test' in s]
        dfv0,dfv1,dfv2 = [pd.DataFrame(s,columns=next(s)[0:]) for s in [dfv0,dfv1,dfv2]]
        dfv0,dfv1,dfv2 = [s.fillna(value=pd.np.nan) for s in [dfv0,dfv1,dfv2]]
        wb.close()
    except:
        df = pd.DataFrame()
        dft0,dft1,dft2 = pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
        dfv0,dfv1,dfv2 = pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    print('\n' + name_str)
    ##########################Loop over folds############################
    for f in range(1,opt.folds+1):
        if opt.pre_path.endswith('.pth'):
            opt.pre_path = pre_format.format(mod_name,f)
            print(opt.pre_path)
            opt.pre = 1
        try:
            del model, optimizer, scheduler, dataset, dataloader, loss, params
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass
        model, params = load_models(name, opt)
        optimizer = torch.optim.Adam(params, lr=opt.lr, weight_decay=opt.reg)            
        if opt.Alr_type=='exp':
            scheduler = ExponentialLR(optimizer, gamma=opt.Alr_coef)
        elif opt.Alr_type=='step':
            scheduler = MultiStepLR(optimizer, gamma=opt.Alr_coef,
                                    milestones= [int(i*opt.epochs) for i in np.linspace(0.5,0.8,num=2)])
        elif opt.Alr_type=='rop':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=opt.Alr_coef, patience=15, 
                                          cooldown=5, threshold=1e-4, verbose=True)
        if opt.pre_path.endswith('.pth'):
            opt_dict = torch.load(opt.pre_path.replace('/','/optimizers/'))
            optimizer.load_state_dict(opt_dict)
            print('\nOptimizer state loaded successfully')
            if opt.Alr_type in ['exp','step','rop']:
                sch_dict = torch.load(opt.pre_path.replace('/','/schedulers/'))
                scheduler.load_state_dict(sch_dict)
                print('\nScheduler state loaded successfully')
            del opt_dict, sch_dict
            gc.collect()
            torch.cuda.empty_cache()
            
        # Get data configuration
        data_config = parse_data_config(opt.data_config)
        # train_path = data_config['train'].split('.')
        train_path = data_config['train']
        print(train_path)
        # train_path = train_path[0] + '_fold{}.'.format(f) +  train_path[1]
        # test_path = data_config['test'].split('.')
        test_path = data_config['test']
        print(test_path)
        # test_path = test_path[0] + '_fold{}.'.format(f) +  test_path[1]
        class_names = load_classes(data_config['names'])
        class_names = [class_names[i] for i in range(opt.n_classes)] ### Choosen classes
        
        # Get dataloader
        dataset = ListDataset(train_path, augment=bool(opt.aug), multiscale=opt.multiscale_training)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=opt.batch_size,
                                                 shuffle=True,
                                                 num_workers=opt.n_cpu,
                                                 pin_memory=True,
                                                 collate_fn=dataset.collate_fn,)
        # # Get data configuration
        # data_config = parse_data_config(opt.data_config)
        # train_folder = data_config['train']  # Folder containing training images
        # test_folder = data_config['test']    # Folder containing test images
        # class_names_path = data_config['names']
        # train_label_folder = data_config['train']
        # print(train_label_folder)
        # # Load class names
        # class_names = load_classes(class_names_path)
        # class_names = [class_names[i] for i in range(opt.n_classes)]  # Chosen classes

        # # Initialize datasets
        # dataset = ListDataset(train_folder, 
        #                             augment=bool(opt.aug), multiscale=opt.multiscale_training)
        # # test_dataset = ListDataset(test_folder, label_folder='path/to/test_labels',
        # #                             augment=False, multiscale=False)
        # print(len(dataset))
        # # Create DataLoaders
        # dataloader = torch.utils.data.DataLoader(dataset,
        #                                             batch_size=opt.batch_size,
        #                                             shuffle=True,
        #                                             num_workers=opt.n_cpu,
        #                                             pin_memory=True,
        #                                             collate_fn=dataset.collate_fn)
        # # test_dataloader = torch.utils.data.DataLoader(test_dataset,
        # #                                             batch_size=opt.batch_size,
        # #                                             shuffle=False,
        # #                                             num_workers=opt.n_cpu,
        # #                                             pin_memory=True,
        # #                                             collate_fn=test_dataset.collate_fn)

        # name_str1 = name_str_old.replace('cv=(','cv=({}-'.format(f)).replace('E=(','E=({}-'.format(0))
        # torch.save(model.state_dict(),'checkpoints/{}.pth'.format(name_str1))
        #######################################################################
        for epoch in range(opt.epochs):
            model.train()
            start_time = time.time()
            dft= pd.DataFrame()
            counter = 0
            ###################################################################
            for batch_i, (_, imgs, targets) in enumerate(dataloader):
                # plt.imshow(imgs[0,:3,:,:].permute(1,2,0))
                targets = targets[targets[:,1] < opt.n_classes]
                if len(targets)>0:
                    if len(np.unique(targets[:,0])) < opt.batch_size:
                        del imgs, targets
                        continue
                else:
                    del imgs, targets
                    continue
                counter += 1
                batches_done = len(dataloader) * epoch + batch_i
                imgs = Variable(imgs[:,cif[0]:cif[1],:,:].to(device))
                targets = Variable(targets.to(device), requires_grad=False)
                loss, outputs = model(imgs, targets)
                loss.backward()
                if batches_done % opt.grad_accum:
                    optimizer.step()
                    optimizer.zero_grad()
                dft= dft.add(data_collect(model,pre_epoch+epoch+1), fill_value=0)
                epoch_batches_left = len(dataloader) - (batch_i + 1)
                time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
                print('\n{} | CV:{}/{} | Epoch:{}/{} | Batch:{}/{} | Loss:{} | ETA {}'.
                      format(mod_name,f,opt.folds,epoch+1,opt.epochs,batch_i+1,len(dataloader),round(loss.item(),5),time_left))
                del imgs, targets, outputs
                # print('\n' + str(torch.cuda.memory_allocated(0)/1024**2))
                # print('\n' + str(torch.cuda.memory_cached(0)/1024**2))
            dft= dft/counter
            # dft0= dft0.append(dft.iloc[[0]],sort=False).reset_index(drop=True)
            dft0 = pd.concat([dft0, dft.iloc[[0]]], ignore_index=True)
            # dft1= dft1.append(dft.iloc[[1]],sort=False).reset_index(drop=True)
            dft1 = pd.concat([dft1, dft.iloc[[1]]], ignore_index=True)
            # dft2= dft2.append(dft.iloc[[2]],sort=False).reset_index(drop=True)
            dft2 = pd.concat([dft2, dft.iloc[[2]]], ignore_index=True)
            if (pre_epoch+epoch+1) % opt.eval_int == 0:
                print('\n---- Evaluating Model ----')
                stats,dfv = evaluate(model,
                                     path=test_path,
                                     iou_thres=0.2,
                                     conf_thres=0.7,
                                     nms_thres=0.1,
                                     opt=opt,
                                     cif=cif,
                                     class_names=class_names,
                                     epoch= pre_epoch+epoch+1,
                                     device=device,
                                     stat=True)
                # dft0= dft0.append(dft.iloc[[0]],sort=False).reset_index(drop=True)
                dft0 = pd.concat([dft0, dft.iloc[[0]]], ignore_index=True)
                # dft1= dft1.append(dft.iloc[[1]],sort=False).reset_index(drop=True)
                dft1 = pd.concat([dft1, dft.iloc[[1]]], ignore_index=True)
                # dft2= dft2.append(dft.iloc[[2]],sort=False).reset_index(drop=True)
                dft2 = pd.concat([dft2, dft.iloc[[2]]], ignore_index=True)
                stats['mTrainLoss']= dft['loss'].sum()
                # df = df.append(stats,sort=False).reset_index(drop=True)
                df = pd.concat([df, stats], ignore_index=True)
            
            if opt.Alr_type in ['step','exp']:
                scheduler.step(pre_epoch+epoch+1)
            elif opt.Alr_type == 'rop':
                scheduler.step(stats.loc[0,'mTestLoss'])
            print('\nLr: {}'.format(optimizer.param_groups[0]['lr']))

            if (pre_epoch+epoch+1) in opt.cpt_int:
                name_str1 = name_str.replace('cv=(','cv=({}-'.format(f)).replace('E=(','E=({}-'.format(epoch+1+pre_epoch))
                torch.save(model.state_dict(),'checkpoints/{}.pth'.format(name_str1))
                # torch.save(scheduler.state_dict(),'checkpoints/schedulers/{}.pth'.format(name_str1))
                # torch.save(optimizer.state_dict(),'checkpoints/optimizers/{}.pth'.format(name_str1))
            
            df = df.groupby('epoch',as_index=False).mean()
            dft0 = dft0.groupby('epoch',as_index=False).mean()
            dft1 = dft1.groupby('epoch',as_index=False).mean()
            dft2 = dft2.groupby('epoch',as_index=False).mean()
            if opt.pre_path.endswith('.pth'):
                dfv0 = dfv0.groupby('epoch',as_index=False).mean()
                dfv1 = dfv1.groupby('epoch',as_index=False).mean()
                dfv2 = dfv2.groupby('epoch',as_index=False).mean()
            
            try:
                wb = load_workbook('results/{}.xlsx'.format(name_str))
                wsheets = [s for s in wb.sheetnames if 'TestStats' not in s]
                for s in wsheets:
                    std = wb[s]
                    wb.remove(std)
                wb.create_sheet()
                wb.save('results/{}.xlsx'.format(name_str))
                wb.close()
                mode = 'a'
            except:
                mode = 'w'
            with pd.ExcelWriter('results/{}.xlsx'.format(name_str),mode=mode) as writer:
                df.to_excel(writer,index=False,sheet_name='TrainStats')
                dft0.to_excel(writer,index=False,sheet_name='S_Train')
                dft1.to_excel(writer,index=False,sheet_name='M_Train')
                dft2.to_excel(writer,index=False,sheet_name='L_Train')
                dfv0.to_excel(writer,index=False,sheet_name='S_Test')
                dfv1.to_excel(writer,index=False,sheet_name='M_Test')
                dfv2.to_excel(writer,index=False,sheet_name='L_Test')
                ws = writer.sheets['TrainStats']
                chart = ScatterChart('smooth')
                xvalues = Reference(ws, min_col=df.columns.get_loc('epoch')+1, min_row=2, max_row=(opt.epochs+1+pre_epoch))
                values1 = Reference(ws, min_col=df.columns.get_loc('mTrainLoss')+1, min_row=2, max_row=(opt.epochs+1+pre_epoch))
                values2 = Reference(ws, min_col=df.columns.get_loc('mTestLoss')+1, min_row=2, max_row=(opt.epochs+1+pre_epoch))
                series1 = Series(values=values1, xvalues=xvalues, title ="Training loss")
                series2 = Series(values=values2, xvalues=xvalues, title ="Testing loss")
                chart.append(series1)
                chart.append(series2)
                chart.title = 'Training and Testing Loss'
                chart.legend.layout = Layout(ManualLayout(yMode='edge',xMode='edge',x=0.78, y=0.15))
                chart.layout = Layout(ManualLayout(x=0.2, y=0.15, h=0.7, w=0.88,layoutTarget='inner',xMode="edge",yMode="edge"))
                chart.x_axis.scaling.min = 0
                chart.y_axis.scaling.min = 0
                chart.x_axis.scaling.max = opt.epochs + pre_epoch
                chart.x_axis.majorTickMark = 'cross'
                chart.y_axis.majorTickMark = 'cross'
                chart.x_axis.title = 'Epoch'
                chart.y_axis.title = 'Loss'
                ws.add_chart(chart,'BQ{}'.format(2))
            plt.plot(df['epoch'],df['mTrainLoss'])
            plt.plot(df['epoch'],df['mTestLoss'])
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training & Testing Loss')
            plt.legend(['Training Loss','Testing Loss'])
            plt.tight_layout()
            plt.savefig('results/{}.png'.format(name_str))
            plt.close()