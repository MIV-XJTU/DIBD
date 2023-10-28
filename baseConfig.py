import torch
import torch.optim as optim
import models
import os,logging
from os.path import join
from utility import *
from utility.ssim import SSIMLoss,SAMLoss
from ptflops import get_model_complexity_info

import time
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

count = 0
idx = 0

def options(parser):
    def _parse_str_args(args):
        str_args = args.split(',')
        parsed_args = []
        for str_arg in str_args:
            arg = int(str_arg)
            if arg >= 0:
                parsed_args.append(arg)
        return parsed_args    
    parser.add_argument('--prefix', '-p', type=str, default='denoise',
                        help='prefix')
    parser.add_argument('--arch', '-a', metavar='ARCH', required=True,
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names))
    parser.add_argument('--batchSize', '-b', type=int,
                        default=16, help='training batch size. default=16')         
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate. default=1e-3.')
    parser.add_argument('--wd', type=float, default=0,
                        help='weight decay. default=0')
    parser.add_argument('--loss', type=str, default='l2',
                        help='which loss to choose.', choices=['l1', 'l2', 'smooth_l1', 'ssim', 'l2_ssim','l2_sam','cons','cons_l2','char'])
    parser.add_argument('--testdir', type=str)
    parser.add_argument('--sigma', type=int)
    parser.add_argument('--init', type=str, default='kn',
                        help='which init scheme to choose.', choices=['kn', 'ku', 'xn', 'xu', 'edsr'])
    parser.add_argument('--no_cuda',action='store_true', help='disable cuda?')
    parser.add_argument('--no-log', action='store_true',
                        help='disable logger?')
    parser.add_argument('--threads', type=int, default=1,
                        help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=2018,
                        help='random seed to use. default=2018')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--no-ropt', '-nro', action='store_true',
                            help='not resume optimizer')          
    parser.add_argument('--chop', action='store_true',
                            help='forward chop')                                      
    parser.add_argument('--clip', type=float, default=1e6)
    parser.add_argument('--update_lr', type=float, default=0.5e-4, help='learning rate of inner loop')
    parser.add_argument('--meta_lr', type=float, default=0.5e-4, help='learning rate of outer loop')
    parser.add_argument('--n_way', type=int, default=1, help='the number of ways')
    parser.add_argument('--k_spt', type=int, default=2, help='the number of support set')
    parser.add_argument('--k_qry', type=int, default=5, help='the number of query set')
    parser.add_argument('--task_num', type=int, default=16, help='the number of tasks')
    parser.add_argument('--update_step', type=int, default=5, help='update step of inner loop in training')
    parser.add_argument('--update_step_test', type=int, default=10, help='update step of inner loop in testing')
    parser.add_argument('--resumePath', '-rp', type=str,
                        default=None, help='checkpoint to use.')
    parser.add_argument('--trainDataset', type=str,
                        default='icvl', help='traing data set')
    parser.add_argument('--trainDir', type=str,
                        default='/home/liuy/data/sst/ICVL/ori_datasets/train_data/train.db', help='traing data root')
    parser.add_argument('--testDir', type=str,
                        default='/home/liuy/data/sst/ICVL/ori_datasets/icvl_test_gaussian/512_50', help='test data root')
    parser.add_argument('--gpu_ids', '-gpu',type=str, default='0', help='gpu ids')
    parser.add_argument('--noiseType', '-nt', type=str,choices=['gaussian','complex','noniid'],
                        default="gaussian", help='checkpoint to use.')
    parser.add_argument('--test_matSize', '-tm', type=int,
                        default=1, help='test mat size to load')
    parser.add_argument('--update_lr_epoch', type=int, default=60, help='learning rate of inner loop')
    parser.add_argument('--epoch_per_save', '-ep', type=int,
                        default=10, help='checkpoint to use.')  
    
    opt = parser.parse_args()
    opt.gpu_ids = _parse_str_args(opt.gpu_ids)

    return opt


def make_dataset(opt, train_transform, target_transform, common_transform, batch_size=None, repeat=1):

    dataset = LMDBDataset(opt.dataroot, repeat=repeat)
    dataset = TransformDataset(dataset, common_transform)

    train_dataset = ImageTransformDataset(dataset, train_transform, target_transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size or opt.batchSize, shuffle=True,
                              num_workers=opt.threads, pin_memory=not opt.no_cuda, worker_init_fn=worker_init_fn)

    return train_loader


class Logger:
    def __init__(self, path,clevel = logging.DEBUG,Flevel = logging.DEBUG):
        self.logger = logging.getLogger(path)
        self.logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')

        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(clevel)

        fh = logging.FileHandler(path)
        fh.setFormatter(fmt)
        fh.setLevel(Flevel)
        self.logger.addHandler(sh)
        self.logger.addHandler(fh)
 
    def debug(self,message):
        self.logger.debug(message)
 
    def info(self,message):
        self.logger.info(message)
 
    def war(self,message):
        self.logger.warn(message)
 
    def error(self,message):
        self.logger.error(message)
 
    def cri(self,message):
        self.logger.critical(message)

class SequentialSelect(object):
    def __pos(self, n):
        i = 0
        while True:
            yield i
            i = (i + 1) % n

    def __init__(self, transforms):
        self.transforms = transforms
        self.pos = LockedIterator(self.__pos(len(transforms)))

    def __call__(self, img):
        out = self.transforms[next(self.pos)](img)
        return out

def get_train_transform(noiseType,HSI2Tensor):
    if noiseType == "gaussian":
        train_transform = Compose([
            AddNoiseBlindv1(10,70),
            HSI2Tensor()
        ])
    elif noiseType == "complex":
        sigmas = [10, 30, 50, 70]
        train_transform =  Compose([
            AddNoiseNoniid(sigmas),
            SequentialSelect(
                transforms=[
                    lambda x: x,
                    AddNoiseImpulse(),
                    AddNoiseStripe(),
                    AddNoiseDeadline()
                ]
            ),
            HSI2Tensor()
        ])
    elif noiseType == "noniid":
        train_transform = Compose([
            AddNoiseNoniid_v2(0,55),
            HSI2Tensor()
        ])        
    return train_transform

class Engine(object):
    def __init__(self, opt):
        self.prefix = opt.prefix
        self.opt = opt
        self.net = None
        self.optimizer = None
        self.criterion = None
        self.basedir = None
        self.iteration = None
        self.epoch = None
        self.best_psnr = None
        self.best_epoch = None
        self.best_loss = None
        self.writer = None
        self.trainDir = None
        self.logger = None

        self.__setup()

    def __setup(self):
        self.basedir = join('checkpoints', self.opt.arch)
        if not os.path.exists(self.basedir):
            os.makedirs(self.basedir)

        self.best_psnr = 0
        self.best_epoch = 0
        self.best_loss = 1e6
        self.epoch = 0
        self.iteration = 0

        cuda = not self.opt.no_cuda

        self.device = 'cuda:'+str(self.opt.gpu_ids[0]) if cuda else 'cpu'
        print('Cuda Acess: %d' % cuda)
        if cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")

        torch.manual_seed(self.opt.seed)
        if cuda:
            torch.cuda.manual_seed(self.opt.seed)

        """Model"""
        print("=> creating model '{}'".format(self.opt.arch))
        self.net = models.__dict__[self.opt.arch]()

        init_params(self.net, init_type=self.opt.init)

        if len(self.opt.gpu_ids) > 1:
            from models.sync_batchnorm import DataParallelWithCallback
            self.net = DataParallelWithCallback(self.net, device_ids=self.opt.gpu_ids)
        
        if self.opt.loss == 'l2':
            self.criterion = nn.MSELoss()
        if self.opt.loss == 'l1':
            self.criterion = nn.L1Loss()
        if self.opt.loss == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
        if self.opt.loss == 'ssim':
            self.criterion = SSIMLoss(data_range=1, channel=31)

        print(self.criterion)

        if cuda:
            print(self.device)
            self.net.to(self.device)
            self.criterion = self.criterion.to(self.device)

        """Logger Setup"""
        log = not self.opt.no_log
        if log:
            self.writer = get_summary_writer(os.path.join(self.basedir, 'logs'), self.opt.prefix)
        
        self.logger = Logger(os.path.join(self.basedir, 'logs/') + self.prefix+ str(time.strftime("%Y-%m-%d", time.localtime())) + '.log',logging.ERROR,logging.DEBUG)

        """Optimization Setup"""

        self.optimizer  = optim.AdamW(self.net.parameters(), lr=self.opt.lr, betas=(0.9, 0.999),eps=1e-8, weight_decay=1e-4)

        """Resume previous model"""
        if self.opt.resume:
            self.load(self.opt.resumePath)
        else:
            print('==> Building model..')

        total = sum([param.nelement() for param in self.net.parameters()])    
        print("Number of parameter: %.2fM" % (total/1e6))


    def reset_params(self):
        init_params(self.net, init_type=self.opt.init)

    def forward(self, inputs):        
        if self.opt.chop:            
            output = self.forward_chop(inputs)
        else:
            output = self.net(inputs)
        return output

    def __step(self, train, inputs, targets):        
        if train:
            self.optimizer.zero_grad()
        loss_data = 0
        total_norm = None
        self.net.eval()
        
        outputs , est_noisy, sigma, mu, var, rec_noisy=self.net(inputs)
        loss = self.criterion(outputs[...], targets)

        if train:
            loss.backward()
        loss_data += loss.item()
        if train:
            total_norm = nn.utils.clip_grad_norm_(self.net.parameters(), self.opt.clip)
            self.optimizer.step()

        return outputs, loss_data, total_norm

    def load(self, resumePath=None, load_opt=False):
        model_best_path = join(self.basedir, self.prefix, 'model_latest.pth')
        if os.path.exists(model_best_path):
            best_model = torch.load(model_best_path)

        print('==> Resuming from checkpoint %s..' % resumePath)
        assert os.path.isdir('checkpoints'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(resumePath or model_best_path)
        if load_opt:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.get_net().load_state_dict(checkpoint['net'])
        
    def judge(self, valid_loader, name,patch_size=64):
            self.net.eval()
            validate_loss = 0
            total_psnr = 0
            total_sam = 0
            RMSE = []
            SSIM = []
            SAM = []
            ERGAS = []
            PSNR = []
            print('[i] Eval dataset {}...'.format(name))
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(valid_loader):
                    
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = inputs
                    psnr_bands = cal_bwpsnr(outputs, targets)
                    psnr = np.mean(psnr_bands)
                    sam = cal_sam(outputs, targets)
                    validate_loss += 0
                    total_sam += sam
                    avg_loss = validate_loss / (batch_idx+1)
                    avg_sam = total_sam / (batch_idx+1)

                    total_psnr += psnr
                    avg_psnr = total_psnr / (batch_idx+1)

                    progress_bar(batch_idx, len(valid_loader), 'Loss: %.4e | PSNR: %.4f | AVGPSNR: %.4f '
                            % (avg_loss, psnr, avg_psnr))
                    
                    psnr = []
                    c,h,w=inputs.shape[-3:]
                    
                    result = outputs.squeeze().cpu().detach().numpy()
                    img = targets.squeeze().cpu().numpy()
                    for k in range(c):
                        psnr.append(10*np.log10((h*w)/sum(sum((result[k]-img[k])**2))))
                    PSNR.append(sum(psnr)/len(psnr))
                    
                    mse = sum(sum(sum((result-img)**2)))
                    mse /= c*h*w
                    mse *= 255*255
                    rmse = np.sqrt(mse)
                    RMSE.append(rmse)

                    ssim = []
                    k1 = 0.01
                    k2 = 0.03
                    for k in range(c):
                        ssim.append((2*np.mean(result[k])*np.mean(img[k])+k1**2) \
                            *(2*np.cov(result[k].reshape(h*w), img[k].reshape(h*w))[0,1]+k2**2) \
                            /(np.mean(result[k])**2+np.mean(img[k])**2+k1**2) \
                            /(np.var(result[k])+np.var(img[k])+k2**2))
                    SSIM.append(sum(ssim)/len(ssim))

                    temp = (np.sum(result*img, 0) + np.spacing(1)) \
                        /(np.sqrt(np.sum(result**2, 0) + np.spacing(1))) \
                        /(np.sqrt(np.sum(img**2, 0) + np.spacing(1)))
                    #print(np.arccos(temp)*180/np.pi)
                    sam = np.mean(np.arccos(temp))*180/np.pi
                    SAM.append(sam)

                    ergas = 0.
                    for k in range(c):
                        ergas += np.mean((img[k]-result[k])**2)/np.mean(img[k])**2
                    ergas = 100*np.sqrt(ergas/c)
                    ERGAS.append(ergas)
            
            print(sum(PSNR)/len(PSNR), sum(RMSE)/len(RMSE), sum(SSIM)/len(SSIM), sum(SAM)/len(SAM), sum(ERGAS)/len(ERGAS))
            
            print("avg_psnr: "+str(avg_psnr)+" avg_SSIM: "+str(sum(SSIM)/len(SSIM))+" avg_sam: "+str(avg_sam))
            return (avg_psnr, sum(SSIM)/len(SSIM),sum(SAM)/len(SAM))


    def train(self, train_loader,val):
        print('\nEpoch: %d' % self.epoch)
        self.net.train()
        train_loss = 0
        train_psnr = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):

            if not self.opt.no_cuda:
                inputs, targets = inputs.to(self.device), targets.to(self.device)            
            outputs, loss_data, total_norm = self.__step(True, inputs, targets)
            train_loss += loss_data
            avg_loss = train_loss / (batch_idx+1)
            psnr = np.mean(cal_bwpsnr(outputs, targets))
            train_psnr += psnr
            avg_psnr = train_psnr/ (batch_idx+1)
            if not self.opt.no_log:
                self.writer.add_scalar(
                    join(self.prefix, 'train_psnr'), avg_psnr, self.iteration)
                self.writer.add_scalar(
                    join(self.prefix, 'train_loss'), loss_data, self.iteration)
                self.writer.add_scalar(
                    join(self.prefix, 'train_avg_loss'), avg_loss, self.iteration)

            self.iteration += 1

            progress_bar(batch_idx, len(train_loader), 'AvgLoss: %.4e | Loss: %.4e | Norm: %.4e | Psnr: %.4f' 
                         % (avg_loss, loss_data, total_norm,psnr))
        self.logger.info('trainepoch: %s | AvgLoss: %.4f | Loss: %.4f | Norm: %.4f | Psnr: %.2f' 
                    % (self.epoch,avg_loss, loss_data, total_norm,avg_psnr))
        self.epoch += 1
        if not self.opt.no_log:
            self.writer.add_scalar(
                join(self.prefix, 'train_loss_epoch'), avg_loss, self.epoch)

 
    def test(self, valid_loader, filen):
        self.net.eval()
        validate_loss = 0
        total_psnr = 0
        total_sam = 0
        book = 0
        RMSE = []
        SSIM = []
        SAM = []
        ERGAS = []
        PSNR = []
        
        print('[i] Eval dataset ...')
                
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valid_loader):

                if book == 1:
                    B,C,H,W = inputs.shape
                    macs, params = get_model_complexity_info(self.net, (C,H,W),as_strings=True,
                                        print_per_layer_stat=False, verbose=False)
                    book = 0
                    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
                    print('{:<30}  {:<8}'.format('Number of parameters: ', params))  
                    self.logger.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
                    self.logger.info('{:<30}  {:<8}'.format('Number of parameters: ', params))  

                if not self.opt.no_cuda:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)  
                   
                outputs, loss_data, _ = self.__step(False, inputs, targets)
                psnr_bands = cal_bwpsnr(outputs, targets)
                psnr = np.mean(psnr_bands)
                sam = cal_sam(outputs, targets)
                
                validate_loss += loss_data
                total_sam += sam
                avg_loss = validate_loss / (batch_idx+1)
                avg_sam = total_sam / (batch_idx+1)

                total_psnr += psnr
                avg_psnr = total_psnr / (batch_idx+1)

                progress_bar(batch_idx, len(valid_loader), 'Loss: %.4e | PSNR: %.4f | AVGPSNR: %.4f '
                          % (avg_loss, psnr, avg_psnr))
                
                psnr = []
                c,h,w=inputs.shape[-3:]
                result = outputs.squeeze().cpu().detach().numpy()
            
                img = targets.squeeze().cpu().numpy()
                
                for k in range(c):
                    psnr.append(10*np.log10((h*w)/sum(sum((result[k]-img[k])**2))))
                PSNR.append(sum(psnr)/len(psnr))
                
                mse = sum(sum(sum((result-img)**2)))
                mse /= c*h*w
                mse *= 255*255
                rmse = np.sqrt(mse)
                RMSE.append(rmse)

                ssim = []
                k1 = 0.01
                k2 = 0.03
                for k in range(c):
                    ssim.append((2*np.mean(result[k])*np.mean(img[k])+k1**2) \
                        *(2*np.cov(result[k].reshape(h*w), img[k].reshape(h*w))[0,1]+k2**2) \
                        /(np.mean(result[k])**2+np.mean(img[k])**2+k1**2) \
                        /(np.var(result[k])+np.var(img[k])+k2**2))
                SSIM.append(sum(ssim)/len(ssim))

                temp = (np.sum(result*img, 0) + np.spacing(1)) \
                    /(np.sqrt(np.sum(result**2, 0) + np.spacing(1))) \
                    /(np.sqrt(np.sum(img**2, 0) + np.spacing(1)))
                sam = np.mean(np.arccos(temp))*180/np.pi
                SAM.append(sam)

                ergas = 0.
                for k in range(c):
                    ergas += np.mean((img[k]-result[k])**2)/np.mean(img[k])**2
                ergas = 100*np.sqrt(ergas/c)
                ERGAS.append(ergas)

        print(sum(PSNR)/len(PSNR), sum(RMSE)/len(RMSE), sum(SSIM)/len(SSIM), sum(SAM)/len(SAM), sum(ERGAS)/len(ERGAS))
        print("avg_psnr: "+str(avg_psnr)+" avg_SSIM: "+str(sum(SSIM)/len(SSIM))+" avg_sam: "+str(avg_sam))
        return avg_psnr, avg_loss,avg_sam


    def validate(self, valid_loader, name,patch_size=64):
        self.net.eval()
        validate_loss = 0
        total_psnr = 0
        total_sam = 0
        RMSE = []
        SSIM = []
        SAM = []
        ERGAS = []
        PSNR = []
        macs = 0
        params = 0
        book = 1
        print('[i] Eval dataset {}...'.format(name))
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valid_loader):
                 
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs, loss_data, _ = self.__step(False, inputs, targets)
                outputs = inputs
                psnr = np.mean(cal_bwpsnr(outputs, targets))
                sam = cal_sam(outputs, targets)
                validate_loss += loss_data
                total_sam += sam
                avg_loss = validate_loss / (batch_idx+1)
                avg_sam = total_sam / (batch_idx+1)

                total_psnr += psnr
                avg_psnr = total_psnr / (batch_idx+1)

                progress_bar(batch_idx, len(valid_loader), 'Loss: %.4e | PSNR: %.4f | AVGPSNR: %.4f '
                          % (avg_loss, psnr, avg_psnr))
                
                psnr = []
                c,h,w=inputs.shape[-3:]
                
                result = outputs.squeeze().cpu().detach().numpy()
            
                img = targets.squeeze().cpu().numpy()
                for k in range(c):
                    psnr.append(10*np.log10((h*w)/sum(sum((result[k]-img[k])**2))))
                PSNR.append(sum(psnr)/len(psnr))
                
                mse = sum(sum(sum((result-img)**2)))
                mse /= c*h*w
                mse *= 255*255
                rmse = np.sqrt(mse)
                RMSE.append(rmse)

                ssim = []
                k1 = 0.01
                k2 = 0.03
                for k in range(c):
                    ssim.append((2*np.mean(result[k])*np.mean(img[k])+k1**2) \
                        *(2*np.cov(result[k].reshape(h*w), img[k].reshape(h*w))[0,1]+k2**2) \
                        /(np.mean(result[k])**2+np.mean(img[k])**2+k1**2) \
                        /(np.var(result[k])+np.var(img[k])+k2**2))
                SSIM.append(sum(ssim)/len(ssim))

                temp = (np.sum(result*img, 0) + np.spacing(1)) \
                    /(np.sqrt(np.sum(result**2, 0) + np.spacing(1))) \
                    /(np.sqrt(np.sum(img**2, 0) + np.spacing(1)))
                #print(np.arccos(temp)*180/np.pi)
                sam = np.mean(np.arccos(temp))*180/np.pi
                SAM.append(sam)

                ergas = 0.
                for k in range(c):
                    ergas += np.mean((img[k]-result[k])**2)/np.mean(img[k])**2
                ergas = 100*np.sqrt(ergas/c)
                ERGAS.append(ergas)
                
        print("validate:")

        print("PSNR: %.2f dB | SSIM: %.4f | SAM: %.4f" % (avg_psnr,sum(SSIM)/len(SSIM),avg_sam) )
        
        if avg_psnr > self.best_psnr:
            self.best_epoch = self.epoch
            self.best_psnr = avg_psnr
            model_best_path = os.path.join(self.basedir, self.prefix, 'model_best.pth')
            self.save_checkpoint(
                model_out_path=model_best_path
            )
        self.logger.info("validate:")

        self.logger.info("PSNR: %.2f | SSIM: %.4f | SAM: %.4f | BestPSNR: %.2f | bestEpoch: %s" % (avg_psnr,sum(SSIM)/len(SSIM),avg_sam,self.best_psnr,self.best_epoch) )

        if not self.opt.no_log:      
            self.writer.add_scalar(
                join(self.prefix, name, 'val_loss_epoch'), avg_loss, self.epoch)
            self.writer.add_scalar(
                join(self.prefix, name, 'val_psnr_epoch'), avg_psnr, self.epoch)
            self.writer.add_scalar(
                join(self.prefix, name, 'val_sam_epoch'), avg_sam, self.epoch)
        print(avg_psnr, avg_loss,avg_sam)
        return avg_psnr, avg_loss,avg_sam


  
    def save_checkpoint(self, model_out_path=None, **kwargs):
        if not model_out_path:
            model_out_path = join(self.basedir, self.prefix, "model_epoch_%d_%d.pth" % (
                self.epoch, self.iteration))

        state = {
            'net': self.get_net().state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'iteration': self.iteration,
        }
        
        state.update(kwargs)

        if not os.path.isdir(join(self.basedir, self.prefix)):
            os.makedirs(join(self.basedir, self.prefix))

        torch.save(state, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))
        
    def get_net(self):
        if len(self.opt.gpu_ids) > 1:
            return self.net.module
        else:
            return self.net           
