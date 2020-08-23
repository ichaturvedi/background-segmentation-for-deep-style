
import torch
from .base_model import BaseModel
from . import networks
import math
import random

class SDModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

            parser.add_argument('--DS_input', type=str, default='concat', help='Whether to stack mask with gan output or to concatenate [concat | stack]') 
            parser.add_argument('--netDS', type=str, default='basic', help='specify segmentation discriminator architecture [basic | n_layers | pixel | full_conv]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
            parser.add_argument('--n_layers_DS', type=int, default=7, help='only used if netDS==n_layers || full_conv')

        return parser
    
    #python train.py --dataroot ./datasets/triple_flipped --name sd_f_bgfg_fc_DS8 --model sd --direction AtoB --dataset_mode triple --netDS full_conv --n_layers_DS 8
    #python test.py --dataroot ./datasets/painted_pairs --name point_lsgan --model pix2pix --direction AtoB --num_test 300 
    # python test.py --dataroot ./datasets/triple --name sd_flipped_hayao --model sd --direction AtoB --num_test 300 --dataset_mode triple
    #python train.py --dataroot ./datasets/triple_flipped --name sd_flipped_bgfg --model sd --direction AtoB --dataset_mode triple --continue_train --epoch_count 101 --niter 150 --niter_decay 50

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'DS_real', 'DS_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D', 'DS']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        if hasattr(opt, 'num_skips'):
            n_skips = opt.num_skips
        else:
            n_skips = 7
        if hasattr(opt, 'DS_input'):
            self.DS_input = opt.DS_input
        else:
            self.DS_input = 'concat'
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netDS = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netDS,
                                          opt.n_layers_DS, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)                              

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_DS = torch.optim.Adam(self.netDS.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_DS)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.mask = input['M'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)
        # print("FORWARRD")
        # print(self.fake_B.shape)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        # print(self.real_A.shape)
        # print(self.fake_B.shape)
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_DS(self):

        switch = 0;
        mask = self.mask.cpu().numpy()
        realA = self.real_A.cpu().numpy()
 
        for itr in range(1000):  
              x = random.randint(1,256)-1
              y = random.randint(1,256)-1
              sum = mask[0][0][x][y]+mask[0][1][x][y]+mask[0][2][x][y]
              if sum != -3:
                error = abs(realA[0][0][x][y]-self.fake_B[0][0][x][y])
                error += abs(realA[0][1][x][y]-self.fake_B[0][1][x][y])
                error += abs(realA[0][2][x][y]-self.fake_B[0][2][x][y])
#                if error > 0 and error < 1:
 #                 switch += 1-math.exp(-0.01*0.2)
 #               elif error < 2:
  #                switch += 1-math.exp(-0.01*0.8)
   #             else:
                switch += 1-math.exp(error) 

   
        """Calculate GAN loss for segmentation discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_MB = torch.cat((self.mask, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netDS(fake_MB.detach())
        self.loss_DS_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_MB = torch.cat((self.mask, self.real_B), 1)
        pred_real = self.netDS(real_MB)
        self.loss_DS_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_DS = (self.loss_DS_fake + self.loss_DS_real) * 0.5 + switch
        self.loss_DS.backward() 

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # Third, G(A) should fake the segmentation discriminator
        pred_fake_DS = self.netDS(fake_AB)
        self.loss_G_DS = self.criterionGAN(pred_fake_DS, True)
        # combine loss and calculate gradients
        self.loss_G = (self.loss_G_GAN + self.loss_G_DS) * 0.5 + self.loss_G_L1 
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update DS
        self.set_requires_grad(self.netDS, True)  # enable backprop for DS
        self.optimizer_DS.zero_grad()     # set DS's gradients to zero
        self.backward_DS()                # calculate gradients for DS
        self.optimizer_DS.step()          # update DS's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
