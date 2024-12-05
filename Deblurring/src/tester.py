import os

import cv2

from Deblurring.schedule.schedule import Schedule
from Deblurring.model.NAFDPM import NAFDPM, EMA
from Deblurring.schedule.diffusionSample import GaussianDiffusion
from Deblurring.schedule.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import copy
from Deblurring.src.sobel import Laplacian
import logging
from collections import OrderedDict
import pyiqa
import utils.util as util
from utils.util import crop_concat, crop_concat_back, min_max
# import wandb


def init__result_Dir(path):
    work_dir = os.path.join(path, 'Training')
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    max_model = 0
    for root, j, file in os.walk(work_dir):
        for dirs in j:
            try:
                temp = int(dirs)
                if temp > max_model:
                    max_model = temp
            except:
                continue
        break
    max_model += 1
    path = os.path.join(work_dir, str(max_model))
    os.mkdir(path)
    return path


class Tester:
    def __init__(self, config):
        torch.manual_seed(0)
        self.mode = config.MODE
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #DEFINE NETWORK
        in_channels = config.CHANNEL_X + config.CHANNEL_Y
        out_channels = config.CHANNEL_Y
        self.out_channels = out_channels
        self.network = NAFDPM(input_channels=in_channels,
            output_channels = config.CHANNEL_Y,
            n_channels      = config.MODEL_CHANNELS,
            middle_blk_num  = config.MIDDLE_BLOCKS, 
            enc_blk_nums    = config.ENC_BLOCKS, 
            dec_blk_nums    = config.DEC_BLOCKS,
            mode=0).to(self.device)
        
        # self.psnr = pyiqa.create_metric('psnr', device=self.device)
        # self.ssim = pyiqa.create_metric('ssim', device=self.device)
        # self.lpips = pyiqa.create_metric('lpips', device=self.device)
        # self.dists = pyiqa.create_metric('dists', device=self.device)
        # self.musiq = pyiqa.create_metric('musiq', device=self.device)
        self.bestPSNR = 0
        self.bestLPIPS = 10
        self.bestDISTS = 10
        
        #DIFFUSION SAMPLING USING GAUSSIAN DIFFUSION
        self.schedule = Schedule(config.SCHEDULE, config.TIMESTEPS)
        self.diffusion = GaussianDiffusion(self.network.denoiser, config.TIMESTEPS, self.schedule).to(self.device)

        #LOGGER AND PATHS
        util.setup_logger(
               "base",
                config.LOGGER_PATH,
                "train" + "DocDiff",
                level=logging.INFO,
                screen=True,
                tofile=True,
            )
        self.logger = logging.getLogger("base")
        self.test_img_save_path = config.TEST_IMG_SAVE_PATH
        self.logger_path = config.LOGGER_PATH
        if not os.path.exists(self.test_img_save_path):
            os.makedirs(self.test_img_save_path)
        if not os.path.exists(self.logger_path):
            os.makedirs(self.logger_path)
        self.training_path = config.TRAINING_PATH
        self.pretrained_path_init_predictor = config.PRETRAINED_PATH_INITIAL_PREDICTOR
        self.pretrained_path_denoiser = config.PRETRAINED_PATH_DENOISER
        self.continue_training = config.CONTINUE_TRAINING
        self.continue_training_steps = 0
        self.path_train_gt = config.PATH_GT
        self.path_train_img = config.PATH_IMG
        self.weight_save_path = config.WEIGHT_SAVE_PATH
        self.test_path_img = config.TEST_PATH_IMG
        self.test_path_gt = config.TEST_PATH_GT
        # self.save_img_path = util.init__result_Dir(self.training_path)
        # self.save_img_path = test_img_save_path

        #LR ITERATIONS AND TRAINING STUFFS
        self.iteration_max = config.ITERATION_MAX
        self.LR = config.LR
        self.cross_entropy = nn.BCELoss()
        self.num_timesteps = config.TIMESTEPS
        self.ema_every = config.EMA_EVERY
        self.start_ema = config.START_EMA
        self.save_model_every = config.SAVE_MODEL_EVERY
        self.EMA_or_not = config.EMA
        self.TEST_INITIAL_PREDICTOR_WEIGHT_PATH = config.TEST_INITIAL_PREDICTOR_WEIGHT_PATH
        self.TEST_DENOISER_WEIGHT_PATH = config.TEST_DENOISER_WEIGHT_PATH
        self.DPM_SOLVER = config.DPM_SOLVER
        self.DPM_STEP = config.DPM_STEP
        self.beta_loss = config.BETA_LOSS
        self.pre_ori = config.PRE_ORI
        self.high_low_freq = config.HIGH_LOW_FREQ
        self.image_size = config.IMAGE_SIZE
        self.native_resolution = config.NATIVE_RESOLUTION
        self.validate_every = config.VALIDATE_EVERY

 
        #DATASETS AND DATALOADERS
        from Deblurring.data.docdata import DocData
        if self.mode == 1:
            dataset_train = DocData(self.path_train_img, self.path_train_gt, config.IMAGE_SIZE, self.mode)
            self.batch_size = config.BATCH_SIZE
            self.dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, drop_last=False,
                                               num_workers=config.NUM_WORKERS)
            dataset_test = DocData(config.TEST_PATH_IMG, config.TEST_PATH_GT, config.IMAGE_SIZE, 0)
            self.dataloader_test = DataLoader(dataset_test, batch_size=config.BATCH_SIZE_VAL, shuffle=False,
                                              drop_last=False,
                                              num_workers=config.NUM_WORKERS)
        else:
            dataset_test = DocData(config.TEST_PATH_IMG, config.TEST_PATH_GT, config.IMAGE_SIZE, self.mode)
            self.dataloader_test = DataLoader(dataset_test, batch_size=config.BATCH_SIZE_VAL, shuffle=False,
                                              drop_last=False,
                                              num_workers=config.NUM_WORKERS)
        if self.mode == 1 and self.continue_training == 'True':
            print('Continue Training')
            self.network.init_predictor.load_state_dict(torch.load(self.pretrained_path_init_predictor))
            self.network.denoiser.load_state_dict(torch.load(self.pretrained_path_denoiser))
            self.continue_training_steps = config.CONTINUE_TRAINING_STEPS
            
        if self.mode == 1 and config.EMA == 'True':
            self.EMA = EMA(0.9999)
            self.ema_model = copy.deepcopy(self.network).to(self.device)
        if config.LOSS == 'L1':
            self.loss = nn.L1Loss()
        elif config.LOSS == 'L2':
            self.loss = nn.MSELoss()
        else:
            print('Loss not implemented, setting the loss to L2 (default one)')
            self.loss = nn.MSELoss()
        if self.high_low_freq == 'True':
            self.high_filter = Laplacian().to(self.device)
        
        #WANDB LOGIN AND SET UP
        self.wandb = config.WANDB
        self.wandb = False


    def test(self):

        with torch.no_grad():
            #LOAD CHECKPOINTS FOR INITIAL PREDICTOR AND DENOISER
            checkpoint_init = torch.load(self.TEST_INITIAL_PREDICTOR_WEIGHT_PATH,map_location=self.device)
            checkpoint_denoiser = torch.load(self.TEST_DENOISER_WEIGHT_PATH, map_location=self.device)
            self.network.init_predictor.load_state_dict(checkpoint_init['model_state_dict'])
            self.network.denoiser.load_state_dict(checkpoint_denoiser['model_state_dict'])
            #self.network.init_predictor.load_state_dict(torch.load(self.TEST_INITIAL_PREDICTOR_WEIGHT_PATH))
            #self.network.denoiser.load_state_dict(torch.load(self.TEST_DENOISER_WEIGHT_PATH))
            self.diffusion = GaussianDiffusion(self.network.denoiser, self.num_timesteps, self.schedule).to(self.device)
            print('Test Model loaded')
            
            #PUT EVERYTHING IN EVALUATION MODE
            self.network.eval()
            
            tq = tqdm(self.dataloader_test)
            iteration = 0
            #FOR IMAGES IN TESTING DATASET
            for img, gt, name in tq:
                tq.set_description(f'Iteration {iteration} / {len(self.dataloader_test.dataset)}')
                iteration += 1
                # e_img = enhance_text_clarity(img)
                # e_img = torch.from_numpy(e_img)
                #IF NATIVE DIVIDE IMAGES IN SUBIMAGES
                # if self.native_resolution == 'True':
                #     temp = img
                #     img = crop_concat(img)
                    
                #INIT RANDOM NOISE
                noisyImage = torch.randn_like(img).to(self.device)
                
                #FIRST INITIAL PREDICTION
                init_predict = self.network.init_predictor(img.to(self.device))

                #REFINE RESIDUAL IMAGE USING DPM SOLVER OR DDIM
                if self.DPM_SOLVER == 'True':
                    #DPM SOLVER BRANCH
                    sampledImgs = dpm_solver(self.schedule.get_betas(), self.network.denoiser,
                                             noisyImage, self.DPM_STEP, init_predict, model_kwargs={})
                else:
                    #DDIM BRANCH
                    sampledImgs = self.diffusion(noisyImage, init_predict, self.pre_ori)

                #COMPUTE FINAL IMAGES   
                finalImgs = (sampledImgs + init_predict)
                # finalImgs = init_predict


                finalImgs = torch.clamp(finalImgs,0,1)
                img_save = torch.cat([img, gt, init_predict.cpu(), finalImgs], dim=3)
                if not os.path.exists(self.test_img_save_path):
                        os.makedirs(self.test_img_save_path)
                save_image(img_save, os.path.join(
                        self.test_img_save_path, f"{iteration}.png"), nrow=3)


def dpm_solver(betas, model, x_T, steps, condition, model_kwargs):
    # You need to firstly define your model and the extra inputs of your model,
    # And initialize an `x_T` from the standard normal distribution.
    # `model` has the format: model(x_t, t_input, **model_kwargs).
    # If your model has no extra inputs, just let model_kwargs = {}.

    # If you use discrete-time DPMs, you need to further define the
    # beta arrays for the noise schedule.

    # model = ....
    # model_kwargs = {...}
    # x_T = ...
    # betas = ....

    # 1. Define the noise schedule.
    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)

    # 2. Convert your discrete-time `model` to the continuous-time
    # noise prediction model. Here is an example for a diffusion model
    # `model` with the noise prediction type ("noise") .
    model_fn = model_wrapper(
        model,
        noise_schedule,
        model_type="x_start",  # or "x_start" or "v" or "score"
        model_kwargs=model_kwargs,
        guidance_type="classifier-free",
        condition=condition
    )

    # 3. Define dpm-solver and sample by singlestep DPM-Solver.
    # (We recommend singlestep DPM-Solver for unconditional sampling)
    # You can adjust the `steps` to balance the computation
    # costs and the sample quality.
    dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++",
                            correcting_x0_fn="dynamic_thresholding")
    # Can also try
    # dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")

    # You can use steps = 10, 12, 15, 20, 25, 50, 100.
    # Empirically, we find that steps in [10, 20] can generate quite good samples.
    # And steps = 20 can almost converge.
    x_sample = dpm_solver.sample(
        x_T,
        steps=steps,
        order=1,
        skip_type="time_uniform",
        method="singlestep",
    )
    return x_sample
