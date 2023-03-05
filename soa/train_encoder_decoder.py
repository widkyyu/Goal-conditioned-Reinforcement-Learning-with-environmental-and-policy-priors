#直接用卷积网络做预测信息测试，中间没有加LSTM

from PIL import Image
from turtle import right
import sys, os
import numpy as np
import time
import datetime
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

from pathlib import Path
from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter
import torchvision
from sklearn.model_selection import train_test_split
from torchvision.utils import save_image, make_grid
from torch.autograd import Variable
import argparse
from torch.optim.lr_scheduler import StepLR
from agent.encoder_LSTM_decoder import encoder_lstm_decoder

# l1_loss = nn.SmoothL1Loss(reduction='none')
l1_loss = nn.MSELoss(reduction='none')
def evaluate_loss(z_theme, model, logs=None):

    z_theme_pred = model(z_theme)

    z_t_recon_loss = l1_loss(z_theme_pred[:,3:], z_theme[:,4:]).mean(2)

    loss =  z_t_recon_loss
    loss = loss.mean()

    if logs is None or len(logs.keys()) == 0:
        logs = {
        "Loss/z_t_recon_loss": z_t_recon_loss.mean(),
        }
    else:
        logs["Loss/z_t_recon_loss"] += z_t_recon_loss.mean()

    return loss, logs

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,help="random seed to generate the environment with",default=3344)
    parser.add_argument("--tile_size", type=int, help="size at which to render tiles", default=17)
    parser.add_argument("--batch_size", type=int, help="size at which to sample", default=2)
    parser.add_argument("--num_workers", type=int, default=500)
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
    parser.add_argument('--epslion', type=float, default=0.9, metavar='G', help='epslion greedy (default: 0.5)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay (default: 0.0001)')
    parser.add_argument('--lr_gamma', type=float, default=0.8, metavar='LG', help='learning rate discount factor (default: 0.8)')
    parser.add_argument('--lr_step_size', type=int, default=200, help='each step_size perform learning rate discount factor (default: 200)')
    parser.add_argument('--num_episodes', type=int, default=10000, help='number all episodes (default: 100000)')
    parser.add_argument("--buffer_file", help="buffer to load", default='/media/yuzhe/yuzhe/predictor/data/bufferpredictor_MiniGrid-twoarmy-17x17-v2_random_20000_prebuffer_2023_02_06_18_37_36.npy2023_02_06_18_45_35.npy')
    parser.add_argument("--log_dir", help="model to load", default='/media/yuzhe/yuzhe/predictor/encoder_decoder/')
    parser.add_argument('--buffer_cutnumber', type=int, default=10, help='number all buffercut (default: 10)')
    parser.add_argument("--cuda", help="buffer to load", default="cuda:1")
    parser.add_argument("--server", help="if running in the server", default=True)
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    if args.server:
        device = torch.device(args.cuda if use_cuda else "cpu")
    else:
        device = torch.device("cuda:0" if use_cuda else "cpu")
    
    torch.set_num_threads(args.num_workers)

    seed = None if args.seed == -1 else args.seed  
    random.seed(seed)     
    np.random.seed(seed)  
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)            
    if use_cuda:
        torch.cuda.manual_seed(seed)       
        torch.cuda.manual_seed_all(seed)    

    #关键性参数
    writer = None
    writer_dir = '/media/yuzhe/yuzhe/tensor/encoder_decoder/logs_'
    filepath = Path(args.log_dir + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+'/')
    if writer_dir is not None:
        writer = SummaryWriter(log_dir=writer_dir)           

    buffer = np.load(args.buffer_file)
    predictor = encoder_lstm_decoder()
    predictor.encoder.to(device)
    predictor.encoder.device = device
    predictor.decoder.to(device)
    predictor.seed = seed
    predictor.num_episodes_en_de = args.num_episodes
    predictor.name = 'MiniGrid-twoarmy-17x17-v2_v4'+'_encoder_decoder_'
    predictor.batch_size = args.batch_size
    
    all_params = chain(predictor.encoder.parameters(), predictor.decoder.parameters())
    model_parameters = filter(lambda p: p.requires_grad, all_params)
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Number of params:{params}')
    
    #优化器
    predictor.optimizer_encoder = torch.optim.Adam(predictor.encoder.parameters(),lr=5e-04, betas=(0.9,0.98), eps=1e-09)
    predictor.optimizer_decoder = torch.optim.Adam(predictor.decoder.parameters(),lr=5e-04, betas=(0.9,0.98), eps=1e-09)
    predictor.scheduler_encoder = StepLR(predictor.optimizer_encoder, step_size=1, gamma=0.9)
    predictor.scheduler_decoder = StepLR(predictor.optimizer_decoder, step_size=1, gamma=0.9)
    predictor.update_encoder_decoder(buffer,device)
    print('update over')
           
            

    
        
