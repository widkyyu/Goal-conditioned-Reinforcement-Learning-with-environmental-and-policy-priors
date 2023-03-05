

import sys, os
import numpy as np
import time
import datetime
from datetime import datetime

from pathlib import Path
from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import random
import argparse
from torch.optim.lr_scheduler import StepLR
from agent.encoder_LSTM_decoder import encoder_lstm_decoder

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,help="random seed to generate the environment with",default=6667)
    parser.add_argument("--tile_size", type=int, help="size at which to render tiles", default=17)
    parser.add_argument("--batch_size", type=int, help="size at which to sample", default=1)
    parser.add_argument("--num_workers", type=int, default=500)
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
    parser.add_argument('--epslion', type=float, default=0.9, metavar='G', help='epslion greedy (default: 0.5)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay (default: 0.0001)')
    parser.add_argument('--lr_gamma', type=float, default=0.9, metavar='LG', help='learning rate discount factor (default: 0.8)')
    parser.add_argument('--lr_step_size', type=int, default=200, help='each step_size perform learning rate discount factor (default: 200)')
    parser.add_argument('--num_episodes', type=int, default=10000, help='number all episodes (default: 100000)')
    parser.add_argument('--max_steps', type=int, default=90, help='number of max steps in environment (default: 100)')
    parser.add_argument("--buffer_file", help="buffer to load", default='/media/yuzhe/yuzhe/predictor/data/bufferpredictor_MiniGrid-twoarmy-17x17-v2_random_20000_prebuffer_2023_02_06_18_37_36.npy2023_02_06_18_45_35.npy')
    parser.add_argument("--log_dir", help="model to load", default='/media/yuzhe/yuzhe/predictor/predictor/')
    parser.add_argument("--net_file", help="net to load", default='/media/yuzhe/yuzhe/predictor/param/ppo_encoder_decoder/gridim_encoder_decoder2023_02_07_02_55_22/MiniGrid-twoarmy-17x17-v2_encoder2_decoder2__net_8.911771693617258e-07average_score_731epoch_3344seed_2023_02_07_17_15_49.pkl')
    parser.add_argument("--cuda", help="buffer to load", default="cuda:2")
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
    predictor.encoder.load_state_dict(torch.load(args.net_file,map_location=args.cuda)['model_encoder'])
    predictor.decoder.load_state_dict(torch.load(args.net_file,map_location=args.cuda)['model_decoder'])

    predictor.predictor.to(device)
    predictor.predictor.device = device
    predictor.seed = seed
    predictor.num_episodes_pre = args.num_episodes
    predictor.name = 'MiniGrid-twoarmy-17x17-v2_predictor_'
    predictor.batch_size = args.batch_size
    
    all_params = chain(predictor.predictor.parameters())
    model_parameters = filter(lambda p: p.requires_grad, all_params)
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Number of params:{params}')

    #优化器
    predictor.optimizer_predictor = torch.optim.Adam(predictor.predictor.parameters(),lr=1e-04, betas=(0.9,0.98), eps=1e-09)
    #学习率衰减
    predictor.scheduler_predictor = StepLR(predictor.optimizer_predictor, step_size=1, gamma=0.9) 
    predictor.update_predictor(buffer,device)
    print('update over')

        
            

    
        
