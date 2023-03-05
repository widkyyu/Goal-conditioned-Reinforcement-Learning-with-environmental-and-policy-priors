
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class Net_Encoder(nn.Module):
    
    def __init__(self):
        super(Net_Encoder, self).__init__()
        self.cnn_base = nn.Sequential(  # input shape (1,1, 68, 68)
            nn.Conv2d(1, 16, kernel_size=4, stride=2), #(2, 33, 33)
            nn.BatchNorm2d(16),
            nn.ReLU(),  # activation
            nn.Conv2d(16, 16, kernel_size=5, stride=4),  # (16, 8, 8)
            nn.BatchNorm2d(16),
            nn.ReLU(),  # activation
            nn.Conv2d(16, 64, kernel_size=2, stride=2),  # (64, 4, 4)
            nn.BatchNorm2d(64),
            nn.ReLU(),  # activation
        )  # output shape
        
        self.apply(self._weights_init)
        self.upsamplingnearest = torch.nn.UpsamplingNearest2d(scale_factor=4)
        self.device = None

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, state_matrix):   #positon是4*4维，goal是2维
        B, T, D_content = state_matrix.shape
        state_matrix = state_matrix.unsqueeze(-2)
        state_matrix = state_matrix.reshape(-1,1,289)
        state_matrix = state_matrix.contiguous().view(-1,1,17,17)
        state_matrix_upsample = self.upsamplingnearest(state_matrix) #上下各扩充1倍，每个值复制4份
        state_matrix_upsample = state_matrix_upsample.type(torch.FloatTensor).to(self.device)
        state_matrix = self.cnn_base(state_matrix_upsample)
        state_matrix = state_matrix.view(-1,T,64,4,4)
        state_matrix_upsample = state_matrix_upsample.view(-1,T,1,68,68)
        
        return state_matrix, state_matrix_upsample

class LSTM(nn.Module):  #predicion 只有用到LSTM
    def __init__(self):
        nn.Module.__init__(self)
        super(LSTM, self).__init__()

        self.extrap_t = 4
        self.nt = 8

        self.recurrent_model = nn.LSTM(1024, 1024,num_layers=3, batch_first=True)

        # ## Setup h_0, c_0 as a trainable param
        num_layers = 3
        input_size = 1024
        hidden_cell = 1024
        
        self.h_0 = Variable(torch.zeros(num_layers,input_size)) #.cuda()
        self.c_0 = Variable(torch.zeros(num_layers,hidden_cell)) #.cuda()

        self.device = None

    def forward(self, z_content):

        B, T, D_content, W, H = z_content.shape  #(2,4,64,4,4)
        z_content = z_content.reshape(B,T,D_content*W*H) #(2,4,1024)

        #Processing observations.
        h_0 = self.h_0.unsqueeze(1)
        h_0 = h_0.repeat(1,B,1).to(self.device)

        c_0 = self.c_0.unsqueeze(1)
        c_0 = c_0.repeat(1,B,1).to(self.device)

        z_past, (h_n, c_n) = self.recurrent_model(z_content, (h_0, c_0)) #(B,T,Dout), (2,B,H)
        z_n = z_past[:,-1].unsqueeze(1) #(B,Dout)

        prediction = []

        for i in range(self.nt-4-1):
            z_n, (h_n, c_n) = self.recurrent_model(z_n, (h_n, c_n)) 
            prediction.append(z_n) #(B,1,Dout)

        prediction = torch.cat(prediction, 1)

        z = torch.cat([z_past, prediction], 1) #(B,T,Dout)
        output_z_content = z.reshape(B,self.nt-1,D_content, W, H)

        return output_z_content, z_content #z_content为LSTM之前的数据

class Net_Decoder(nn.Module):
    
    def __init__(self):
        super(Net_Decoder, self).__init__()
        self.cnn_base = nn.Sequential(                                                  # input shape (1,1024,3,3)  o = (i-1)×s + k - 2p
            nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2),  # (1024,3,3)
            nn.ReLU(),  # activation
            nn.ConvTranspose2d(16, 16, kernel_size=5, stride=4),   # (128, 9, 9)
            nn.ReLU(),  # activation
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2),    # (64, 21, 21) 
        )  # output shape (4, 96, 96)
        
        self.apply(self._weights_init)
        self.pool = torch.nn.AvgPool2d(4, stride=4)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, state_matrix):   #positon是4*4维，goal是2维
        B, T, D_content, W, H = state_matrix.shape
        state_matrix = state_matrix.contiguous().view(-1,D_content, W, H)
        state_matrix_pool = self.cnn_base(state_matrix)
        state_matrix = self.pool(state_matrix_pool) #均值池化
        state_matrix = state_matrix.view(-1,1,289)
        state_matrix = state_matrix.squeeze(-2)
        state_matrix = state_matrix.reshape(-1,T,289)
        state_matrix_pool = state_matrix_pool.view(-1,T,1,68,68)
        
        return state_matrix, state_matrix_pool

class TINet(nn.Module):
    
    def __init__(self):
        super(TINet, self).__init__()
        self.cnn_base = nn.Sequential(  # input shape (batchsize,4, 68, 68)
            nn.Conv2d(4, 64, kernel_size=4, stride=2), #(64, 33, 33)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 64, kernel_size=3, stride=2),  # (128, 16, 16)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 128, kernel_size=4, stride=2),  # (128, 7, 7)
            nn.ReLU(),  # activation
            nn.Conv2d(128, 256, kernel_size=3, stride=2),  # (256, 3, 3)
            nn.ReLU(),  # activation
            nn.Flatten()
        )  # output shape
        self.positionnet = nn.Linear(10, 128)
        self.fc0 = nn.Linear(2304, 256)
        self.fc1 = nn.Linear(256+128, 512)
        self.upsamplingnearest = torch.nn.UpsamplingNearest2d(scale_factor=4)

        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, state_matrix,  position, goal):   #positon是4*4维，goal是2维
        B, T, D_content = state_matrix.shape
        position     = position.contiguous().view(-1,8)
        positon_goal = torch.cat([position, goal], 1)
        
        position_goal = self.positionnet(positon_goal)
        position_goal = torch.relu(position_goal)
        state_matrix = state_matrix.contiguous().view(-1,T,17,17)
        state_matrix = self.upsamplingnearest(state_matrix) #上下各扩充1倍，每个值复制4份
        state_matrix = self.cnn_base(state_matrix)
        state_matrix =  self.fc0(state_matrix)
        state_matrix = torch.relu(state_matrix)
        x = torch.cat([state_matrix, position_goal], 1)
        x =  self.fc1(x)
        x = torch.relu(x)               
                     
        return x

class Net_PPO_actor(nn.Module):
    
    def __init__(self):
        super(Net_PPO_actor, self).__init__()
        #actor
        self.bone1 = TINet()
        self.A = nn.Linear(512, 5)
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, state_matrix,  position, goal, model='actor'):   #positon是4*4维，goal是2维

        x_actor = self.bone1(state_matrix,  position, goal)               
        A = self.A(x_actor)
        a_prob = torch.softmax(A, dim=1) 
        return a_prob  
        
        
        
class Net_PPO_critic(nn.Module):
    
    def __init__(self):
        super(Net_PPO_critic, self).__init__()
        
        #critic
        self.bone2 = TINet()
        self.V = nn.Linear(512, 1)
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, state_matrix,  position, goal, model='actor'):   #positon是4*4维，goal是2维
  
        x_critic = self.bone2(state_matrix,  position, goal)
        V = self.V(x_critic)
        return V

class Net_PPO_Predictor_actor(nn.Module):
    
    def __init__(self):
        super(Net_PPO_Predictor_actor, self).__init__()
        #actor
        self.bone1 = TINet()
        self.bone1.cnn_base[0] = nn.Conv2d(8, 64, kernel_size=4, stride=2)
        self.A = nn.Linear(512, 5)
        self.apply(self._weights_init)
        
    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, state_matrix, position, goal):   #positon是4*4维，goal是4维
        
        x_actor = self.bone1(state_matrix,  position, goal)            
        A = self.A(x_actor)
        a_prob = torch.softmax(A, dim=1)  
        return a_prob
        
class Net_PPO_Predictor_critic(nn.Module):
    
    def __init__(self):
        super(Net_PPO_Predictor_critic, self).__init__()
        #critic
        self.bone2 = TINet()
        self.bone2.cnn_base[0] = nn.Conv2d(8, 64, kernel_size=4, stride=2)
        self.V = nn.Linear(512, 1)
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, state_matrix, position, goal):   #positon是4*4维，goal是4维
        
        x_critic = self.bone2(state_matrix,  position, goal)             
        V = self.V(x_critic)
        return V

class Net_SoA_actor(nn.Module):
    
    def __init__(self):
        super(Net_SoA_actor, self).__init__()

        self.bone1 = TINet()
        self.bone1.cnn_base[0] = nn.Conv2d(8, 64, kernel_size=4, stride=2)
        self.bone1.positionnet = nn.Linear(8+4, 128)
        self.A = nn.Linear(512, 5)
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, state_matrix,  position, goal):   #positon是4*4维，goal是4维
        
        x_actor = self.bone1(state_matrix,  position, goal)            
        A = self.A(x_actor)
        a_prob = torch.softmax(A, dim=1)  
        return a_prob
        

class Net_SoA_critic(nn.Module):
    
    def __init__(self):
        super(Net_SoA_critic, self).__init__()

        self.bone2 = TINet()
        self.bone2.cnn_base[0] = nn.Conv2d(8, 64, kernel_size=4, stride=2)
        self.bone2.positionnet = nn.Linear(8+4, 128)
        self.V = nn.Linear(512, 1)
        
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, state_matrix,  position, goal):   #positon是4*4维，goal是4维
        
        x_critic = self.bone2(state_matrix,  position, goal)             
        V = self.V(x_critic)
        return V

class Net_SoA_orient(nn.Module):
    
    def __init__(self):
        super(Net_SoA_orient, self).__init__()

        self.bone3 = TINet()
        self.bone3.cnn_base[0] = nn.Conv2d(8, 64, kernel_size=4, stride=2)
        self.bone3.positionnet = nn.Linear(10, 128)
        self.Px = nn.Linear(512, 7)
        self.Py = nn.Linear(512, 7)
        
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, state_matrix,  position, goal):   #positon是4*4维，goal是4维
        
        x = self.bone3(state_matrix,  position, goal)
                       
        Px = self.Px(x)
        Py = self.Py(x)
        Px_prob = torch.softmax(Px, dim=1)
        Py_prob = torch.softmax(Py, dim=1)             
        
        return Px_prob, Py_prob