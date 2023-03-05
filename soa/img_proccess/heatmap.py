import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time
import datetime
from datetime import datetime
from pathlib import Path



def readname(filepath):
        filePath = filepath
        name = os.listdir(filePath)
        return name

def heatmap_test(buffer,filename,i_ep,device):
    values = buffer['p'][:,4]

    values_ = buffer['p'][:,3]#用于计算预测
    values_ROG = buffer['f'][:,0]
    
    count = np.unique(values[:,1],return_counts=True)
    values_matrix = np.zeros((17,17))
    values_matrix_ROG = np.zeros((20,20))
    for i in range(len(values)):
        values_matrix[values[i][0].astype(int), values[i][1].astype(int)] += 1
        values_matrix_ROG[values_[i][0].astype(int)+values_ROG[i][1].astype(int), values_[i][1].astype(int)+values_ROG[i][0].astype(int)] += 1

    ax = sns.heatmap(values_matrix, cmap="summer",mask=values_matrix < 1)
    ax.set_title('Heatmap')  # 图标题
    ax.set_xlabel('x label')  # x轴标题
    ax.set_ylabel('y label')

    figure = ax.get_figure()
    savefiles_path = Path('/home/yuzhe/gym-minigrid_1216/rrl_cnn/low_level/img_proccess/test/'+filename+str(device))
    if not savefiles_path.exists():
        os.makedirs(savefiles_path)
    figure.savefig(str(savefiles_path)+'/'+'sns_heatmap.jpg') 
    #rollout track
    # plt.close()

    ax = sns.heatmap(values_matrix_ROG, cmap="GnBu",mask=values_matrix_ROG < 1)
    # ax.set_title('Heatmap')  # 图标题
    # ax.set_xlabel('x label')  # x轴标题
    # ax.set_ylabel('y label')

    figure = ax.get_figure()
    savefiles_path = Path('/home/yuzhe/gym-minigrid_1216/rrl_cnn/low_level/img_proccess/test/'+filename+str(device))
    if not savefiles_path.exists():
        os.makedirs(savefiles_path)
    figure.savefig(str(savefiles_path)+'/'+'sns_heatmap_RoG.jpg')
    

    plt.close()


def heatmap(buffer,filename,i_ep,device):
    values = buffer['p'][:,4]
    count = np.unique(values[:,1],return_counts=True)
    values_matrix = np.zeros((17,17))
    for i in range(len(values)):
        values_matrix[values[i][0].astype(int), values[i][1].astype(int)] += 1

    ax = sns.heatmap(values_matrix, cmap="YlGnBu",mask=values_matrix < 1)
    ax.set_title(str(i_ep)+'ep '+'Heatmap for '+filename)  # 图标题
    ax.set_xlabel('x label')  # x轴标题
    ax.set_ylabel('y label')

    figure = ax.get_figure()
    savefiles_path = Path('/home/yuzhe/gym-minigrid_1216/rrl_cnn/low_level/img_proccess/example/'+filename+str(device))
    if not savefiles_path.exists():
        os.makedirs(savefiles_path)
    figure.savefig(str(savefiles_path)+'/'+'sns_heatmap.jpg') 

    track_savefiles_path = Path('/datadisk/yuzhe/predictor/track_buffer/'+filename)
    if not track_savefiles_path.exists():
        os.makedirs(track_savefiles_path)
    np.save(str(track_savefiles_path)+'/' + str(i_ep)+'i_ep_' + 'track_buffer_' +datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+".npy", values)

    plt.close()

def single():
    buffer_file = '/media/yuzhe/yuzhe/predictor/track_buffer/ppo_2net_baseline1_1_MiniGrid-twoarmy-17x17-v4_6667seed2023_02_08_18_58_49/track_buffer_1610001counter_2023_02_09_01_34_33.npy'
    pre_buffer = np.load(buffer_file)
    # values = np.random.randint(0, high = 17, size=(10000,2) )
    # values = pre_buffer['p'][:,4]
    values = pre_buffer
    count = np.unique(values[:,1],return_counts=True)
    values_matrix = np.zeros((17,17))
    for i in range(len(values)):
        values_matrix[values[i][0].astype(int), values[i][1].astype(int)] += 1

    ax = sns.heatmap(values_matrix, cmap="YlGnBu",mask=values_matrix < 1)
    # ax.set_title('Heatmap for test')  # 图标题
    # ax.set_xlabel('x label')  # x轴标题
    # ax.set_ylabel('y label')

    figure = ax.get_figure()
    
    figure.savefig('/home/yuzhe/gym-minigrid_1216/rrl_cnn/low_level/img_proccess/example/'+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+'sns_heatmap.jpg') 
    return True

def files():
    #读取文件
    agent_name = 'ppo_pure_MiniGrid-twoarmy-17x17-v6_2535seed_2023_02_10_17_31_12'
    datafiles_path = Path('/datadisk/yuzhe/predictor/track_buffer/'+agent_name)
    names = readname(str(datafiles_path))
    savefiles_path = Path('/datadisk/yuzhe/predictor/track_buffer/example/env_v6/' + agent_name)
    if not savefiles_path.exists():
        os.makedirs(savefiles_path)
    
    for i in names:
        buffer_file = str(datafiles_path) + '/'
        pre_buffer = np.load(buffer_file + i)
        # values = np.random.randint(0, high = 17, size=(10000,2) )
        # values = pre_buffer['p'][:,4]
        values = pre_buffer
        count = np.unique(values[:,1],return_counts=True)
        values_matrix = np.zeros((17,17))
        for j in range(len(values)):
            values_matrix[values[j,0].astype(int), values[j,1].astype(int)] += 1
        
        ax = sns.heatmap(values_matrix, cmap="YlGnBu",mask=values_matrix < 1)
        ax.set_title('Heatmap for '+agent_name+'_'+i[13:16])  # 图标题
        # ax.set_xlabel('x label')  # x轴标题
        # ax.set_ylabel('y label')
        figure = ax.get_figure()

        figure.savefig(str(savefiles_path)+'/'+i[13:21]+'.jpg') #i[0:7]
        plt.close()
        print('save'+str(i))  
        
if __name__ == "__main__":
    a = files( )
    # print('over')       
