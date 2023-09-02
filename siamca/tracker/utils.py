import os
import torch
from matplotlib import pyplot as plt 
import numpy as np
from PIL import Image
from scipy.interpolate import griddata

# 用插值方法扩展特征图。因为要将热力图（24,8）与原始输入的图片（384,128）叠加，
# 但特征图（24,8）比较小，生成的热力图（24,8）也一样小，所以先要将特征图扩展成原始图片一样大小，即将24,8-->384,128
def expend_data(x, y, z, smooth_degree=10):
    num_x = len(x)
    max_x = np.max(x)
    min_x = np.min(x)

    num_y = len(y)
    max_y = np.max(y)
    min_y = np.min(y)

    X, Y = np.meshgrid(x, y)
    coordinates = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
    # print(z.shape)

    m = np.linspace(min_x, max_x, (num_x -1) * 8+16 )
    n = np.linspace(min_y, max_y, (num_y -1) *8+16 )
    M, N = np.meshgrid(m, n)

    U = griddata(coordinates, z, (M, N), method='cubic')

    return M, N, U
def merge():
 for i in range(1,50):
    img1 = Image.open("/home/psq/Documents/AdaptBan/heat_crop/map"+str(i)+".jpg")
    img2 = Image.open("/home/psq/Documents/AdaptBan/heat_point/"+str(i)+".jpg")
    merge = Image.blend(img1, img2, 0.4)# blend_img = img1 * (1 – alpha) + img2* alpha
    merge.save("/home/psq/Documents/AdaptBan/heat_map/heat_"+str(i)+".jpg")

def heat_map(img4,videoname,id=0):
# 下面是代码片段
       # (batch,channel,width,hight: 64,2048,8,24)   img4是resnet50最后的输出，即你想要变成热力图的特征图

    # 将一个批次的特征图全部转化成热力图并保存
        img4_one = img4.squeeze(0)
        img4_one_channel_sum =img4_one[1,:,:].unsqueeze(0)# 通道降维，2048-->1
        inp = img4_one_channel_sum.cpu().detach().numpy() #  24,8  tensor转换成numpy

        sizee = img4_one_channel_sum.size() #2048，8，24 3
        # 生成图片对应的每个像素点的坐标
        x = np.array(([i for i in range(sizee[2])]))
        y = np.array(([i for i in range(sizee[1])]))
        xx = np.array(([i for i in range((sizee[2]-1)*8+16)]))
        yy = np.array(([i for i in range((sizee[1]-1)*8+16)]))
        #ravel让多为数组变一维
        _, _, U = expend_data(x,y, inp.ravel(), smooth_degree=10) #特征图放大
        U = np.flip(U,axis=0)# 翻转
##
##
        #开始用matplotlib画图
        fig, ax = plt.subplots(figsize=(0.57,1.85),subplot_kw={'xticks': [], 'yticks': []}) # 创建画布，图片分辨率=figsize*1000还是100来着？
        c = ax.pcolormesh(xx,yy,U, cmap='jet')# 坐标（xx，yy）和其对应的值（U）放进去。cmap是热力图颜色样式
        ax._frameon = False # 去除图片边框 
        plt.tight_layout(pad=0) # 自动调整填充区域
        save_path = '/home/psq/Documents/AdaptBan/heat_point/'+videoname+'/'+str(id)+'.jpg'
        if not os.path.exists('/home/psq/Documents/AdaptBan/heat_point/'+videoname):
             os.mkdir('/home/psq/Documents/AdaptBan/heat_point/'+videoname)
        plt.savefig(save_path)
        plt.close()

        test_img_6148 = Image.open(save_path)
        test_img_6148 = test_img_6148.resize((255,255)) # 为了美观，缩放为统一大小
        test_img_6148.save(save_path)
#x=torch.rand(1,3,25,25)
#heat_map(x,1)
#merge()