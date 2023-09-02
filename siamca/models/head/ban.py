from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
from siamca.core.Defrom2 import DeformConv2d
from siamca.core.xcorr import xcorr_fast, xcorr_depthwise,pg_xcorr
class BAN(nn.Module):
    def __init__(self):
        super(BAN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError


class Channel_Attention(nn.Module):

    def __init__(self, channel):
        super(Channel_Attention, self).__init__()

        self.__avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.__max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.__fc = nn.Sequential(
            nn.Conv2d(channel, channel // 16, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(channel // 16, channel, 1, bias=False),
        )
        self.__sigmoid = nn.Sigmoid()

    def forward(self, x):
        y1 = self.__avg_pool(x)
        y1 = self.__fc(y1)

        y2 = self.__max_pool(x)
        y2 = self.__fc(y2)

        y = self.__sigmoid(y1 + y2)
        return x * y


class Spartial_Attention(nn.Module):

    def __init__(self, ):
        super(Spartial_Attention, self).__init__()

        self.__layer = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_mask = torch.mean(x, dim=1, keepdim=True)
        max_mask, _ = torch.max(x, dim=1, keepdim=True)
        mask = torch.cat([avg_mask, max_mask], dim=1)

        mask = self.__layer(mask)
        return x * mask


class UPChannelBAN(BAN):
    def __init__(self, feature_in=256, cls_out_channels=2):
        super(UPChannelBAN, self).__init__()

        cls_output = cls_out_channels
        loc_output = 4
        self.template_cls_conv = nn.Conv2d(feature_in,  
                                           feature_in * cls_output, kernel_size=3)
        self.template_loc_conv = nn.Conv2d(feature_in,
                                           feature_in * loc_output, kernel_size=3)

        self.search_cls_conv = nn.Conv2d(feature_in,
                                         feature_in, kernel_size=3)
        self.search_loc_conv = nn.Conv2d(feature_in,
                                         feature_in, kernel_size=3)

        self.loc_adjust = nn.Conv2d(loc_output, loc_output, kernel_size=1)

    def forward(self, z_f, x_f):
        cls_kernel = self.template_cls_conv(z_f) 
        loc_kernel = self.template_loc_conv(z_f)  

        cls_feature = self.search_cls_conv(x_f) 
        loc_feature = self.search_loc_conv(x_f)

        cls = xcorr_fast(cls_feature, cls_kernel)  
        loc = self.loc_adjust(xcorr_fast(loc_feature, loc_kernel)) 
        return cls, loc
class Graph_Attention_Union(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Graph_Attention_Union, self).__init__()

        # search region nodes linear transformation
        self.support = nn.Conv2d(in_channel, in_channel, 1, 1)

        # target template nodes linear transformation
        self.query = nn.Conv2d(in_channel, in_channel, 1, 1)

        # linear transformation for message passing
        self.g = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )

        # aggregated feature
        self.fi = nn.Sequential(
            nn.Conv2d(in_channel*3, out_channel, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, zf, xf):
        # linear transformation
        xf_trans = self.query(xf)
        zf_trans = self.support(zf)

        # linear transformation for message passing
        xf_g = self.g(xf)
        zf_g = self.g(zf)

        # calculate similarity
        shape_x = xf_trans.shape
        shape_z = zf_trans.shape
        zf_g_plain = zf_g.view(-1, shape_z[1], shape_z[2] * shape_z[3]).permute(0, 2, 1)  
        xf_g_plain = xf_g.view(-1, shape_x[1], shape_x[2] * shape_x[3]).permute(0, 2, 1)  
       
        zf_trans_plain = zf_trans.view(-1, shape_z[1], shape_z[2] * shape_z[3])

        xf_trans_plain = xf_trans.view(-1, shape_x[1], shape_x[2] * shape_x[3]).permute(0, 2, 1)

        xf_trans_plain2=xf_trans.view(-1, shape_x[1], shape_x[2] * shape_x[3])


        similar = torch.matmul(xf_trans_plain, zf_trans_plain)
        similar = F.softmax(similar, dim=2)

        self_similar=torch.matmul(xf_trans_plain, xf_trans_plain2)
        self_similar=F.softmax(self_similar, dim=2)
        self_embedding=torch.matmul(self_similar, xf_g_plain).permute(0, 2, 1)
        self_embedding = self_embedding.view(-1, shape_x[1], shape_x[2], shape_x[3])
        #
        embedding = torch.matmul(similar, zf_g_plain).permute(0, 2, 1)
        embedding = embedding.view(-1, shape_x[1], shape_x[2], shape_x[3])

        # aggregated feature
        output = torch.cat([embedding, self_embedding,xf_g], 1)
        output = self.fi(output)
        return output

class CBAM(nn.Module):
    def __init__(self):
        super(CBAM, self).__init__()
        self.channel = Channel_Attention(256)
        self.spaec = Spartial_Attention()
        self.Relu = nn.ReLU(inplace=True)

    def forward(self, x):
        CA = self.channel(x)
        SA = self.spaec(CA)
        return self.Relu(x + SA)

class Dodeform(nn.Module):
    def __init__(self):
        super(Dodeform, self).__init__()

        self.Deform = DeformConv2d(inc=256, outc=256)
        self.Relu = nn.ReLU(inplace=True)
        self.Relu2=nn.ReLU(inplace=True)
    def forward(self, feature, offset):
        feature1 = self.Relu(self.Deform(feature, offset))
        return self.Relu2(feature1+feature)
class ShapeHand(nn.Module):
    def __init__(self, in_channels=256*3, out_channels=256,kernel_size=3):
        super(ShapeHand, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_search = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 2, kernel_size=1),
            nn.Tanh()
        )
        self.mergeTemplete=nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size=kernel_size, bias=False,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.selfatt = Graph_Attention_Union(256, 256)
    def forward(self,z_f,z_f2,x_f,mask):
        z=torch.cat(z_f,1)
        x=torch.cat(x_f,1)
        z2=torch.cat(z_f2,1)
        shape_kernel=self.downsample(z)
        shape_kernel2=self.downsample(z2)
        shape_search=self.downsample2(x)

        kernel = shape_kernel * mask
        kernel2=shape_kernel2
        
        kernel =self.mergeTemplete(torch.cat((kernel,kernel2),dim=1))
        search = self.conv_search(shape_search)
        kernel = self.conv_kernel(kernel)
        feature = self.selfatt(kernel, search)
        shape=self.head(feature)
        return shape
class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),

        )
        self.conv_search= nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv_kernel2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv_search2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, kernel_size=1)
        )
        self.head2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 4, kernel_size=1)
        )
        self.cls= nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),

        )
        self.loc= nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.mergeTemplete=nn.Sequential(
            nn.Conv2d(hidden*2, hidden, kernel_size=kernel_size, bias=False,padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.fi = nn.Sequential(
            nn.Conv2d(256 * 2, 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.attloc=CBAM()
        self.attcls=CBAM()
        self.offset = nn.Conv2d(2, 18, kernel_size=1)  # 用shape生成偏移量
        self.Deform = Dodeform()
        self.selfatt=Graph_Attention_Union(256,256)
    def forward(self, z,z2, x,mask,shape):
        kernel=z*mask
        kernel2=z2
        kernel =self.mergeTemplete(torch.cat((kernel,kernel2),dim=1))
        kernel=self.conv_kernel2(kernel)
        search=self.conv_search2(x)
        feature1=self.selfatt(kernel,search)
        search=self.conv_search(x)
        l = 4
        r = l + 7
        z = z[:, :, l:r, l:r]
        kernel=self.conv_kernel(z)
        feature2 = xcorr_depthwise(search, kernel)  # (batch,hidden,w3,h3)
        offset = self.offset(shape)
        feature=self.fi(torch.cat([feature1,feature2],1))
        feature = self.attloc(feature)
        clsfeature=self.cls(feature)
        thefeature = self.Deform(feature, offset)
        locfeature=self.loc(thefeature)
        cls = self.head(clsfeature)  # (batch,out_channel,w3,h3)
        loc = self.head2(locfeature)
        return cls, loc


class DepthwiseBAN(BAN):
    def __init__(self, in_channels=256, out_channels=256, cls_out_channels=2, weighted=False):
        super(DepthwiseBAN, self).__init__()
        self.out = DepthwiseXCorr(in_channels, out_channels, cls_out_channels)
    # 更改中
    def forward(self, z_f,z_f2, x_f,t_box,shape):
        cls, loc = self.out(z_f,z_f2, x_f,t_box,shape)
        return cls, loc


class MultiBAN(BAN):
    def __init__(self, in_channels, cls_out_channels, weighted=False):  # in_channels[256.256.256].cls=2
        super(MultiBAN, self).__init__()
        self.weighted = weighted

        for i in range(len(in_channels)):
            self.add_module('box' + str(i + 2), DepthwiseBAN(in_channels[i], in_channels[i], cls_out_channels))
        if self.weighted:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))  # [1,1,1]
            self.loc_weight = nn.Parameter(torch.ones(len(in_channels)))  # [1,1,1]
            # self.loc_scale = nn.Parameter(torch.ones(len(in_channels)))#[1,1,1]
        # 修改
        self.creatshape = ShapeHand()

    def forward(self, z_fs,z_fs2, x_fs,t_box):#两个模板特征融入
        cls = []
        loc = []
        mask = torch.zeros(z_fs[0].shape).cuda()
       
        roi = (torch.round((t_box + 1 - 7) / 8)).int()
      
        for i in range(z_fs[0].shape[0]):
            mask[i, :, max(0, roi[i][1]): (min(roi[i][3], 14)), max(0, roi[i][0]): (min(roi[i][2], 14))] = 1
              
        shape = self.creatshape(z_fs,z_fs2, x_fs,mask)
        for idx, (z_f,z_f2, x_f) in enumerate(zip(z_fs,z_fs2, x_fs), start=2):
            box = getattr(self, 'box' + str(idx))
            c, l= box(z_f,z_f2, x_f,mask,shape)
            cls.append(c)
            # loc.append(torch.exp(l * self.loc_scale[idx - 2]))
            loc.append(l)
        if self.weighted:
            cls_weight = F.softmax(self.cls_weight, 0)
            loc_weight = F.softmax(self.loc_weight, 0)
            shape_weight=F.softmax(self.loc_weight, 0)
        def avg(lst):
            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s

        if self.weighted:
            return weighted_avg(cls, cls_weight), weighted_avg(loc, loc_weight),torch.exp(shape)
        else:
            return avg(cls), avg(loc),torch.exp(shape)
