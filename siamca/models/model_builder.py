# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F

from siamca.core.config import cfg
from siamca.models.loss import select_cross_entropy_loss, select_shape_loss,select_lociou_loss
from siamca.models.backbone import get_backbone
from siamca.models.head import get_ban_head
from siamca.models.neck import get_neck
import torch

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build ban head
        if cfg.BAN.BAN:
            self.head = get_ban_head(cfg.BAN.TYPE,
                                     **cfg.BAN.KWARGS)

    def template(self, z,template_box):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf
        self.template_box=template_box
     
        self.updata_zf=zf
    def update(self,upz):
        upTemplate = self.backbone(upz)
        if cfg.ADJUST.ADJUST:
            upTemplate = self.neck(upTemplate)
        self.updata_zf=upTemplate
      
    def track(self, x):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, loc,shape = self.head(self.zf,self.updata_zf, xf,self.template_box)
       
        return {
                'cls': cls,
                'loc': loc,
                'shape':shape
               }

    def log_softmax(self, cls):
        if cfg.BAN.BAN:
            cls = cls.permute(0, 2, 3, 1).contiguous()
            cls = F.log_softmax(cls, dim=3)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        template_box=data['template_box'].cuda()
        template2 = data['template2'].cuda() 
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc1 = data['label_loc'].cuda()
        label_shape=data['shape'].cuda()
        
        # get feature
        zf = self.backbone(template)
        zf2=self.backbone(template2)
        xf = self.backbone(search)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            zf2= self.neck(zf2)
            xf = self.neck(xf)
        cls, loc,shape = self.head(zf, zf2,xf,template_box)
        shapec=shape*64
        label_shapec=label_shape*64
        
        # get loss

        # cls loss with cross entropy loss
        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        # label_loc2
        label_loc3=label_loc1/(shapec+1e-7)
        label_loc2 = torch.log(label_shapec / (shapec+1e-7))
        # loc loss with iou loss
        label_loc=torch.cat([label_loc3,label_loc2],dim=1)

        loc_loss = select_lociou_loss(loc, label_loc, shapec,label_cls)
        

        shape_loss=select_shape_loss(shape,label_shape,label_cls)
        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss \
                               +(0.1*shape_loss)
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['shape_loss'] =0.1*shape_loss
        return outputs