from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F
import torch
import cv2
from siamca.core.config import cfg
from siamca.tracker.base_tracker import SiameseTracker
from siamca.utils.bbox import corner2center
class SiamCATrackerLaSOT(SiameseTracker):
    def __init__(self, model):
        super(SiamCATrackerLaSOT, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
                          cfg.POINT.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.cls_out_channels = cfg.BAN.KWARGS.cls_out_channels
        self.window = window.flatten()
        self.points = self.generate_points(cfg.POINT.STRIDE, self.score_size)
        self.model = model
        self.model.eval()
        print('我是lasot的ce11')
    def generate_points(self, stride, size):
        ori = - (size // 2) * stride
        x, y = np.meshgrid([ori + stride * dx for dx in np.arange(0, size)],
                           [ori + stride * dy for dy in np.arange(0, size)])
        points = np.zeros((size * size, 2), dtype=np.float32)
        points[:, 0], points[:, 1] = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()

        return points

    def _convert_bbox(self, delta, point,shape):#(self,delta,point,shape)
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)#[4,25x25]
        delta = delta.detach().cpu().numpy()
        shape = shape.permute(1, 2, 3, 0).contiguous().view(2, -1)  # [4,25x25]

        shape = shape.detach().cpu().numpy()
        delta[0, :] =delta[0, :] *shape[0,:] + point[:,0]
        delta[1, :] = delta[1, :] *shape[1,:]+  point[:,1]
        delta[2, :] = np.exp(delta[2, :]) * shape[0, :]
        delta[3, :] = np.exp(delta[3, :]) * shape[1, :]
        # delta[0, :], delta[1, :], delta[2, :], delta[3, :] = corner2center(delta)#这个可以不要
        #我预测的直接是center，直接return不需要再改delta了
        return delta

    def _convert_score(self, score):
        if self.cls_out_channels == 1:
            score = score.permute(1, 2, 3, 0).contiguous().view(-1)
            score = score.sigmoid().detach().cpu().numpy()
        else:
            score = score.permute(1, 2, 3, 0).contiguous().view(self.cls_out_channels, -1).permute(1, 0)
            score = score.softmax(1).detach()[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox,name):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        
        scale = cfg.TRACK.EXEMPLAR_SIZE / s_z
        self.c = (cfg.TRACK.EXEMPLAR_SIZE - 1) / 2  # 图片中心的一半
        roi = torch.tensor([[self.c - bbox[2] * scale / 2, self.c - bbox[3] * scale / 2,
                             self.c + bbox[2] * scale / 2, self.c + bbox[3] * scale / 2]])
        self.model.template(z_crop,roi)#init的时候就把z弄好了
        self.frame_id = 0
        self.best_score=1.0
        self.lostnum=0
        self.last_score=0.0
    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        if(self.best_score<0.80 and self.lostnum>3):
             w_z = self.size[0]*1 + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size*1)
             h_z = self.size[1]*1 + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size*1)
             self.lostnum=0
        else:
             w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
             h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        
        
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)
        self.frame_id += 1
        outputs = self.model.track(x_crop)


        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.points,outputs['shape']*64)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)
        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])
       
        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        self.best_score = score[best_idx]
        if self.best_score<0.85 and self.last_score<0.85:
            self.last_score=self.best_score
            self.lostnum+=1
        else:
            self.last_score=self.best_score
            self.lostnum=0
        #print(self.frame_id,self.best_score)
        if self.frame_id > 100   and self.best_score > 0.96 and self.lostnum==0:#帧数为200就更新
            #先获取 更新模板的裁剪图片 和 roi
             w_z = self.size[0]*1.1+ cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
             h_z = self.size[1]*1.1 + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
             s_z = round(np.sqrt(w_z * h_z))
             updatez_crop = self.get_subwindow(img, self.center_pos,
                                  cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
            #更新跟踪器里的 这两个值
             self.model.update(updatez_crop)
             self.frame_id=1
            #结束
        return {
                'bbox': bbox,
                'best_score': self.best_score
               }
