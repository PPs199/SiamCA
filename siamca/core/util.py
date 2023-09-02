
import torch
import math
from siamca.core.config import cfg
class Point:
    """
    This class generate points.
    """
    def __init__(self, stride, size, image_center):
        self.stride = stride#8
        self.size = size#25
        self.image_center = image_center#255/2

        self.points = self.generate_points(self.stride, self.size, self.image_center)

    def generate_points(self, stride, size, im_c):  # im_c图像中心点
        ori = im_c - size // 2 * stride
        x, y = torch.meshgrid(torch.tensor([ori + stride * dx for dx in range(0, size)]),
                              torch.tensor([ori + stride * dy for dy in range(0, size)]))
        points = torch.zeros((2, size, size), dtype=torch.float32)
        points[0, :, :], points[1, :, :] = x.type(torch.float32), y.type(torch.float32)
        return points.cuda()

def myIOULoss(pre_boxes, gt_boxes,
                GIoU=False, DIoU=False, CIoU=True):
    #torch.Size([691, 4]) torch.Size([691, 4])

    #
    ### 1. to conner type box要将box变成x1,y1,x2,y2的形态


    ### 2.compute IOU
    b1_x1, b1_y1, b1_x2, b1_y2 = pre_boxes[:,0], pre_boxes[:,1], pre_boxes[:,2], pre_boxes[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = gt_boxes[:,0], gt_boxes[:,1], gt_boxes[:,2], gt_boxes[:,3]

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    #print(inter.shape, union.shape)
    iou = inter / union  # iou
    # print(iou.shape) #[691]
    # b
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            loss = iou - (c_area - union) / c_area  # GIoU
            loss = 1-loss
        else:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:

                loss = iou - rho2 / c2  # DIoU
                loss = 1-loss
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                loss = iou - (rho2 / c2 + v * alpha)  # CIoU
                loss = 1-loss

        loss = loss.mean()
    else:
        iou = -torch.log(iou + 1e-16) #防止为0
        loss = iou.mean()
    return loss
def Iou(pre_boxes, gt_boxes):
    b1_x1, b1_y1, b1_x2, b1_y2 = pre_boxes[:, 0], pre_boxes[:, 1], pre_boxes[:, 2], pre_boxes[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter
    # print(inter.shape, union.shape)
    iou = inter / union  # iou
    return iou
def convert_box(pred_loc, shape):
    point =Point(cfg.POINT.STRIDE, cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.SEARCH_SIZE//2)
    center = torch.zeros_like(pred_loc)
    center[:, 0, :, :] = pred_loc[:, 0, :, :] * shape[:, 0, :, :] + point.points[0]
    center[:, 1, :, :] = pred_loc[:, 1, :, :] * shape[:, 1, :, :] + point.points[1]
    center[:, 2, :, :] = torch.exp(pred_loc[:, 2, :, :]) * shape[:, 0, :, :]
    center[:, 3, :, :] = torch.exp(pred_loc[:, 3, :, :]) * shape[:, 1, :, :]
    return center
def mycenter2corner(center):
        conner = torch.zeros_like(center)
        # x, y, w, h = center[:,0,:,:], center[:,1,:,:], center[:,2,:,:], center[:,3,:,:]
        conner[:, 0, :, :] = center[:, 0, :, :] - center[:, 2, :, :] * 0.5
        conner[:, 1, :, :] = center[:, 1, :, :] - center[:, 3, :, :] * 0.5
        conner[:, 2, :, :] = center[:, 0, :, :] + center[:, 2, :, :] * 0.5
        conner[:, 3, :, :] = center[:, 1, :, :] + center[:, 3, :, :] * 0.5
        return conner
