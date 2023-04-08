import torch
import torchvision.transforms as T
from util.misc import MetricLogger, nested_tensor_from_tensor_list
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import random
from torchinfo import summary
# import argparse
# from models.detr import DETR
# from models.backbone import build_backbone
# from models.transformer import build_transformer
# from main import get_args_parser

# parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
# args = parser.parse_args()


# print(args)
# backbone = build_backbone(args)
# transformer = build_transformer(args)

# num_classes = 91
# model = DETR(
#     backbone,
#     transformer,
#     num_classes=num_classes,
#     num_queries=args.num_queries,
#     aux_loss=args.aux_loss,
# )


from hubconf import detr_resnet50

model = detr_resnet50(pretrained=False, return_postprocessor=False)
model.eval()
model = model.cuda()
summary(model, input_size=(1, 3, 480, 600), depth=100)
# print(model)
# CLASSES = [
#     'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#     'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
#     'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
#     'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
#     'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
#     'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
#     'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
#     'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
#     'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
#     'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
#     'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
#     'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
#     'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
#     'toothbrush'
# ]

# transform = T.Compose([
#     T.Resize(800),
#     T.ToTensor(),
#     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])


# def box_cxcywh_to_xyxy(x):
#     x_c, y_c, w, h = x.unbind(1)
#     b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
#          (x_c + 0.5 * w), (y_c + 0.5 * h)]
#     return torch.stack(b, dim=1)


# def rescale_bboxes(out_bbox, size):
#     img_w, img_h = size
#     b = box_cxcywh_to_xyxy(out_bbox)
#     b = b.cpu() * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
#     return b


# def plot_one_box(x, img, color=None, label=None, line_thickness=3):
#     # Plots one bounding box on image img
#     tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
#     color = color or [random.randint(0, 255) for _ in range(3)]
#     c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
#     cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
#     if label:
#         tf = max(tl - 1, 1)  # font thickness
#         t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
#         c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
#         cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
#         cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


# # 000000145887.jpg  000000290557.jpg  000000436731.jpg  000000581911.jpg
# # 000000145893.jpg  000000290579.jpg  000000436733.jpg  000000581918.jpg
# # 000000145907.jpg  000000290580.jpg  000000436739.jpg
# # 000000145934.jpg  000000290640.jpg  000000436789.jpg

# img_path = "../test2017/000000145887.jpg"
# img_path = "pic/human_dog.jpg"
# img = Image.open(img_path).convert('RGB')
# img_tensor = transform(img).unsqueeze(0).cuda()

# with torch.no_grad():
#     output = model(img_tensor)

# thresh_hold = 0.7
# class_prob = output['pred_logits'].softmax(-1)[0, :, :-1]
# keep = class_prob.max(-1).values > thresh_hold

# scores = class_prob[keep]
# boxes = rescale_bboxes(output['pred_boxes'][0, keep], img.size)

# pred_logits = output['pred_logits'][0][:, :len(CLASSES)]
# pred_boxes = output['pred_boxes'][0]


# img_result = cv2.imread(img_path)

# for score, box in zip(scores, boxes):
#     cls = score.argmax()
#     label = f'{CLASSES[cls]} {score[cls]:.2f}'
#     plot_one_box(box, img_result, label=label)

# cv2.imwrite("img_result.jpg", img_result)
