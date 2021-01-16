from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import torch
from torch import nn

class DecodeBox(nn.Module):
    """Some Information about DecodeBox"""
    def __init__(self, anchors, num_classes, img_size):
        super(DecodeBox, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size

    def forward(self, input):
        # shape of input: batch_size, [3 * (1+4+num_classes), 13,  13]
        batch_size = input.size(0)
        input_height = input.size(2)
        input_width = input.size(3)

        # rate = image / grid
        stride_h = self.img_size[1] / input_height
        stride_w = self.img_size[0] / input_width

        # grid = image / rate
        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in self.anchors]
        
        #    batch_size, 3, (4+1+num_classes), 13, 13
        # -> batch_size, 3, 13, 13, (4+1+num_classes)
        prediction = input.view(batch_size, self.num_anchors, self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

        # batch_size, 3, 13, 13
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]

        conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        # batch_size, 3, 13, 13
        grid_x = torch.linspace(0, input_width-1, input_width).repeat(
            input_width, 1).repeat(batch_size*self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, input_height-1, input_height).repeat(
            input_height, 1).repeat(batch_size*self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)

        anchor_w = torch.FloatTensor(
            scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = torch.FloatTensor(
            scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(
            1, 1, input_height*input_width).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(
            1, 1, input_height*input_width).view(h.shape)

        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        return input
