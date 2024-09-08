import torch
import torch.nn as nn 
import torchvision.transforms as T
import time
import numpy as np 
import cv2
import torchvision
from rtdetrv2_pytorch.src.core import YAMLConfig


class Model(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.model = cfg.model.deploy()
        
    def forward(self, images):
        outputs = self.model(images)
        return outputs
    
    
class rtdetrv2TorchEYE:

    def __init__(self,  model_path, config_path, threshold=0.6, device='cuda:0',use_focal_loss=True, num_top_queries=300, num_classes=80):

        self.threshold = threshold
        self.use_focal_loss=use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = num_classes
        self.device = device
        cfg = YAMLConfig(config_path, resume=model_path)
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
        cfg.model.load_state_dict(state)
        self.model = Model(cfg).to(self.device)
        self.transforms = T.Compose([
                            T.ToPILImage(),
                            T.Resize((640, 640)),
                            T.ToTensor(),
                        ])

    def mod(self,a, b):
        out = a - a // b * b
        return out
    
    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image)[None].to(self.device)
        return image 
    
    def inference(self, image):
        h,w,_ = image.shape
        orig_target_sizes = torch.tensor([w,h]).to(self.device)
        image = self.preprocess(image)  
        model_output = self.model(image)

        logits, boxes = torch.Tensor(list(model_output.values())[0]), torch.Tensor(list(model_output.values())[1])
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy').to(self.device)
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)
        scores = torch.nn.functional.sigmoid(logits)
        scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)

        scores = scores.to('cpu')
        index = index.to('cpu')
        labels = self.mod(index, self.num_classes)
        labels = labels.to('cpu')
        index = index // self.num_classes
        bbox_pred = bbox_pred.to("cpu")
        boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))

        scr = scores[0]
        lab = labels[0][scr > self.threshold]
        box = boxes[0][scr > self.threshold]
        scrs = scores[0][scr > self.threshold]

        results = []
        for i, bx in enumerate(box):
            bx = bx.detach().numpy()
            x1, y1, x2, y2 = bx
            results.append(
                {
                    "conf": scrs[i],
                    "class_id": lab[i],
                    "xmin":x1,
                    "ymin":y1,
                    "xmax":x2,
                    "ymax":y2 
                }
            )
        return results

        




