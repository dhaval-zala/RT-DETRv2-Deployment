import torch 
import torchvision.transforms as T 
import numpy as np 
import onnxruntime as ort 
import cv2
import torchvision

class rtdetrv2OnnxEYE:

    def __init__(self, model_path, threshold,use_focal_loss=True, num_top_queries=300, num_classes=80 ):

        self.threshold = threshold 
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries 
        self.num_classes = num_classes
        self.transforms = T.Compose([
                            T.ToPILImage(),
                            T.Resize((640, 640)),
                            T.ToTensor(),
                        ])
        self.model = ort.InferenceSession(model_path)

    def mod(self,a, b):
        out = a - a // b * b
        return out
    
    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image)[None]
        return image 
    
    def inference(self, image):
        h,w,_ = image.shape
        orig_target_sizes = torch.tensor([w,h])
        image = self.preprocess(image)        
        model_output = self.model.run(
            output_names = None,
            input_feed = {'images': image.data.numpy()}
        )
        logits, boxes = torch.Tensor(list(model_output)[0]), torch.Tensor(list(model_output)[1])
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)
        scores = torch.nn.functional.sigmoid(logits)
        scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
        labels = self.mod(index, self.num_classes)
        index = index // self.num_classes
        boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))

        scr = scores[0]
        lab = labels[0][scr > self.threshold]
        box = boxes[0][scr > self.threshold]
        scrs = scores[0][scr > self.threshold]

        results = []
        for i, bx in enumerate(box):
            bx = bx.numpy()
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


    