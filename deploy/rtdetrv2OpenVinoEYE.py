import cv2
import openvino as ov 
import torch 
import torchvision
import torchvision.transforms as T
import numpy as np



# class rtdetrv2OpenVinoEYE:
#     def __init__(self, model_path, threshold=0.6,use_focal_loss=True, num_top_queries=300, num_classes=80, remap_mscono_category=True):
        
#         self.threshold = threshold
#         self.use_focal_loss=use_focal_loss
#         self.num_top_queries = num_top_queries
#         self.num_classes = num_classes
#         self.remap_mscoco_category = remap_mscono_category
#         self.model = ov.Core().compile_model(model_path)
#         input_ir = self.model.input(0)
#         self.N, self.C, self.H, self.W = input_ir.partial_shape
#         self.W = self.W.get_length()
#         self.H = self.H.get_length()
#         self.transforms = T.Compose([
#             T.ToPILImage(),
#             T.Resize((self.W, self.H)),
#             T.ToTensor(),
#         ])
#         # self.tmp_cnt = 0
#         # self.logits = None
#         # self.boxes = None

#     def mod(self,a, b):
#         out = a - a // b * b
#         return out

#     def preprocess(self, image):
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = self.transforms(image)[None]
#         image = np.array(image)
#         return image

#     def inference(self, image):
#         h,w,_ = image.shape
#         orig_target_sizes = torch.tensor([w,h])

#         image = self.preprocess(image)
#         # if self.tmp_cnt==0:

#         model_output = self.model(ov.Tensor(image))
#         logits, boxes = torch.Tensor(list(model_output.values())[0]), torch.Tensor(list(model_output.values())[1])
#         #     self.logits, self.boxes = logits, boxes
#         # logits, boxes = self.logits, self.boxes
#         # self.tmp_cnt+=1
#         bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
#         bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)
#         scores = torch.nn.functional.sigmoid(logits)
#         scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
#         labels = self.mod(index, self.num_classes)
#         index = index // self.num_classes
#         boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))

#         scr = scores[0]
#         lab = labels[0][scr > self.threshold]
#         box = boxes[0][scr > self.threshold]
#         scrs = scores[0][scr > self.threshold]

#         results = []
#         for i, bx in enumerate(box):
#             bx = bx.numpy()
#             x1, y1, x2, y2 = bx
#             results.append(
#                 {
#                     "conf": scrs[i],
#                     "class_id": lab[i],
#                     "xmin":x1,
#                     "ymin":y1,
#                     "xmax":x2,
#                     "ymax":y2 
#                 }
#             )
#         return results
        



        

import cv2
import openvino as ov
import torch 
import torchvision
import torchvision.transforms as T
import numpy as np
import asyncio



class rtdetrv2OpenVinoEYE:
    def __init__(self, model_path, threshold=0.6,use_focal_loss=True, num_top_queries=300, num_classes=80):
        
        self.core = ov.Core()
        self.threshold = threshold
        self.use_focal_loss=use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = num_classes
        self.remap_mscoco_category = True

        config = {
            # "CPU_THROUGHPUT_STREAMS": "1",
            # "PERFORMANCE_HINT": "THROUGHPUT",
            # "PERFORMANCE_HINT_NUM_REQUESTS": 2,
            # "ENABLE_CPU_PINNING": True,
            # "INFERENCE_NUM_THREADS": 16,
            # "NUM_STREAMS": str(self.jobs),
        }

        self.model = self.core.compile_model(model_path, "CPU", config=config)
        self.infer_request_curr = self.model.create_infer_request()
        self.infer_request_next = self.model.create_infer_request()
        self.last_frame = None

        self.input_ir = self.model.input(0)
        print((self.input_ir.get_partial_shape))
        # raise ValueError('sdf')
        self.N, self.C, self.H, self.W = self.input_ir.partial_shape
        self.W = self.W.get_length()
        self.H = self.H.get_length()
        self.transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((self.W, self.H)),
            T.ToTensor(),
        ])
        self.last_frame = None
        # self.tmp_cnt = 0
        # self.logits = None
        # self.boxes = None

    def mod(self,a, b):
        out = a - a // b * b
        return out

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image)[None]
        image = np.array(image)
        return image

    def inference(self, image):
        ori_image = image.copy()
        image = self.preprocess(image)

        self.infer_request_next.set_tensor(self.input_ir, ov.Tensor(image))
        self.infer_request_next.start_async()

        results = []
        
        if self.last_frame is None:
            pass
        
        else:
            self.infer_request_curr.wait()
            model_output = [self.infer_request_curr.get_output_tensor(0),self.infer_request_curr.get_output_tensor(1)] 
            
            results = self.callback(model_output, self.last_frame)


        self.last_frame = ori_image
        self.infer_request_curr, self.infer_request_next = self.infer_request_next, self.infer_request_curr
        
        return results
    
    async def inference_async(self, image):
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self.inference, image)
        return result
    

    def callback(self,model_output, image):

        h,w,_ = image.shape
        orig_target_sizes = torch.tensor([w,h])

        
        logits, boxes = torch.Tensor(model_output[0].data), torch.Tensor(model_output[1].data)
        #     self.logits, self.boxes = logits, boxes
        # logits, boxes = self.logits, self.boxes
        # self.tmp_cnt+=1
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



        

