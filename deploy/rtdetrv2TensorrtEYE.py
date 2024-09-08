import time 
import contextlib
import collections
from collections import OrderedDict

import numpy as np
from PIL import Image, ImageDraw

import torch
import torchvision.transforms as T 
import torchvision
import cv2
import tensorrt as trt




class rtdetrv2TensorrtEYE:

    def __init__(self, model_path, threshold=0.6, num_classes=80, num_top_queries=300,  device='cuda:0',  max_batch_size=32, verbose=False):
        self.model_path = model_path
        self.device = device
        self.max_batch_size = max_batch_size
        self.threshold = threshold
        self.num_classes=num_classes
        self.num_top_queries = num_top_queries
        
        self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)  

        self.engine = self.load_engine(model_path)

        self.context = self.engine.create_execution_context()

        self.bindings = self.get_bindings(self.engine, self.context, self.max_batch_size, self.device)
        self.bindings_addr = OrderedDict((n, v.ptr) for n, v in self.bindings.items())

        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()

        #bcz of only one input layer
        self.N, self.C, self.W, self.H = self.bindings[self.input_names[0]].shape
        self.transforms = T.Compose([
                    T.ToPILImage(),
                    T.Resize((self.W, self.H)),
                    T.ToTensor(),
                ]) 

    def init(self, ):
        self.dynamic = False 

    def load_engine(self, path):
        '''load engine
        '''
        trt.init_libnvinfer_plugins(self.logger, '')
        with open(path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    
    def get_input_names(self, ):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names
    
    def get_output_names(self, ):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names

    def get_bindings(self, engine, context, max_batch_size=32, device=None) -> OrderedDict:
        '''build binddings
        '''
        Binding = collections.namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        bindings = OrderedDict()
        # max_batch_size = 1

        for i, name in enumerate(engine):
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))

            if shape[0] == -1:
                dynamic = True 
                shape[0] = max_batch_size
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:  # dynamic
                    context.set_input_shape(name, shape)

            data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, shape, data, data.data_ptr())

        return bindings

    def run_torch(self, blob):
        '''torch input
        '''
        for n in self.input_names:
            if self.bindings[n].shape != blob[n].shape:
                self.context.set_input_shape(n, blob[n].shape) 
                self.bindings[n] = self.bindings[n]._replace(shape=blob[n].shape)
            
            # TODO (lyuwenyu): check dtype, 
            assert self.bindings[n].data.dtype == blob[n].dtype, '{} dtype mismatch'.format(n)
            # if self.bindings[n].data.dtype != blob[n].shape:
            #     blob[n] = blob[n].to(self.bindings[n].data.dtype)

        self.bindings_addr.update({n: blob[n].data_ptr() for n in self.input_names})
        self.context.execute_v2(list(self.bindings_addr.values()))
        outputs = {n: self.bindings[n].data for n in self.output_names}

        return outputs
    
    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image)[None]
        blob = {
                'images': image.to(self.device)
            }
        return blob 
    
    def mod(self,a, b):
        out = a - a // b * b
        return out

    
    def inference(self, image):
        h,w,_ = image.shape
        orig_target_sizes = torch.tensor([w,h])
        blob = self.preprocess(image)
        model_output = self.run_torch(blob)
        logits, boxes = torch.Tensor(list(model_output.values())[0]), torch.Tensor(list(model_output.values())[1])
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy').to("cpu")
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)
        scores = torch.nn.functional.sigmoid(logits)
        scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)

        scores = scores.to('cpu')
        index = index.to('cpu')
        labels = self.mod(index, self.num_classes)
        labels = labels.to('cpu')
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


