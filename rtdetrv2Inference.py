
import json 
import numpy as np 

import os
import cv2 
import time 
from deploy.rtdetrv2OpenVinoEYE import rtdetrv2OpenVinoEYE
from deploy.rtdetrv2TensorrtEYE import rtdetrv2TensorrtEYE
from deploy.rtdetrv2OnnxEYE import rtdetrv2OnnxEYE
from deploy.rtdetrv2TorchEYE import rtdetrv2TorchEYE

def get_class_color(class_id):
    np.random.seed(class_id)  # Seed with the class ID to get a consistent color for each class
    color = np.random.randint(0, 255, 3).tolist()  # Generate a random color
    return color


def main(args, ):
    

    with open("rtdetrv2_pytorch/configs/classes.json", 'r') as file:
        classes = json.load(file)
        
    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"ERROR: The file at {args.video_path} does not exist.")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"ERROR: The file at {args.model_path} does not exist.")
    
    if args.model_path.endswith('.pth'):
        if args.config:
            if not os.path.exists(args.config):
                raise FileNotFoundError(f"ERROR: The file at {args.config} does not exist.")
        else:
             raise ValueError(f"Error: The required config path is missing.")
        detector = rtdetrv2TorchEYE(model_path=args.model_path,config_path=args.config, num_classes=args.num_classes, threshold=0.6, device=args.device)

    if args.model_path.endswith('.onnx'):
        detector = rtdetrv2OnnxEYE(model_path=args.model_path, num_classes=args.num_classes, threshold=0.6)

    if args.model_path.endswith('.xml'):
        detector = rtdetrv2OpenVinoEYE(model_path=args.model_path, threshold=0.6, num_classes=args.num_classes)

    if args.model_path.endswith('.trt'):
        detector = rtdetrv2TensorrtEYE(model_path=args.model_path, num_classes=80, threshold=0.6)

    cap = cv2.VideoCapture(args.video_path)
    
    if not cap.isOpened():
        raise IOError("ERROR: Could not read the stream...!")
    
    _, frame = cap.read()
    frame_width,frame_height,_ = frame.shape

    if args.output_path:
        folder_path = os.path.dirname(args.output_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(args.output_path, fourcc, 55, (frame_height, frame_width))
    
    start_time = time.time()
    frame_cnt = 0
    while cap.isOpened():

        _, frame = cap.read()
        frame_cnt += 1
        results = detector.inference(frame)
        for object in results:
            conf = object["conf"]
            class_id = object["class_id"].numpy()
            xmin, xmax, ymin, ymax = object["xmin"], object["xmax"], object["ymin"], object["ymax"]

            color = get_class_color(class_id)
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, thickness=1)
            cv2.putText(frame, f'{classes[str(class_id+1)]}', (int(xmin)-10, int(ymin)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        end_time = time.time()
        try:
            fps = frame_cnt/ (end_time - start_time)
            cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # print(fps)
        except:
            pass

        if args.display==True:
            cv2.imshow('RT-DETRv2 Output', frame)
            # Press 'q' to exit the video early
        
            if cv2.waitKey(1) == ord('q'):
                break
        
        if args.output_path:
            out.write(frame)

    cap.release()
    if args.output_path:
        out.release()
cv2.destroyAllWindows()

        

    


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='Path to the configuration file')
    parser.add_argument('-m', '--model_path', type=str, help='Path to the model file', required=True)
    parser.add_argument('-v', '--video_path', type=str, help='Path to the video file', required=True)
    parser.add_argument('-o', "--output_path", type=str)
    parser.add_argument('-ds', "--display", type=bool, default=True)
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument( '--num_classes', type=int, default=80)
    args = parser.parse_args()
    main(args)

