

<h2 align="center">🔥RT-DETRv2: Deployment (CPU / GPU)</h2>


<div align="center">
<img width="150" alt="image" src="./rtdetrv2_pytorch/extra/pytorch.png">
<img width="150" alt="image" src="./rtdetrv2_pytorch/extra/onnxruntime.jpeg">
<img width="180" alt="image" src="./rtdetrv2_pytorch/extra/openvino.png">
<img width="120" alt="image" src="./rtdetrv2_pytorch/extra/tensorrt.png">

</div>



---


- [RT-DETR Official GitHub](https://github.com/lyuwenyu/RT-DETR)
- [DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069)
- [RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer](https://arxiv.org/abs/2407.17140)


## Quick start

<video width="1000" height="600" controls>
  <source src="./rtdetrv2_pytorch/extra/inference.mp4" type="video/mp4">
</video>



## 📍 Installation

```
_________common installation_________
pip install opencv-python
pip install torch torchvision torchaudio
pip install pycocotools
pip install PyYAML
pip install tensorboard

_______onnx installation_______
onnx                          1.15.0
onnxruntime-gpu               1.19.2

_______openvino installation_______
openvino                      2024.0.0
openvino-dev                  2024.0.0
openvino-telemetry            2024.1.0

_______tensortrt installation_______
tensorrt                      8.6.1
tensorrt-bindings             8.6.1
tensorrt-libs                 8.6.1

```


## 📍 [ONNX, OpenVino and Tensorrt conversion](convert_rtdetrv2.ipynb)

- <span style="color: orange;">**PyTorch**</span> -> <span style="color: #555555;">**ONNX**</span>
```
python deploy/export_onnx.py -c rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml -r deploy/models/torchmodels/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth --output_file deploy/models/onnxmodels/model.onnx --check

```
- <span style="color: #555555;">**ONNX**</span> --> <span style="color:darkblue">**OpenVINO**</span>   (FP32)

```
!mo --input_model deploy/models/onnxmodels/model.onnx --output_dir deploy/models/openvinomodels

 ```

- INT8 <span style="color:darkblue">**OpenVINO**</span> Quantization [here](convert_rtdetrv2.ipynb)

- <span style="color: #555555;">**ONNX**</span> -> <span style="color: #009B77;">**TRT**</span> (FP16) [here](convert_rtdetrv2.ipynb)

- <span style="color: #555555;">**ONNX**</span> -> <span style="color: #009B77;">**TRT**</span> (INT8) [here](convert_rtdetrv2.ipynb)


## 📍 Inference

- <span style="color: orange;">**PyTorch**</span>
```
  python rtdetrv2Inference.py  -m path/to/.pth -c rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_xxx_xxx_coco_.yml -v path/to/video  --device cuda:0
  ```

- <span style="color: #555555;">**ONNX**</span>

```
python rtdetrv2Inference.py  -m path/to/.onnx -v path/to/video
```

- <span style="color:darkblue">**OpenVINO**</span> 

```
python rtdetrv2Inference.py  -m path/to/.xml -v path/to/video
```
- <span style="color: #009B77;">**TRT**</span>

```
python rtdetrv2Inference.py  -m path/to/.trt -v path/to/video
```



## 📍 Model Zoo

### Base models

| Model | Dataset | Input Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | #Params(M) | FPS | config| checkpoint | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
**RT-DETRv2-S** | COCO | 640 | **48.1** <font color=green>(+1.6)</font> | **65.1** | 20 | 217 | [config](./rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth) |
**RT-DETRv2-M**<sup>*<sup> | COCO | 640 | **49.9** <font color=green>(+1.0)</font> | **67.5** | 31 | 161 | [config](./rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r34vd_120e_coco.yml) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r34vd_120e_coco_ema.pth)
**RT-DETRv2-M** | COCO | 640 | **51.9** <font color=green>(+0.6)</font> | **69.9** | 36 | 145 | [config](./rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_m_7x_coco_ema.pth)
**RT-DETRv2-L** | COCO | 640 | **53.4** <font color=green>(+0.3)</font> | **71.6** | 42 | 108 | [config](./rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_6x_coco_ema.pth)
**RT-DETRv2-X** | COCO | 640 | 54.3 | **72.8** <font color=green>(+0.1)</font> | 76 | 74 | [config](./rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r101vd_6x_coco_from_paddle.pth)
<!-- rtdetrv2_hgnetv2_l | COCO | 640 | 52.9 | 71.5 | 32 | 114 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_hgnetv2_l_6x_coco_from_paddle.pth) 
rtdetrv2_hgnetv2_x | COCO | 640 | 54.7 | 72.9 | 67 | 74 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_hgnetv2_x_6x_coco_from_paddle.pth) 
rtdetrv2_hgnetv2_h | COCO | 640 | 56.3 | 74.8 | 123 | 40 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_hgnetv2_h_6x_coco_from_paddle.pth) 
rtdetrv2_18vd | COCO+Objects365 | 640 | 49.0 | 66.5 | 20 | 217 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_5x_coco_objects365_from_paddle.pth)
rtdetrv2_r50vd | COCO+Objects365 | 640 | 55.2 | 73.4 | 42 | 108 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_2x_coco_objects365_from_paddle.pth)
rtdetrv2_r101vd | COCO+Objects365 | 640 | 56.2 | 74.5 | 76 | 74 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r101vd_2x_coco_objects365_from_paddle.pth)
 -->

**Notes:**
- `AP` is evaluated on *MSCOCO val2017* dataset.
- `FPS` is evaluated on a single T4 GPU with $batch\\_size = 1$, $fp16$, and $TensorRT>=8.5.1$.
- `COCO + Objects365` in the table means finetuned model on `COCO` using pretrained weights trained on `Objects365`.



### Models of discrete sampling

| Model | Sampling Method | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | config| checkpoint 
| :---: | :---: | :---: | :---: | :---: | :---: |
**RT-DETRv2-S_dsp** | discrete_sampling | 47.4 | 64.8 <font color=red>(-0.1)</font> | [config](./rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_dsp_3x_coco.yml) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_dsp_3x_coco.pth)
**RT-DETRv2-M**<sup>*</sup>**_dsp** | discrete_sampling | 49.2 | 67.1 <font color=red>(-0.4)</font> | [config](./rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r34vd_dsp_1x_coco.yml) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rrtdetrv2_r34vd_dsp_1x_coco.pth)
**RT-DETRv2-M_dsp** | discrete_sampling | 51.4 | 69.7 <font color=red>(-0.2)</font> | [config](./rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r50vd_m_dsp_3x_coco.yml) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_m_dsp_3x_coco.pth)
**RT-DETRv2-L_dsp** | discrete_sampling | 52.9 | 71.3 <font color=red>(-0.3)</font> |[config](./rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r50vd_dsp_1x_coco.yml)| [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_dsp_1x_coco.pth)


<!-- **rtdetrv2_r18vd_dsp1** | discrete_sampling | 21600 | 46.3 | 63.9 | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_dsp1_1x_coco.pth) -->

<!-- rtdetrv2_r18vd_dsp1 | discrete_sampling | 21600 | 45.5 | 63.0 | 4.34 | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_dsp1_120e_coco.pth) -->
<!-- 4.3 -->

**Notes:**
- The impact on inference speed is related to specific device and software.
- `*_dsp*` is the model inherit `*_sp*` model's knowledge and adapt to `discrete_sampling` strategy. **You can use TensorRT 8.4 (or even older versions) to inference for these models**
<!-- - `grid_sampling` use `grid_sample` to sample attention map, `discrete_sampling` use `index_select` method to sample attention map.  -->


### Ablation on sampling points

<!-- Flexible samping strategy in cross attenstion layer for devices that do **not** optimize (or not support) `grid_sampling` well. You can choose models based on specific scenarios and the trade-off between speed and accuracy. -->

| Model | Sampling Method | #Points | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | checkpoint 
| :---: | :---: | :---: | :---: | :---: | :---: |
**rtdetrv2_r18vd_sp1** | grid_sampling | 21,600 | 47.3 | 64.3 <font color=red>(-0.6) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_sp1_120e_coco.pth)
**rtdetrv2_r18vd_sp2** | grid_sampling | 43,200 | 47.7 | 64.7 <font color=red>(-0.2) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_sp2_120e_coco.pth)
**rtdetrv2_r18vd_sp3** | grid_sampling | 64,800 | 47.8 | 64.8 <font color=red>(-0.1) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_sp3_120e_coco.pth)
rtdetrv2_r18vd(_sp4)| grid_sampling | 86,400 | 47.9 | 64.9 | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_120e_coco.pth) 

**Notes:**
- The impact on inference speed is related to specific device and software.
- `#points` the total number of sampling points in decoder for per image inference.





## 📍 Training

- COCO-format dataset
```shell
--dataset
    |-- test
    |   |-- test1.jpg
    |   |-- test2.jpg
    |     ...
    |   `-- _annotations.coco.json
    |-- train
    |   |-- train1.jpg
    |   |-- train2.jpg
    |   |-- train3.jpg
    |   |-- train4.jpg
    |     ....
    |   `-- _annotations.coco.json
    |`-- valid
    |   |-- valid1.jpg
    |   |-- valid2.jpg
    |      ....  
    |   `-- _annotations.coco.json

```
- select the training [yml](./rtdetrv2_pytorch/configs/) and change  [coco_detection.yml](./rtdetrv2_pytorch/configs/dataset/coco_detection.yml)

``` 
-  num_classes
-  img_folder (for train_dataloader and val_dataloader)
-  ann_file   (for train_dataloader and val_dataloader)
```

- change [class names](./rtdetrv2_pytorch/configs/classes.json) 
- start training
```shell
cd RT-DETRv2
python rtdetrv2_pytorch/tools/train.py -c rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_xxxx_xxxx_coco.yml --use-amp --seed=0 &> log.txt 2>&1 &
```


- Check the training logs in the ```log.txt``` file.
- The model will be saved in the ```output_dir``` specified in the ```rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_xxxx_xxxx_coco.yml``` file.

