# RM_4-points_yolov5
# 基于yolov5-face修改的四点模型。
## 更新日期2023年3月20日

## 已知BUG
    detect.py 检测源为视频时会闪退。（正在修改）
    
    detect.py 检测大小只能为640。（正在修改）
    
    建议:采用trt推理后自己绘图

## 介绍
    平台环境
    
    操作系统：Windows 10
    
    CUDA：11.7

    显卡：NVIDIA Tesla P100-16GB
    
    Pytorch：1.11
    
    ONNX、ONNXRuntime：1.12.0、1.13.1
    
    TensorRT：8.5.1.7 GA
数据集类别来源于上交数据标注软件。<br>
<br>
上交格式数据集参考：https://github.com/TAber-W/RobomasterDataset<br>
<br>
数据集标签格式：类别序号+xywh+4点，长度13.<br>
<br>
运行transform_labels.py可将上交四点格式转为此模型适用的格式。<br>
<br>
由于上交格式是有超出边界的坐标，在此模型下会产生警告，将超出范围的点收回到最大值，不影响识别，可忽略。<br>
<br>
交流方式：qq 852707293

## 环境配置（步骤）
### 安装pytorch
![image](https://github.com/TAber-W/RM_4-points_yolov5/blob/master/images/pytorch.png)
### 安装CUDA，CUDNN
下图为建议版本
<br>
![image](https://github.com/TAber-W/RM_4-points_yolov5/blob/master/images/cuda.png)
### 安装工具包
    pip install -r requirements.txt

## 识别效果：
https://www.bilibili.com/video/BV1cG4y187UZ/ <br>
![image](https://github.com/TAber-W/RM_4-points_yolov5/blob/master/test.jpg)
## 权重文件（weights目录）
更新在了Release <br>
~~目录下的RM-NET.pt是训练了101 epochs ,batch-size=16 ，Map0.5 为 0.63。~~<br>
~~并且转换成了其他格式、onnx、TensorRT。~~
<br>
数据集采用的西南大学——GKD战队的四点数据集。
## 训练
### mobilenet为主干网络：
    python train.py --weights yolov5s.pt --cfg models/mobilenet_small.yaml --data data/widerface.yaml --batch-size 16 --epochs 500
### yolov5为主干网络：
    python train.py --weights yolov5s.pt --cfg models/yolov5s.yaml --data data/widerface.yaml --batch-size 16 --epochs 500
## 验证(视频暂时有BUG)
    python detect.py --weights best.pt --source test.jpg --save-img(保存目录/runs/detect/下)
## 导出onnx
    python export.py --weights best.pt --img_size 640 --batch_size 1
## TensorRT部署
<b>提供了python版本的推理文件（包含了预处理和后处理，直接输入图片即可）</b><br>
<b>在测试平台下,完整推理速度达到平均40FPS，可以删除预处理，保证输入永远为640x640，</b><br>
<b>或者部署C++，训练时减小输入大小等方式来提升速度！</b>
### 1、修改配置
    打开weights/trt_infer.py 修改img_path和trt_path
### 2、运行
    python trt_infer.py
![image](https://github.com/TAber-W/RM_4-points_yolov5/blob/master/infer.jpg)
## Todo
    修改为纯四点 🚀
   
    修改边界限制 🚀
    
    修改网络结构 🐌
                      
    基于mobilevit主干替换 🚀
## 开源许可
本开源项目请遵守GNU AGPL3.0 License许可认证。


