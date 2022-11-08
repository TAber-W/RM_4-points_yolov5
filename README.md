# RM_4-points_yolov5
# 基于yolov5-face修改的四点模型。
## 介绍
数据集类别来源于上交数据标注软件。<br>
<br>
上交格式数据集参考：https://github.com/TAber-W/RobomasterDataset<br>
<br>
数据集标签格式：类别序号+xywh+4点，长度13.<br>
<br>
运行transform_labels.py可将上交四点格式转为此模型适用的格式。<br>
## 识别效果：
https://www.bilibili.com/video/BV1cG4y187UZ/
## 训练
### mobilenet为主干网络：
    python train.py --weights yolov5s.pt --cfg models/yolov5s.yaml --data data/mobilenet_small.yaml --batch-size 16 --epochs 500
### yolov5为主干网络：
    python train.py --weights yolov5s.pt --cfg models/yolov5s.yaml --data data/widerface.yaml --batch-size 16 --epochs 500
## 验证
    python detect.py --weights best.pt --source test.jpg/.mp4--save-img(保存目录/runs/detect/下)


