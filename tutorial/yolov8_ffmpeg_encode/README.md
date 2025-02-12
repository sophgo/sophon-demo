# 简介：
本例程的处理流程是：ffmpeg decode + bmcv preprocess + bmrt yolov8 inference + cpu postprocess + bmcv rectangle + ffmpeg encode，支持在BM1684X/BM1688/CV186X上测试，如果用户需要实现ffmpeg编解码、ffmpeg和bmcv格式转换等逻辑，可以参考本例程。

# 目录结构说明：

```bash
├── CMakeLists.txt
├── coco.names
├── ff_decode # ff_decode依赖，不同于别的例程的ff_decode，这里设置了解码器输出格式为压缩格式，并且优先把bm_image的内存放到heap2上。
├── ff_encode # ff_encode依赖，来自sophon-mw-soc_0.10.0_aarch64/opt/sophon/sophon-sample_0.10.0/samples/ff_bmcv_transcode/ff_video_encode
├── main.cpp  # 主程序，包含主要调用逻辑
├── README.md
├── utils.hpp # timestamp依赖，用于计时
├── yolov8_det.cpp # yolov8_det封装，来自sophon-demo/sample/YOLOv8_plus_det
├── yolov8_det.hpp
```

# 获取测试视频和模型：

本例程的测试视频和模型均来自[YOLOv8_plus_det](../../sample/YOLOv8_plus_det/README.md#31-数据准备)
```bash
pip3 install dfss --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/common/test_car_person_1080P.mp4

python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv8_plus_det/BM1684X.tar.gz #bm1684x
tar xvf BM1684X.tar.gz && rm BM1684X.tar.gz #bm1684x

python3 -m dfss --url=open@sophgo.com:sophon-demo/YOLOv8_plus_det/BM1688.tar.gz #bm1688/cv168ah
tar xvf BM1688.tar.gz && rm BM1688.tar.gz #bm1688/cv168ah

```


# 编译运行方法：

编译方法同[yolov8_bmcv](../../sample/YOLOv8_plus_det/cpp/README.md)。

运行方法：

首先在目标推流服务器上运行rtsp服务器，准备接收流。
然后在搭载BM1684X/BM1688/CV186X设备的机器上运行如下命令：
```bash
./yolov8_bmcv.soc --output=rtsp://172.21.80.56:8554/test --bmodel=BM1684X/yolov8s_int8_1b.bmodel --input=test_car_person_1080P.mp4
```