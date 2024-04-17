# PP-OCR流程解析
这里只考虑det+rec（检测+识别）模型，跳过cls方向分类模型（这个模型实际不常用，作用不大）。
PP-OCR的标准流程可以简单概括为：解码->det前处理->det推理->det后处理->根据得到的框去裁剪图片->rec前处理->rec推理->rec后处理。
这里解释下各个部分都在代码的哪些位置，并解释一些关键代码的含义：

## 1.cpp/ppocr_bmcv
### 1.1 解码

在`src/main.cpp`中，对应这些代码:
```cpp
    bm_image bmimg;
#if USE_OPENCV_DECODE        
    cv::Mat cvmat = cv::imread(img_file, IMREAD_COLOR, dev_id);
    batch_cvmats.push_back(cvmat);//so that cvmat will not be released.
    cv::bmcv::toBMI(cvmat, &bmimg);
#else    
    picDec(h, img_file.c_str(), bmimg);
#endif
```
这里提供两种解码方式，cv::imread和picDec。他们都是使用硬件解码，cv::imread使用sophon-opencv，picDec使用sophon-ffmpeg。
解码得到的帧会放到`batch_imgs`中，送入到det前处理。

### 1.2 det前处理

对应`src/ppocr_det.cpp`的`PPOCR_Detector::preprocess_bmcv`这个函数。使用bmcv进行前处理，关于bmcv的接口用法，请查阅[BMCV开发指南](https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/bmcv/reference/html/index.html)。
这个部分主要的作用有：
1. 将解码得到的`bm_image`，进行缩放+填充+均值归一化处理，跟det源模型的前处理方式对齐。
2. 将完成处理之后的bm_image的device memory，设置为模型推理的输入。

### 1.3 det推理

本例程支持batch_size=1和batch_size=4的bmodel combine而成的bmodel推理，det模型加载部分对应`main.cpp`的这个地方：
```bash
    std::shared_ptr<BMNNContext> bm_ctx_det = std::make_shared<BMNNContext>(handle, bmodel_det.c_str());
    PPOCR_Detector ppocr_det(bm_ctx_det);
    CV_Assert(0 == ppocr_det.Init());
```
BMNNContext是`sophon-demo/include/bmnn_utils.h`里面的一个管理底层接口的封装，只需要传入设备handle、bmodel路径即可，它的功能是自动管理bmrt接口的内存申请与释放、获取模型信息、执行推理、获取推理输出等。具体可以看bmnn_utils.h中的实现。
推理部分主要对应`src/ppocr_det.cpp`的`PPOCR_Detector::run`函数中这段代码，核心就是`m_bmNetwork->forward();` 即前处理准备好数据之后，模型就可以进行推理：
```cpp
    int ret = 0;
    std::vector<bm_image> batch_images;
    std::vector<OCRBoxVec> batch_boxes;
    for(int i = 0; i < input_images.size(); i++){
        batch_images.push_back(input_images[i]);
        if(batch_images.size() == max_batch){
            vector<vector<int>> resize_vector = preprocess_bmcv(batch_images);
            m_ts->save("(per image)Det inference", batch_images.size());
            m_bmNetwork->forward();
            m_ts->save("(per image)Det inference", batch_images.size());
            ret = postForward(batch_images, resize_vector, batch_boxes);
            output_boxes.insert(output_boxes.end(), batch_boxes.begin(), batch_boxes.end());
            batch_images.clear();
            batch_boxes.clear();
        }
    }
    // Last incomplete batch, use single batch model stage.
    ...
```

### 1.4 det后处理

对应`src/ppocr_det.cpp`的`PPOCR_Detector::postForward`，除了获取推理输出的接口不同之外，其他流程与PP-OCR官方源码相同。

### 1.5 根据得到的框去裁剪图片

对应`src/main.cpp`的这个部分，得到了检测框之后，进行裁剪和仿射变换。
```cpp
                for (int i = 0; i < batch_imgs.size(); i++) {
                    bm_image input_bmimg_planar;
                    bm_image_create(h, batch_imgs[i].height, batch_imgs[i].width, FORMAT_BGR_PLANAR,
                                    batch_imgs[i].data_type, &input_bmimg_planar);
                    auto ret = bmcv_image_vpp_convert(h, 1, batch_imgs[i], &input_bmimg_planar);

                    bm_image_destroy(batch_imgs[i]);
                    batch_imgs[i] = input_bmimg_planar;

#if DEBUG
                    std::cout << "original image: " << batch_imgs[i].height << " " << batch_imgs[i].width << std::endl;
#endif
                    for (int j = 0; j < batch_boxes[i].size(); j++) {
                    #if DEBUG
                        batch_boxes[i][j].printInfo();
                    #endif
                        LOG_TS(&ts, "(per crop)get crop time");
                        bm_image crop_bmimg = get_rotate_crop_image(h, input_bmimg_planar, batch_boxes[i][j]);
                        LOG_TS(&ts, "(per crop)get crop time");
                        batch_crops.push_back(crop_bmimg);
                        batch_ids.push_back(std::make_pair(i, j));
                        total_crop_num += 1;
                    }
                }
```

### 1.6 rec前处理

对应`src/ppocr_rec.cpp`的`PPOCR_Rec::preprocess_bmcv`这个函数。使用bmcv进行前处理，关于bmcv的接口用法，请查阅[BMCV开发指南](https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/bmcv/reference/html/index.html)。
这个部分主要的作用有：
1. 将解码得到的`bm_image`，进行缩放+填充+均值归一化处理，跟rec源模型的前处理方式对齐。
2. 将完成处理之后的bm_image的device memory，设置为模型推理的输入。

### 1.7 rec推理

初始化部分和det的思想基本相同，均是使用BMNNContext管理推理，并做一些基本的检查。
推理部分主要对应`src/ppocr_rec.cpp`的`PPOCR_Rec::run`函数，核心就是`m_bmNetwork->forward();` 即前处理准备好数据之后，模型就可以进行推理。

由于rec使用的模型有多种batch_size和width，batch_size=4的模型比batch_size=1的模型推理效率高，为了最大化利用tpu算力，我们把多张原图的crop出来的小图统一放到一起，即`std::vector<bm_image> input_images`，这些小图每张图都对应着原图的(image_id, crop_id)，即`std::vector<std::pair<int, int>> ids`。我们按照一定的策略（即宽高比和模型哪个shape最接近），把这些小图分给bmodel的各个stage进行推理。后续做前后处理并行的pipeline，采用此策略可以大大提高吞吐量。

如果用户对推理效率的要求没那么高，那也可以只使用batch_size=1的bmodel，每获取一个crop就推理一次。

### 1.8 rec后处理

对应`src/ppocr_rec.cpp`的`PPOCR_Rec::postprocess`函数，提供了beam_search和greedy_search两种方法。