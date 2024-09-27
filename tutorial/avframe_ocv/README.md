# avframe_ocv例程

### 1.说明

实现avframe到cv::Mat的转换过程，本例程支持在1684&X的SoC和PCIE上使用，1688和CV186H暂不支持，原因是ffmpeg版本更新，一些接口发生变化，具体信息可参考[FFmpeg接口改动](https://doc.sophgo.com/bm1688_sdk-docs/v1.7/docs_latest_release/docs/BM1688_CV186AH_SophonSDK_doc/appendix/4_compatibility_doc.html#ffmpeg)

### 2.样例测试

- [C++例程](./cpp)
