* [1. 精度对齐](#1-精度对齐)
* [2. 后处理速度过慢](#2-后处理速度过慢)

## 1. 精度对齐
在python_bmcv的移植过程中如果发现精度无法对齐，请检查bmcv当中使用的插值方式是否和论文对齐（论文中使用双线性插值），而bmcv的默认插值方式为最邻近插值。
解决方式为: 在插值时使用 self.bmcv.resize(bmimg, self.net_w, self.net_h,  sail.bmcv_resize_algorithm.BMCV_INTER_LINEAR)

## 2. 后处理速度过慢
在运行时如果觉得速度达不到需求，大概率是因为推理所使用的conf_thresh设置的过低，从而在nms过程中会处理更多的框(proposals),可以尝试提高conf_thresh的阈值（系统默认是0.15），如提高到0.5.