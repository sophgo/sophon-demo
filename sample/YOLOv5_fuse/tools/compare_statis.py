#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import re
import argparse
import math
import os
import sys
import multiprocessing

baseline = """
| 测试平台 | 测试程序         | 测试模型                                  | decode_time | preprocess_time | inference_time | postprocess_time |
| -------- | ---------------- | ----------------------------------------- | ----------- | --------------- | -------------- | ---------------- |
|   SE7-32    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp32_1b.bmodel|      13.95      |      1.01       |      37.94      |      2.86       |
|   SE7-32    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp16_1b.bmodel|      13.92      |      0.95       |      23.69      |      2.53       |
|   SE7-32    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_1b.bmodel|      13.97      |      1.03       |      20.76      |      2.72       |
|   SE7-32    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_4b.bmodel|      13.95      |      1.03       |      19.56      |      2.74       |
|   SE7-32    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp32_1b.bmodel|      3.08       |      1.40       |      21.02      |      3.08       |
|   SE7-32    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp16_1b.bmodel|      3.05       |      1.38       |      6.63       |      3.12       |
|   SE7-32    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_1b.bmodel|      3.04       |      1.37       |      3.62       |      2.91       |
|   SE7-32    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_4b.bmodel|      2.83       |      1.23       |      2.91       |      2.95       |
|   SE7-32    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      4.45       |      0.31       |      20.23      |      0.03       |
|   SE7-32    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      4.58       |      0.32       |      6.00       |      0.02       |
|   SE7-32    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      4.52       |      0.31       |      3.01       |      0.02       |
|   SE7-32    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      4.38       |      0.30       |      2.75       |      0.01       |
|   SE7-32    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      2.85       |      0.85       |      20.41      |      0.02       |
|   SE7-32    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      2.89       |      0.86       |      6.17       |      0.02       |
|   SE7-32    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      2.89       |      0.85       |      3.20       |      0.02       |
|   SE7-32    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      2.76       |      0.59       |      2.80       |      0.01       |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp32_1b.bmodel|      31.96      |      1.21       |     127.52      |      3.66       |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp16_1b.bmodel|      20.72      |      1.19       |      56.29      |      3.79       |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_1b.bmodel|      19.25      |      1.18       |      34.98      |      3.69       |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_4b.bmodel|      19.29      |      1.15       |      33.44      |      3.98       |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp32_1b.bmodel|      4.28       |      3.40       |     103.85      |      3.84       |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp16_1b.bmodel|      4.22       |      3.38       |      32.57      |      3.90       |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_1b.bmodel|      4.24       |      3.39       |      11.33      |      4.43       |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_4b.bmodel|      4.05       |      3.20       |      10.28      |      3.63       |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      6.04       |      1.05       |     102.85      |      0.04       |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      5.97       |      1.05       |      31.69      |      0.04       |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      5.98       |      1.05       |      10.44      |      0.04       |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      5.80       |      1.03       |      10.05      |      0.01       |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      4.01       |      1.80       |     103.12      |      0.04       |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      4.00       |      1.78       |      31.91      |      0.03       |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      3.95       |      1.79       |      10.63      |      0.03       |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      3.73       |      1.28       |      10.11      |      0.01       |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp32_1b_2core.bmodel|      19.48      |      1.21       |      82.17      |      3.74       |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp16_1b_2core.bmodel|      19.27      |      1.17       |      44.72      |      3.67       |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_1b_2core.bmodel|      19.35      |      1.22       |      34.27      |      3.68       |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_4b_2core.bmodel|      19.23      |      1.10       |      31.61      |      3.97       |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp32_1b_2core.bmodel|      4.23       |      3.37       |      58.55      |      3.95       |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp16_1b_2core.bmodel|      4.21       |      3.37       |      21.03      |      3.79       |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_1b_2core.bmodel|      4.22       |      3.38       |      10.54      |      4.45       |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_4b_2core.bmodel|      4.06       |      3.20       |      8.50       |      3.61       |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp32_1b_2core.bmodel|      6.11       |      1.05       |      57.61      |      0.04       |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp16_1b_2core.bmodel|      5.94       |      1.05       |      20.15      |      0.04       |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_1b_2core.bmodel|      6.01       |      1.06       |      9.66       |      0.04       |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_4b_2core.bmodel|      5.77       |      1.03       |      8.27       |      0.01       |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp32_1b_2core.bmodel|      3.98       |      1.78       |      57.83      |      0.04       |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp16_1b_2core.bmodel|      3.93       |      1.78       |      20.36      |      0.03       |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_1b_2core.bmodel|      3.95       |      1.78       |      9.86       |      0.03       |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_4b_2core.bmodel|      3.74       |      1.28       |      8.33       |      0.01       |
|    SE9-8    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp32_1b.bmodel|      10.41      |      1.31       |     127.92      |      3.60       |
|    SE9-8    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp16_1b.bmodel|      10.33      |      1.22       |      56.81      |      3.71       |
|    SE9-8    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_1b.bmodel|      9.54       |      1.27       |      35.42      |      4.35       |
|    SE9-8    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_4b.bmodel|      9.48       |      1.17       |      33.82      |      4.12       |
|    SE9-8    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp32_1b.bmodel|      4.14       |      3.22       |     104.38      |      4.37       |
|    SE9-8    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp16_1b.bmodel|      4.11       |      3.20       |      33.06      |      4.01       |
|    SE9-8    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_1b.bmodel|      4.08       |      3.21       |      11.75      |      4.26       |
|    SE9-8    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_4b.bmodel|      3.94       |      3.04       |      10.71      |      4.23       |
|    SE9-8    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      5.93       |      0.99       |     103.33      |      0.04       |
|    SE9-8    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      5.80       |      0.99       |      32.11      |      0.04       |
|    SE9-8    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      5.82       |      0.99       |      10.86      |      0.03       |
|    SE9-8    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      5.63       |      0.97       |      10.47      |      0.02       |
|    SE9-8    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      3.82       |      1.83       |     103.61      |      0.05       |
|    SE9-8    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      3.78       |      1.73       |      32.34      |      0.03       |
|    SE9-8    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      3.85       |      1.76       |      11.08      |      0.03       |
|    SE9-8    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      3.60       |      1.25       |      10.53      |      0.01       |

"""
basetime = """
| 测试平台 | 测试程序         | 测试模型                                  | yolov5_fuse | yolov5_opt | yolov5 |
| -------- | ---------------- | ----------------------------------------- | --------------- | -------------- | ---------- |
|   SE7-32    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp32_1b.bmodel|      41.81      |      60.74      |     170.07      |
|   SE7-32    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp16_1b.bmodel|      27.17      |      47.32      |     155.55      |
|   SE7-32    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_1b.bmodel|      24.51      |      44.74      |     151.53      |
|   SE7-32    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_4b.bmodel|      23.33      |      30.31      |     146.52      |
|   SE7-32    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp32_1b.bmodel|      25.50      |      29.60      |     135.20      |
|   SE7-32    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp16_1b.bmodel|      11.13      |      29.60      |     120.09      |
|   SE7-32    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_1b.bmodel|      7.90       |      13.63      |     117.15      |
|   SE7-32    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_4b.bmodel|      7.09       |      12.71      |     120.18      |
|   SE7-32    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      20.57      |      21.87      |      37.28      |
|   SE7-32    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      6.34       |      8.47       |      24.06      |
|   SE7-32    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      3.34       |      5.60       |      19.16      |
|   SE7-32    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      3.06       |      5.17       |      19.76      |
|   SE7-32    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|      21.28      |      24.08      |      37.74      |
|   SE7-32    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      7.05       |      10.68      |      25.13      |
|   SE7-32    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      4.07       |      7.89       |      21.22      |
|   SE7-32    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      3.40       |      6.56       |      20.26      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp32_1b.bmodel|     132.39      |        -        |     266.86      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp16_1b.bmodel|      61.27      |        -        |     231.47      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_1b.bmodel|      39.85      |        -        |     209.08      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_4b.bmodel|      38.57      |        -        |     202.89      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp32_1b.bmodel|     111.09      |        -        |     258.31      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp16_1b.bmodel|      39.85      |        -        |     187.56      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_1b.bmodel|      19.15      |        -        |     166.03      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_4b.bmodel|      17.11      |        -        |     169.91      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|     103.94      |        -        |     124.70      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      32.78      |        -        |      54.17      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      11.53      |        -        |      32.42      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      11.09      |        -        |      31.90      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|     104.96      |        -        |     128.97      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      33.72      |        -        |      57.18      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      12.45      |        -        |      35.42      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      11.40      |        -        |      34.72      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp32_1b_2core.bmodel|      87.12      |        -        |     264.83      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp16_1b_2core.bmodel|      49.56      |        -        |     219.21      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_1b_2core.bmodel|      39.17      |        -        |     205.57      |
|   SE9-16    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_4b_2core.bmodel|      36.68      |        -        |     200.90      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp32_1b_2core.bmodel|      65.87      |        -        |     221.47      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp16_1b_2core.bmodel|      28.19      |        -        |     176.49      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_1b_2core.bmodel|      18.37      |        -        |     164.46      |
|   SE9-16    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_4b_2core.bmodel|      15.31      |        -        |     168.98      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp32_1b_2core.bmodel|      58.70      |        -        |      87.48      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp16_1b_2core.bmodel|      21.24      |        -        |      42.30      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_1b_2core.bmodel|      10.76      |        -        |      30.57      |
|   SE9-16    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_4b_2core.bmodel|      9.31       |        -        |      28.90      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp32_1b_2core.bmodel|      59.65      |        -        |      90.71      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp16_1b_2core.bmodel|      22.17      |        -        |      45.48      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_1b_2core.bmodel|      11.67      |        -        |      33.74      |
|   SE9-16    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_4b_2core.bmodel|      9.62       |        -        |      31.87      |
|    SE9-8    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp32_1b.bmodel|     132.83      |        -        |     266.86      |
|    SE9-8    | yolov5_opencv.py  |yolov5s_v6.1_3output_fp16_1b.bmodel|      61.74      |        -        |     231.47      |
|    SE9-8    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_1b.bmodel|      41.04      |        -        |     209.08      |
|    SE9-8    | yolov5_opencv.py  |yolov5s_v6.1_3output_int8_4b.bmodel|      39.11      |        -        |     202.89      |
|    SE9-8    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp32_1b.bmodel|     111.97      |        -        |     258.31      |
|    SE9-8    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_fp16_1b.bmodel|      40.27      |        -        |     187.56      |
|    SE9-8    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_1b.bmodel|      19.22      |        -        |     166.03      |
|    SE9-8    |  yolov5_bmcv.py   |yolov5s_v6.1_3output_int8_4b.bmodel|      17.98      |        -        |     169.91      |
|    SE9-8    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|     104.36      |        -        |     124.70      |
|    SE9-8    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      33.14      |        -        |      54.17      |
|    SE9-8    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      11.88      |        -        |      32.42      |
|    SE9-8    |  yolov5_bmcv.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      11.46      |        -        |      31.90      |
|    SE9-8    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp32_1b.bmodel|     105.49      |        -        |     128.97      |
|    SE9-8    |  yolov5_sail.soc  |yolov5s_v6.1_3output_fp16_1b.bmodel|      34.10      |        -        |      57.18      |
|    SE9-8    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_1b.bmodel|      12.87      |        -        |      35.42      |
|    SE9-8    |  yolov5_sail.soc  |yolov5s_v6.1_3output_int8_4b.bmodel|      11.79      |        -        |      34.72      |
"""
table_data = {
    "platform": [],
    "program": [],
    "bmodel": [],
    "decode": [],
    "preprocess": [],
    "inference": [],
    "postprocess": []
}
table_time = {
    "platform": [],
    "program": [],
    "bmodel": [],
    "yolov5_fuse": [],
    "yolov5_opt": [],
    "yolov5": []
}

patterns_cpp = {
    'decode': re.compile(r'\[.*decode time.*\]  loops:.*avg: ([\d.]+) ms'),
    'preprocess': re.compile(r'\[.*preprocess.*\]  loops:.*avg: ([\d.]+) ms'),
    'inference': re.compile(r'\[.*inference.*\]  loops:.*avg: ([\d.]+) ms'),
    'postprocess': re.compile(r'\[.*postprocess.*\]  loops:.*avg: ([\d.]+) ms'),
}

patterns_python = {
    'decode': re.compile(r'decode_time\(ms\): ([\d.]+)'),
    'preprocess': re.compile(r'preprocess_time\(ms\): ([\d.]+)'),
    'inference': re.compile(r'inference_time\(ms\): ([\d.]+)'),
    'postprocess': re.compile(r'postprocess_time\(ms\): ([\d.]+)'),
}

def extract_times(text, patterns):
    results = {}
    for key, pattern in patterns.items():
        match = pattern.search(text)
        if match:
            results[key] = round(float(match.group(1)),2)
    return results


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--target', type=str, default='BM1684X', help='path of label json')
    parser.add_argument('--platform', type=str, default='soc', help='path of result json')
    parser.add_argument('--bmodel', type=str, default='yolov5s_v6.1_3output_fp32_1b.bmodel')
    parser.add_argument('--program', type=str, default='yolov5_bmcv.py')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--input', type=str, default='../log/bmcv_yolov5s_v6.1_3output_fp32_1b.bmodel_python_test.log')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    compare_pass = True
    cnt_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(cnt_file_path)
    args = argsparser()

    benchmark_path = current_dir + "/benchmark.txt"
    benchmark_time_path = current_dir + "/benchmark_time.txt"
        
        
    for line in baseline.strip().split("\n")[2:]:
        match = re.search(r'\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|', line)
        if match:
            table_data["platform"].append(match.group(1))
            table_data["program"].append(match.group(2))
            table_data["bmodel"].append(match.group(3))
            table_data["decode"].append(float(match.group(4)))
            table_data["preprocess"].append(float(match.group(5)))
            table_data["inference"].append(float(match.group(6)))
            table_data["postprocess"].append(float(match.group(7)))
    
    for line in basetime.strip().split("\n")[2:]:
        match = re.search(r'\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|', line)
        if match:
            table_time["platform"].append(match.group(1))
            table_time["program"].append(match.group(2))
            table_time["bmodel"].append(match.group(3))
            table_time["yolov5_fuse"].append(float(match.group(4)))
            table_time["yolov5_opt"].append((match.group(5)))
            table_time["yolov5"].append((match.group(6)))

    if args.platform == "soc":
        if args.target == "BM1684X":
            platform = "SE7-32"
        elif args.target == "BM1684":
            platform = "SE5-16"
        elif args.target == "BM1688":
            platform = "SE9-16"
            if multiprocessing.cpu_count() == 6:
                platform = "SE9-8"
        elif args.target == "CV186X":
            platform = "SE9-8"
    else:
        platform = args.target + " SoC" if args.platform == "soc" else args.target + " PCIe"
    min_width = 17
    
    if not os.path.exists(benchmark_path):
        with open(benchmark_path, "w") as f:
            benchmark_str = "|{:^13}|{:^19}|{:^35}|{:^{width}}|{:^{width}}|{:^{width}}|{:^{width}}|\n".format(
           "platform", "program", "bmodel", "decode_time", "preprocess_time", "inference_time", "postprocess_time", width=min_width)
            f.write(benchmark_str)

    if not os.path.exists(benchmark_time_path):
        with open(benchmark_time_path, "w") as f:
            benchmark_time_str = "|{:^13}|{:^19}|{:^35}|{:^{width}}|{:^{width}}|{:^{width}}|\n".format(
           "platform", "program", "bmodel", "yolov5_fuse", "yolov5_opt", "yolov5", width=min_width)
            f.write(benchmark_str)      

    with open(args.input, "r") as f:
        data = f.read()
    if args.language == "python":    
        extracted_data = extract_times(data, patterns_python)
    elif args.language == "cpp":
        extracted_data = extract_times(data, patterns_cpp)
    else:
        print("unsupport code language")
    match_index = -1
    for i in range(0, len(table_data["platform"])):
        if platform == table_data["platform"][i] and args.program == table_data["program"][i] and args.bmodel == table_data["bmodel"][i]:
            match_index = i
            break
    baseline_data = {}
    if match_index == -1:
        print("Unmatched case.")
    else:
        baseline_data["decode"] = table_data["decode"][match_index]
        baseline_data["preprocess"] = table_data["preprocess"][match_index]
        baseline_data["inference"] = table_data["inference"][match_index]
        baseline_data["postprocess"] = table_data["postprocess"][match_index]

    match_index = -1
    for i in range(0, len(table_time["platform"])):
        if platform == table_time["platform"][i] and args.program == table_time["program"][i] and args.bmodel == table_time["bmodel"][i]:
            match_index = i
            break
    baseline_time_data = {}
    if match_index == -1:
        print("Unmatched case.")
    else:
        baseline_time_data["yolov5_fuse"] = table_time["yolov5_fuse"][match_index]
        baseline_time_data["yolov5_opt"] = table_time["yolov5_opt"][match_index]
        baseline_time_data["yolov5"] = table_time["yolov5"][match_index]

    for key, statis in baseline_data.items():
        if abs(statis - extracted_data[key]) / statis > 0.2:
            print("{:} time, diff ratio > 0.2".format(key))
            print("Baseline is:", statis)
            print("Now is: ", extracted_data[key])
            compare_pass = False
        
    benchmark_str = "|{:^13}|{:^19}|{:^35}|{decode:^{width}.2f}|{preprocess:^{width}.2f}|{inference:^{width}.2f}|{postprocess:^{width}.2f}|\n".format(
                     platform, args.program, args.bmodel, **extracted_data, width=min_width)
    
    with open(benchmark_path, "a") as f:
        f.write(benchmark_str)

    fuse_time = float(extracted_data["preprocess"])+float(extracted_data["inference"])+float(extracted_data["postprocess"])   
    if abs(fuse_time - float(baseline_time_data["yolov5_fuse"])) / fuse_time > 0.2:
            print("Baseline is:", float(baseline_time_data["yolov5_fuse"]))
            print("Now is: ", fuse_time)
            compare_pass = False

    benchmark_time_str = "|{:^13}|{:^19}|{:^35}|{:^{width}.2f}|{:^17}|{:^17}|\n".format(
                     platform, args.program, args.bmodel, fuse_time, baseline_time_data["yolov5_opt"], baseline_time_data["yolov5"], width=min_width)
    
    with open(benchmark_time_path, "a") as f:
        f.write(benchmark_time_str)

    if compare_pass == False:
        sys.exit(1)
        
