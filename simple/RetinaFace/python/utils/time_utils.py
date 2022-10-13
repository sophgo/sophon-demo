# -*- coding: utf-8 -*-
"""
Function: Time Utils
Author: jin.zhang
Create Time: 2022/01/19 13:48
"""

import time
import numpy as np
import collections
from loguru import logger
from prettytable import PrettyTable

from utils.cls_utils import singleton

class Record:
    def __init__(self):
        self.set       = False
        self.mark_time = 0.0
        self.laps      = []

@singleton
class TimeStamp:
    # __lock = False
    __records = collections.OrderedDict()
    
    def mark(self, marker):
        
        if marker in self.__records.keys():
            if self.__records[marker].set:
                self.__records[marker].marktime = time.time()
                self.__records[marker].set = bool(1 - self.__records[marker].set)
            else:
                self.__records[marker].laps.append(round(time.time() - self.__records[marker].marktime, 3))
                self.__records[marker].marktime = time.time()
                self.__records[marker].set = bool(1 - self.__records[marker].set)
        else: # 没有则创建一个
            self.__records[marker] = Record()
            now = time.time()
            self.__records[marker].marktime = time.time()
            self.__records[marker].set = False

    def print(self):

        table = PrettyTable(["函数名称","运行次数","平均耗时(秒)","最大耗时(秒)","最小耗时(秒)"])
        table.title = 'Running Time Cost Summary'

        for key, value in self.__records.items():
            table.add_row([key, len(value.laps), np.mean(value.laps), np.max(value.laps), np.min(value.laps)])
            # print("{}: {}".format(key, value.laps))

        print(table)


def timeit(func):
    """ 计时warp函数，可作为待测试运行时间的函数的装饰器
    :param func: 需要传入的函数
    :return:
    """
    time_stamp = TimeStamp()
 
    def _warp(*args, **kwargs):
        """
        :param args: func需要的位置参数
        :param kwargs: func需要的关键字参数
        :return: 函数的执行结果
        """
        start_time = time.time()
        logger.trace(" >>>>>>>>> {} start >>>>>>>>> ".format(func.__name__))
        time_stamp.mark(func.__name__)
        result = func(*args, **kwargs)
        time_stamp.mark(func.__name__)
        logger.trace(" <<<<<<<<< {} end <<<<<<<<< ".format(func.__name__))
        elastic_time = time.time() - start_time
        logger.trace("Execution time of function '%s' : %.6fs\n" % (
            func.__name__, elastic_time))
        return result
 
    return _warp