import logging

LOG_FORMAT = "%(levelname)s %(asctime)s.%(msecs)d %(thread)d %(filename)s:%(lineno)d] %(funcName)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S"

logging.basicConfig(level=logging.DEBUG,
                    format=LOG_FORMAT, datefmt=DATE_FORMAT)

# # 建立一个filehandler来把日志记录在文件里，级别为debug以上
# fh = logging.FileHandler("infer.log")
# fh.setLevel(logging.DEBUG)
# 建立一个streamhandler来把日志打在CMD窗口上，级别为debug以上
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)


logger = logging.getLogger()