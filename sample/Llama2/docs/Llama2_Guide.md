# Llama2常见问题

(1) 运行完后如何退出
Answer: 在`demo.cpp`中已经设置好了退出功能, 比如某一轮不想继续对话, 则在当前轮对话中输入`exit`即可退出。

(2) 在`scripts/download.sh` 中如果出现`Archive: models.zip, unzip: short read`. 您可以使用7z解压的方式对压缩包进行解压

(3) 如果推理过程中模型推理结束后依旧输入空格的情况，可能使由于模型量化过程中的精度损失导致。请等待后续我们提供更好的量化方式。

(4) 如果出现类似`undefined reference to 'fstat@GLIBC_2.33'`的问题，则说明编译cpp的环境是在比较高的GLIBC版本下实现的，而运行环境的GLIBC版本过低。建议在相同环境下编译和运行，比如都在编译模型提供的docker环境[docker环境](../README.md/#32-开发环境准备)中进行编译和运行。
