**介绍**

Sophon Demo，欢迎各位开发者

**贡献要求**

开发者提交的模型包括源码、README、参考模型、license文件、测试用例，并遵循以下标准

**一、源码**

1、离线推理请使用C++或python代码，符合第四部分编码规范

2、请将各模块的内容提交到相应的代码目录内

3、从其他开源迁移的代码，请增加License声明

**二、License规则**

1、若引用参考源项目的文件，且源项目已包含License文件则必须拷贝引用，否则在模型顶层目录下添加使用的深度学习框架或其他组件的必要的License

2、每个复制或修改于源项目的文件，都需要在源文件开头附上源项目的License头部声明，并在其下追加新增完整算能License声明

```
# ...... 源项目的License头部声明 ......
# ============================================================================
# Copyright 2022 Sophgo Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

2、每个自己新编写的文件，都需要在源文件开头添加算能License声明

```
# Copyright 2022 Sophgo Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

> 关于License声明时间，应注意：
>
> 1. 2021年新建的文件，应该是Copyright 2021 Sophgo Technologies Co.
> 2. Python文件的行注释符号为`#`，C/CPP文件中的注释符号为`//`

**三、README**

README用于指导用户理解和部署样例，要包含如下内容：

1. 关于例程功能的说明；
2. 编译或测试例程所需的环境的配置方法；
3. 例程的编译或者测试方法；

针对模型的参考样例，要包含如下内容：

1. 模型的来源及简介；
2. 训练原始模型使用的数据集及预训练好的原始模型的下载方式；
3. FP32 BModel（1batch及4batch）及INT8 BModel（1batch及4batch）的生成脚本；
4. 模型推理的步骤和源码（Python、C++），入口请封装成`.sh`、`.py`；
5. 模型的性能测试方法和结果；
6. 模型的精度测试方法和结果。

- 关键要求：

1. 模型的出处、对数据的要求、免责声明等，开源代码文件修改需要增加版权说明；

2. 模型转换得到的离线模型对输入数据的要求；

3. 环境变量设置，依赖的第三方软件包和库，以及安装方法；

4. 精度和性能达成要求：尽量达到原始模型水平；

5. 原始模型及转换后FP32和INT8 BModel的下载地址；

6. 数据集说明：

   -  关于数据集，可使用词汇：用户自行准备好数据集，可选用“XXX”，“XXX”，“XXX”，

     例如：请用户自行准备好数据集，包含训练集和验证集两部分，可选用的数据集包括ImageNet2012，CIFAR10、Flower等，包含train和val两部分。
   
   -  脚本中不允许提供链接下载数据集，如果开源脚本上存在对应的链接，请修改或者删除对应的脚本


**四、编程规范**

- 规范标准

1. C++代码遵循google编程规范：[Google C++风格指南](https://zh-google-styleguide.readthedocs.io/en/latest/google-cpp-styleguide)([Google C++ Coding Guidelines](https://google.github.io/styleguide/cppguide.html))；单元测试遵循规范： [Googletest Primer](https://google.github.io/googletest/primer.html)。
2. Python代码遵循PEP8规范：[Python PEP 8 Coding Style](https://www.python.org/dev/peps/pep-0008/)；单元测试遵循规范：[pytest](https://docs.pytest.org/en/stable/)
3. Shell脚本遵循google编程规范：[Google Shell风格指南](https://zh-google-styleguide.readthedocs.io/en/latest/google-shell-styleguide/contents/)([Google Shell Style Guide](https://google.github.io/styleguide/shellguide.html))
4. git commit信息遵循规范：[Angular规范](https://docs.google.com/document/d/1QrDFcIiPjSLDn3EL15IJygNPiHORgU1_OOAqWjiDU5Y/edit#)

- 规范补充

1. C++代码中使用cout而不是printf；
2. 内存管理尽量使用智能指针；
3. 控制第三方库依赖，如果引入第三方依赖，则需要提供第三方依赖安装和使用指导书；
4. 一律使用英文注释，注释率30%--40%，鼓励自注释；
5. 函数头必须有注释，说明函数作用，入参、出参；
6. 统一错误码，通过错误码可以确认那个分支返回错误；
7. 禁止出现打印一堆无影响的错误级别的日志。