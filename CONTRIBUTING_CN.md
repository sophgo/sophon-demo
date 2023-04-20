**介绍**

Sophon Demo，欢迎各位开发者

**贡献要求**

开发者提交的模型包括源码、README、参考模型、测试用例，并遵循以下标准

**一、源码**

1、推理请使用C++或python代码，符合第四部分编码规范

2、请将各模块的内容提交到相应的代码目录内

3、从其他开源迁移的代码，请增加License声明

**二、License规则**

1、若引用参考源项目的文件，且源项目已包含License文件则必须拷贝引用，否则在模型顶层目录下添加使用的深度学习框架或其他组件的必要的License

2、每个复制或修改于源项目的文件，都需要在源文件开头附上源项目的License头部声明，并在其下追加新增完整算能License声明

```
# ...... 源项目的License头部声明 ......
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
```

2、每个自己新编写的文件，都需要在源文件开头添加算能License声明

```
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
```

> 关于License声明时间，应注意：
>
> 1. 2021年新建的文件，应该是Copyright 2021 Sophgo Technologies Co.
> 2. Python文件的行注释符号为`#`，C/CPP文件中的注释符号为`//`

**三、README**

README用于指导用户理解和测试样例，要包含如下内容：

1. 关于例程功能的说明；
2. 编译或测试例程所需的环境的配置方法；
3. 例程的编译或者测试方法；

针对模型的参考样例，要包含如下内容：

1. 模型的来源及简介；
2. 相关模型和数据的下载方式；
3. FP32 BModel、FP16(BM1684X)及INT8 BModel（1batch及4batch）的生成脚本；
4. 模型推理的步骤和源码（Python、C++）；
5. 模型的性能测试方法和结果；
6. 模型的精度测试方法和结果。

- 关键要求：

1. 模型的出处、对数据的要求、免责声明等，开源代码文件修改需要增加版权说明；

2. 模型转换得到的模型对输入数据的要求；

3. 环境变量设置，依赖的第三方软件包和库，以及安装方法；

4. 精度和性能达成要求：尽量达到原始模型水平；

5. 原始模型及转换后FP32和INT8 BModel的下载地址。


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