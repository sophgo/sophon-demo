[简体中文](./CONTRIBUTING_CN.md) | [English](./CONTRIBUTING_EN.md)

**Introduction**

Sophon Demo, developers are welcome!

**Contribution Requirements**

Developers submit models including source code, README, reference models, test cases, and follow the following criteria

**I. Source code**

1、Please use C++ or python code for inferencing, in line with the fourth part of the programming specifications

2、Please submit the content of each module to the corresponding code directory

3、The code migrated from another open source, please add the License statement

**II. License rules**

1、if you refer to the reference source project files, and the source project has included License files, you must copy the reference, otherwise, add the necessary License for the deep learning framework or other components used in the model top-level directory

2、each file copied or modified in the source project, need to attach the source project's License header statement at the beginning of the source file and add a new full Sophon License statement under it

```
# ...... License header declaration of the source project ......
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
```

2、 For Each new file you write, you need to add a License statement at the beginning of the source file

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

> Concerning License declaration time, it should be noted that:
>
> 1. for new files created in 2021, it should be Copyright 2021 Sophgo Technologies Co.
> 2. The line comment notation in Python files is `#` and in C/CPP files is `//`

**III. README**

The README is used to guide the user in understanding and testing the sample and is to contain the following.

1. a description of the function of the sample;
2. the configuration method of the environment required to compile or test the sample;
3. the method of compiling or testing the sample.

Reference examples for the model, including the following.

1. the source and a brief description of the model;
2. how to download the relevant models and data;
3. the generation scripts for FP32 BModel, FP16 (BM1684X) and INT8 BModel (1batch and 4batch);
4. the steps and source code for model inference (Python, C++);
5. performance testing methods and results of the model;
6. the accuracy testing methods and results of the model.

- Key requirements.

1. the provenance of the model, requirements for data, disclaimers, etc. The modification of the open-source code file needs to add a copyright statement;

2. the requirements of the input data for the model obtained by model conversion;

3. environment variable settings, dependent third-party packages and libraries, and installation methods;

4. accuracy and performance requirements: try to reach the level of the original model;

5. the download address of the original model, converted FP32 BModel and INT8 BModel.

**IV. Programming Specifications**

- Specification standards

1. C++ code follows google programming specifications: [Google C++ Style Guide](https://zh-google-styleguide.readthedocs.io/en/latest/google-cpp-styleguide)([Google C++ Coding Guidelines](https://google.github.io/styleguide/cppguide.html)); unit tests follow the specification: [Googletest Primer](https://google.github.io/googletest/primer.html)
2. Python code follows the PEP8 specification: [Python PEP 8 Coding Style](https://www.python.org/dev/peps/pep-0008/); unit tests follow the specification: [pytest](https://docs.pytest.org/en/stable/)
3. Shell scripts follow google programming specifications: [Google Shell Style Guide](https://zh-google-styleguide.readthedocs.io/en/latest/google-shell-styleguide/contents/) ([ Google Shell Style Guide](https://google.github.io/styleguide/shellguide.html))
4. git commit messages follow the specification: [Angular Specification](https://docs.google.com/document/d/1QrDFcIiPjSLDn3EL15IJygNPiHORgU1_OOAqWjiDU5Y/edit#)

- Specification additions

1. use cout instead of printf in C++ code;
2. use smart pointers for memory management as much as possible;
3. control third-party library dependencies, and if third-party dependencies are introduced, instructions for installing and using third-party dependencies need to be provided;
4. use English comments as a rule, with a comment rate of 30% - 40%, and encourage self-commentation;
5. the function header must be commented, stating the function role, input parameters, output parameters;
6. unified error code, through the error code can confirm that the branch returns an error;
7. prohibit the appearance of printing a bunch of unaffected error-level logs.