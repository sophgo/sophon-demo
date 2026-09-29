"""torchvision 轻量 stub。

GLC 源码里 `from torchvision import models` 是无用 import，而当前环境的 torch 2.14
没有可用的匹配 torchvision 轮子。此 stub 只提供 GLC 代码路径真正用到的最小接口。
"""
from . import models  # noqa: F401
from . import transforms  # noqa: F401

__version__ = "0.0.0-stub"
