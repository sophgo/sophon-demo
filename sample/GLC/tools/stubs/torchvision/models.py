"""占位：GLC 的 layers.py 只 import 未使用。"""


def __getattr__(name):
    raise NotImplementedError("torchvision.models.%s 在 stub 中不可用" % name)
