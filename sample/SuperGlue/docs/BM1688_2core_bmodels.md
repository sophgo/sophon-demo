# BM1688双核模型编译方法
由于目前公版的mlir编译superglue的双核模型存在bug，这里提供一个私版mlir，可以编译出双核模型。下载方式如下：
```bash
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:sophon-demo/SuperGlue/tpu_mlir-1.12b0.tar.gz
tar xvf tpu_mlir-1.12b0.tar.gz #解压得到文件夹：tpu_mlir-1.12b0
```

配置方式如下：

```bash
#进入之前配置的mlir环境：
docker attach {tpu_mlir}
cd /path/to/tpu_mlir-1.12b0 #私版tpu-mlir在docker里的路径

#卸载原有mlir：
pip3 uninstall tpu_mlir

#配置私版mlir环境变量：
source release_tools/envsetup.sh
```

双核模型编译方法如下：
只需要给`gen_fp16bmodel_mlir.sh`里的model_deploy.py命令加一个`--num_core 2`的参数，这样编译出来的模型就是2core模型。

这里提供两个编译好的模型：
```bash
python3 -m dfss --url=open@sophgo.com:sophon-demo/SuperGlue/superpoint_fp16_1b_2core.bmodel
python3 -m dfss --url=open@sophgo.com:sophon-demo/SuperGlue/superglue_fp16_1b_iter20_1024_2core.bmodel
```