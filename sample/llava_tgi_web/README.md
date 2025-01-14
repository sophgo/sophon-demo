# LLaVA Web Demo

## 1. 初始化TGI容器

从SDK目录中获取镜像，镜像文件名参考 `docker-soph_tgi-<version>-slim-<date>-<commit_id>-<hash>.tar.bz2`

```bash
bunzip2 -c docker-soph_tgi-<version>-slim-<date>-<commit_id>-<hash>.tar.bz2 | docker load
```

```bash
 docker run --privileged --name <your_container_name> -td -p <host port>:<container port> --ipc host -v /dev/:/dev/ -v <your data path>:/data -v /opt/:/opt/ -v <your work directory>:/workspace/ --entrypoint bash soph_tgi:<version>-slim
```

## 2. 获取模型

```bash
git lfs install
git clone https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf/tree/main
```

## 3. 在容器中启动TGI服务

```bash
text-generation-launcher --model-id /data/llava-v1.6-vicuna-7b-hf/  --hostname 0.0.0.0 -p <container port>
```

## 4. 启动Web Demo

```bash
pip3 install streamlit
pip3 install text-generation
streamlit run ./llava_web_demo.py <server url>
```

这里，启动web_demo时，默认的`server url`为`http://localhost:8099`。如果上文中的容器与这里的web demo在同一台设备上，而且容器使用了宿主机的8099端口，那么不需要输入此参数。

运行后，会在终端看到类似下面的打印：

```bash
You can now view your Streamlit app in your browser.

Local URL: http://localhost:port
Network URL: http://host_ip:port
External URL: http://ip:port
```

如果使用ssh在服务器上运行本例程，可以使用第二个`URL`打开浏览器。

## 5. 操作说明

`llava`是多模态大模型，支持针对图片进行问答。本例程目前使用英文对话效果较好。

首先，点击左侧的`Browse files`按钮选择图片，点击`submit`上传。此时可以在右侧看到图片预览。

然后在右侧对话框中输入问题，按`Enter`进行推理。如果图片格式无误的话，稍等即可看到推理结果。

推理结果的最大长度为左侧的`Max New Tokens`滑动条的值，可以自行拖动设置。
