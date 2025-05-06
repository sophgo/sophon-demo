# Qwen Web Demo

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
git clone https://huggingface.co/Qwen/Qwen2-7B-Instruct-GPTQ-Int4
```

## 3. 在容器中启动TGI服务

```bash
text-generation-launcher --model-id /data/Qwen2-7B-Instruct-GPTQ-Int4 --dtype bfloat16 --hostname 0.0.0.0 -p <container port>
```

## 4. 启动Web Demo

```bash
pip3 install streamlit
pip3 install text-generation
pip3 install tiktoken
pip3 install openai
streamlit run ./qwen_web_demo.py <server url> <MAX_CONTEXT_TOKENS>
```

这里，启动web_demo时，默认的`server url`为`http://localhost:8090/v1`,默认的`MAX_CONTEXT_TOKENS`为2048。如果上文中的容器与这里的web demo在同一台设备上，而且容器使用了宿主机的8090端口，那么不需要输入此参数。

运行后，会在终端看到类似下面的打印：

```bash
You can now view your Streamlit app in your browser.

Local URL: http://localhost:port
Network URL: http://host_ip:port
External URL: http://ip:port
```

如果使用ssh在服务器上运行本例程，可以使用第二个`URL`打开浏览器。