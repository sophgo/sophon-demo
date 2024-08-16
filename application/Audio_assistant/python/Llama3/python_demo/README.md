# 环境准备
> （python demo和web demo运行之前都需要执行这个）
```
sudo apt-get update
sudo apt-get install pybind11-dev
pip3 install sentencepiece transformers==4.30.2
pip3 install gradio==3.39.0 mdtex2html==1.2.0 dfss
```

如果不打算自己编译模型，可以直接用下载好的模型
```
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/llama3-8b_int4_1dev_seq512.bmodel
```

编译库文件
```
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

# python demo
```
python3 pipeline.py --model_path llama3-8b_int4_1dev_seq512.bmodel --tokenizer_path ../support/token_config/ --devid 0 --generation_mode greedy
```

# web demo
```
python3 web_demo.py --model_path llama3-8b_int4_1dev_seq512.bmodel --tokenizer_path ../support/token_config/ --devid 0
```

## 常见问题
### Could not create share link. Missing file: /usr/local/lib/python3.10/dist-packages/gradio/frpc_linux_amd64_v0.2.

AMD / PCIE
```
1. Download this file: https://cdn-media.huggingface.co/frpc-gradio-0.2/frpc_linux_amd64
2. Rename the downloaded file to: frpc_linux_amd64_v0.2
3. Move the file to this location: /usr/local/lib/python3.10/dist-packages/gradio
```

ARM / SOC

### ImportError:/home/linaro/.local/lib/python3.8/site-packages/torch/libs/libgomp-6e1a1d1b.so.1.0.0: cannot allocate memory in static TLs block

```
export LD_PRELOAD=/home/linaro/.local/lib/python3.8/site-packages/torch/lib/libgomp-d22c30c5.so.1
```

### OSError: /home/linaro/.local/lib/python3.8/site-packages/torch/lib/libgomp-d22c30c5.so.1: cannotallocate memoryin staticTLS block

```
export LD_PRELOAD=/home/linaro/.local/lib/python3.8/site-packages/torch/lib/libgomp-d22c30c5.so.1
```
