# Clip模型的导出与编译


需要通过源码来导出 onnx 文件，clip的变种很多，但是思想类似，以 openai 原始仓库[CLIP官方开源仓库](https://github.com/openai/CLIP)为例。

## 导出encode_image部分

模型分encode_image和encode_text两部分，以ViT-B/32模型为例，如果需要导出encode_image部分，修改源码 CLIP/clip/model.py:358
```python
    # def forward(self, image, text):
    def forward(self, image):
        image_features = self.encode_image(image)
        # text_features = self.encode_text(text)

        # normalized features
        # image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # # cosine similarity as logits
        # logit_scale = self.logit_scale.exp()
        # logits_per_image = logit_scale * image_features @ text_features.t()
        # logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        # return logits_per_image, logits_per_text
        return image_features
```
然后运行以下代码导出onnx模型

```python
import torch
from clip import *
from PIL import Image
import torch

device =  "cpu"
model, preprocess = clip.load("ViT-B/32", device=device) # 此处路径可以替换成本地pt文件路径
image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog"] * 256).to(device)

with torch.no_grad():
    # Assuming 'model' is your PyTorch model and 'text' is the input tensor
    torch.onnx.export(
        model,                # model being run
        image,                 # model input (or a tuple for multiple inputs)
        "clip_image_vitb32.onnx",          # where to save the model (can be a file or file-like object)
        dynamic_axes={'image': {0: 'batch_size'},
                      'output': {0: 'batch_size'}},  # dynamic axes of the input
        input_names=['image'], # setting the input name to 'text'
        output_names=['output'] # you can also set the output name(s) if necessary
    )
```

## 导出text_image部分

同理，修改源码 CLIP/clip/model.py:358处；
```python
    # def forward(self, image, text):
    def forward(self, text):
        # image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        # image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # # cosine similarity as logits
        # logit_scale = self.logit_scale.exp()
        # logits_per_image = logit_scale * image_features @ text_features.t()
        # logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        # return logits_per_image, logits_per_text
        return text_features
```
另外，注意在CLIP/clip/model.py:354行也需要注释掉，并同时保存导出onnx时的self.text_projection数据，为后续推理使用；
```python
    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)

        
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        np.save('text_projection_512_512.npy',self.text_projection)
        
        return x
```


然后运行以下代码导出onnx模型

```python
import torch
from clip import *
from PIL import Image
import torch

device =  "cpu"
model, preprocess = clip.load("ViT-B/32", device=device) # 此处路径可以替换成本地pt文件路径
image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog"] * 256).to(device)

with torch.no_grad():
    # Assuming 'model' is your PyTorch model and 'text' is the input tensor
    torch.onnx.export(
        model,                # model being run
        text,                 # model input (or a tuple for multiple inputs)
        "clip_text_vitb32.onnx",          # where to save the model (can be a file or file-like object)
        dynamic_axes={'text': {0: 'batch_size'},
                      'output': {0: 'batch_size'}},  # dynamic axes of the input
        input_names=['text'], # setting the input name to 'text'
        output_names=['output'] # you can also set the output name(s) if necessary
    )
```

