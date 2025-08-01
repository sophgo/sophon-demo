# Open Clip模型的导出与编译
## 1. 导出公版onnx模型

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

## 导出encode_text部分

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

## 2. 性能优化
为避免不必要的数据搬运，导出onnx时按以下步骤修改，可显著缩短推理时间。
导出text前在[1. 导出公版onnx模型](#1-导出公版onnx模型)
的基础上还需要修改CLIP/clip/model.py文件里clip类下的encode_text函数

```python
def encode_text(self, text):
    x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

    x = x + self.positional_embedding.type(self.dtype)
    # x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.transformer(x)
    # x = x.permute(1, 0, 2)  # LND -> NLD
    x = self.ln_final(x).type(self.dtype)

    # x.shape = [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
    np.save('text_projection_512_512.npy', self.text_projection)

    return x
```

导出image部分需要调整CLIP/clip/model.py文件中VisionTransformer类的forward函数

```python
def forward(self, x: torch.Tensor):
    x = self.conv1(x)  # shape = [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    x = torch.cat([
        self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
        x
    ], dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + self.positional_embedding.to(x.dtype)
    x = self.ln_pre(x)

    # x = x.permute(1, 0, 2)  # NLD -> LND (optional)
    x = self.transformer(x)
    # x = x.permute(1, 0, 2)  # LND -> NLD (optional)

    x = self.ln_post(x[:, 0, :])

    if self.proj is not None:
        x = x @ self.proj

    return x
```

接着找到这个文件：~/.local/lib/python3.10/site-packages/torch/nn/functional.py。如果找不到，请通过pip3 show torch查包的安装位置。
找到里面的multi_head_attention_forward函数，修改以下四处：

step 1
```python
    # set up shape vars
    # tgt_len, bsz, embed_dim = query.shape
    # src_len, _, _ = key.shape
    bsz, tgt_len, embed_dim = query.shape
    _, src_len, _ = key.shape
```

step 2
```python
    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        assert (
            in_proj_weight is not None
        ), "use_separate_proj_weight is False but in_proj_weight is None"
        # q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
        q_proj_weight, k_proj_weight, v_proj_weigh = in_proj_weight.chunk(3)
        b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(
            query,
            key,
            value,
            q_proj_weight,
            k_proj_weight,
            v_proj_weigh,
            b_q,
            b_k,
            b_v,
        )
```

step 3
```python
    #
    # reshape q, k, v for multihead attention and make them batch first
    #
    # q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    q = q.view(bsz, tgt_len, num_heads, head_dim).transpose(1, 2)
    if static_k is None:
        # k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        k = k.view(bsz, tgt_len, num_heads, head_dim).transpose(1, 2)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert (
            static_k.size(0) == bsz * num_heads
        ), f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert (
            static_k.size(2) == head_dim
        ), f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        # v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        v = v.view(bsz, tgt_len, num_heads, head_dim).transpose(1, 2)
```

step 4
```python
    else:
        # attn_mask can be either (L,S) or (N*num_heads, L, S)
        # if attn_mask's shape is (1, L, S) we need to unsqueeze to (1, 1, L, S)
        # in order to match the input for SDPA of (N, num_heads, L, S)
        if attn_mask is not None:
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)

        # q = q.view(bsz, num_heads, tgt_len, head_dim)
        # k = k.view(bsz, num_heads, src_len, head_dim)
        # v = v.view(bsz, num_heads, src_len, head_dim)

        attn_output = scaled_dot_product_attention(
            q, k, v, attn_mask, dropout_p, is_causal
        )
        attn_output = (
            attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
        )

        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
        # attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
        return attn_output, None
```
修改完毕后导出onnx模型

# Mobile Clip模型的导出与编译
## 1. 导出公版mobile clip onnx模型

原始仓库：[Mobile CLIP官方开源仓库](https://github.com/apple/ml-mobileclip/tree/main/mobileclip)

## 导出encode_image部分

模型分encode_image和encode_text两部分，如果需要导出encode_image部分，修改源码 ml-mobileclip/mobileclip/clip.py 的 forward 函数
```python
def forward(
    self,
    image: Optional[torch.Tensor] = None,
    *args,
    **kwargs
) -> Any:

    image_embeddings = (
        self.encode_image(image, normalize=True) if image is not None else None
    )
    # text_embeddings = (
    #     self.encode_text(text, normalize=True) if text is not None else None
    # )

    # if self.output_dict:
    #     return {
    #         "image_features": image_embeddings,
    #         "text_features": text_embeddings,
    #         "logit_scale": self._exponentiate_and_clip_logits(),
    #     }
    return image_embeddings
```
然后运行以下代码导出onnx模型

```python
import torch
from PIL import Image
import mobileclip
import numpy as np

model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_b', pretrained='../mobileclip_b.pt')# pretrained参数按照模型实际路径填写，该参数也可以填blt模型

image = preprocess(Image.open("docs/fig_accuracy_latency.png").convert('RGB')).unsqueeze(0)

with torch.no_grad():
    # Assuming 'model' is your PyTorch model and 'image' is the input tensor
    torch.onnx.export(
        model,                # model being run
        image,                 # model input (or a tuple for multiple inputs)
        "mobile_clip_image_b.onnx",          # where to save the model (can be a file or file-like object)
        dynamic_axes={'image': {0: 'batch_size'},
                      'output': {0: 'batch_size'}},  # dynamic axes of the input
        input_names=['image'], # setting the input name to 'text'
        output_names=['output'] # you can also set the output name(s) if necessary
    )
```

## 导出encode_text部分

同理，修改源码 ml-mobileclip/mobileclip/clip.py 的 forward 函数；
```python
def forward(
    self,
    text: Optional[torch.Tensor] = None,
    *args,
    **kwargs
) -> Any:

    # image_embeddings = (
    #     self.encode_image(image, normalize=True) if image is not None else None
    # )
    text_embeddings = (
        self.encode_text(text, normalize=True) if text is not None else None
    )

    # if self.output_dict:
    #     return {
    #         "image_features": image_embeddings,
    #         "text_features": text_embeddings,
    #         "logit_scale": self._exponentiate_and_clip_logits(),
    #     }
    return text_embeddings
```
另外，注意在ml-mobileclip/mobileclip/text_encoder.py需要注释掉argmax以及投影变换操作，并同时保存导出onnx时的self.projection_layer数据，为后续推理使用；

```python
def encode_text(
    self,
    text_tokens: Tensor,
    key_padding_mask: Optional[Tensor] = None,
    return_all_tokens: bool = False,
    *args,
    **kwargs
) -> Tensor:
    """Return text token embeddings.

    Args:
        text_tokens: a tensor of token indices. Shape: [batch_size, context_length]
        key_padding_mask: a tensor of boolean values as the padding mask.
            Shape: [batch_size, context_length]
        return_all_tokens: a boolean flag to return all tokens, defaults to False
            to return only EOT token embedding.
    Returns:
        A tensor of [batch_size, context_length, hidden_dim] if return_all_tokens is
        True, otherwise a tensor of [batch_size, hidden_dim].
    """
    # Discrete tokens to continuous embeddings
    # [batch_size, context_length] --> [batch_size, context_length, hidden_dim]
    token_emb = self.forward_embedding(text_tokens)

    # [1, context_length, context_length]
    attn_mask = None
    if self.causal_masking:
        attn_mask = self.build_attention_mask(
            context_length=text_tokens.shape[1], batch_size=text_tokens.shape[0]
        )
        attn_mask = attn_mask.to(device=token_emb.device, dtype=token_emb.dtype)
        key_padding_mask = None

    for layer in self.transformer:
        token_emb = layer(
            token_emb,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )

    # Apply layer norm
    token_emb = self.final_layer_norm(token_emb)

    if return_all_tokens:
        return token_emb

    # Take features from the eot embedding (eot_token is the highest number in each sequence)
    # token_emb = token_emb[
    #     torch.arange(text_tokens.shape[0]), text_tokens.argmax(dim=-1)
    # ]

    # token_emb = token_emb @ self.projection_layer
    np.save('text_projection_b.npy',self.projection_layer)
    return token_emb
```

同时，在源码 ml-mobileclip/mobileclip/text_encoder.py 的 build_attention_mask 函数中，将 mask.fill_(float("-inf")) 改成 mask.fill_(float("-10000"))
```python
def build_attention_mask(self, context_length: int, batch_size: int) -> Tensor:
    """Build causal attention mask [batch_size, context_length, context_length]."""
    # Build mask with full attention between the tokens
    # pytorch uses additive attention mask; fill with -inf
    mask = torch.empty(context_length, context_length)
    # mask.fill_(float("-inf"))
    mask.fill_(float("-10000"))
    mask.triu_(1)  # zero out the lower diagonal
    mask = mask.unsqueeze(0)  # add dummy batch dimension
    mask = mask.expand(batch_size, -1, -1)
    return mask
```

然后运行以下代码导出onnx模型

```python
import torch
from PIL import Image
import mobileclip
import numpy as np

model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_b', pretrained='../mobileclip_b.pt')# pretrained参数按照模型实际路径填写，该参数也可以填blt模型
tokenizer = mobileclip.get_tokenizer('mobileclip_b')

text = tokenizer(["a diagram", "a dog", "a cat"])

with torch.no_grad():
    # Assuming 'model' is your PyTorch model and 'text' is the input tensor
    torch.onnx.export(
        model,                # model being run
        text,                 # model input (or a tuple for multiple inputs)
        "mobile_clip_text_b.onnx",          # where to save the model (can be a file or file-like object)
        dynamic_axes={'text': {0: 'batch_size'},
                      'output': {0: 'batch_size'}},  # dynamic axes of the input
        input_names=['text'], # setting the input name to 'text'
        output_names=['output'] # you can also set the output name(s) if necessary
    )
```

