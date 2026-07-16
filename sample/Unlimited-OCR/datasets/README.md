# datasets/

测试数据目录。

## 文件说明
| 文件 | 用途 |
|------|------|
| `test_prompts.txt` | 文本生成提示词列表（每行一个，空行和 # 行忽略） |
| `doc_english.png` | 英文文档（公司年度报告摘要） |
| `doc_chinese.png` | 中文文档（工作计划通知） |
| `doc_receipt.png` | 英文收据（超市购物小票） |

## 测试图片说明
三张测试图覆盖典型 OCR 场景：英文正式文档、中文办公通知、结构化票据。视觉塔 bmodel 就绪后可直接用。如需更多测试素材，推荐：
- 复杂排版：多栏论文、表格文档、混排中英文
- PDF：多页 PDF 用于 `--image_mode base` 逐页解析（放入后可用 `pdf_to_images.py` 转成单页图）

放入图片后，用以下命令测试：

### 文本生成（当前可用）
```bash
while IFS= read -r line; do
    [[ "$line" =~ ^# ]] || [ -z "$line" ] && continue
    echo "=== prompt: $line ==="
    python3 ../python/unlimited_ocr_sail.py \
        --bmodel ../models/<bmodel>.bmodel \
        --tokenizer ../models/config \
        --prompt "$line" --max_new_tokens 64
done < test_prompts.txt
```

### 图像 OCR（视觉塔 bmodel 就绪后）
```bash
python3 ../python/unlimited_ocr_sail.py \
    --bmodel ../models/<withvit>.bmodel \
    --tokenizer ../models/config \
    --vit_extras ../models/config/vit_extras.npz \
    --image your_document.jpg --image_mode gundam \
    --prompt "<image>document parsing." \
    --ngram_size 35 --ngram_window 128 --max_new_tokens 2048
```
