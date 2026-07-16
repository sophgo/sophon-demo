"""Unlimited-OCR SAIL inference runtime (BM1688 / SE9-16).

Loads the combined LLM bmodel (12 blocks prefill/cache + lm_head +
greedy_head [+ optional in-bmodel 'vit' net]) and runs prefill + decode
generation with KV cache and the sliding-window no-repeat-n-gram logit
processor. The deeplip vision tower (CLIP-L + SAM ViT-B + projector) is either
a net inside the combined bmodel ('vit') or a separate bf16 bmodel.

Embedding: bmodels compiled with `--embedding_disk` have no 'embedding' net;
the table is gathered on CPU from `config/embedding.bin` ([VOCAB,H] bf16).
Otherwise the bmodel's 'embedding' net is used.

STATUS:
  * LLM generation loop: implemented (prefill/decode/KV/lm_head/ngram).
  * CPU embedding (--embedding_disk): implemented (memmap gather).
  * Vision tower (deeplip): implemented with full gundam multi-tile support.
    640x640 tiles are padded to 1024x1024, processed by the fixed-size vit
    bmodel, then cropped back to 10x10 query outputs. Embeddings follow HF
    order: [tiles...] [global + newlines] [view_seperator].
    In-bmodel 'vit' net runs on the same engine; separate --vit_bmodel uses
    SE9 staged load (release LLM -> run vit -> reload LLM) for <5120 MB
    npu layouts.
  * SE9 sophon.sail API: create_max_input_tensors + update_data + process;
    bf16 I/O via ml_dtypes (.view(bfloat16).astype(float32), NOT direct cast).
  * KV cache: manual numpy management (prefill k/v + per-step append) — robust
    against io_alone persistent-buffer semantics differences across sail versions.

Requires the bmodel to expose logits (compile with `--do_sample`) for the
no-repeat-n-gram processor to intervene; otherwise it falls back to the
bmodel's built-in topk and ngram is skipped (with a warning).
"""

import os
import time
import numpy as np
import ml_dtypes
from PIL import Image

import sophon.sail as sail
from transformers import AutoTokenizer

import preprocess as P
from ngram_processor import SlidingWindowNoRepeatNgramProcessor

# ---- model constants (DeepseekV2, see sample README §3) ----
HIDDEN_SIZE = 1280
NUM_LAYERS = 12
NUM_HEADS = 10
HEAD_DIM = 128
VOCAB_SIZE = 129280
BOS_ID = 0
EOS_ID = 1

# vit (deeplip) constants
VIT_IMG_SIZE = 1024
VIT_PATCH = 16
VIT_DOWN = 4
NQB = VIT_IMG_SIZE // VIT_PATCH // VIT_DOWN   # 16 (global image tokens per side)


class _TokWrapper:
    """Wrapper around tokenizers.Tokenizer mimicking AutoTokenizer.encode/decode.

    transformers 5.x misloads this tokenizer.json (model_type unset -> drops
    CJK bytes), so we load tokenizer.json via the tokenizers library directly.
    bos/eos are model-defined (BOS_ID=0, EOS_ID=1) and not read from here.
    """

    def __init__(self, path):
        from tokenizers import Tokenizer
        self._tok = Tokenizer.from_file(path)

    def encode(self, text, add_special_tokens=False):
        return self._tok.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids, skip_special_tokens=True):
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def __getattr__(self, name):
        return getattr(self._tok, name)


def _to_bf16(a):
    return a.astype(ml_dtypes.bfloat16).view(np.uint16)

def _from_bf16(raw):
    return raw.view(ml_dtypes.bfloat16).astype(np.float32) if raw.dtype == np.uint16 else raw.astype(np.float32)


class UnlimitedOCR:

    def __init__(self, bmodel_path, tokenizer_path, dev_id=0,
                 no_repeat_ngram_size=35, ngram_window=128,
                 vit_bmodel_path=None, vit_extras_path=None,
                 embedding_bin_path=None):
        self.dev_id = dev_id
        self.bmodel_path = bmodel_path
        self.vit_bmodel_path = vit_bmodel_path
        # CPU embedding path (bmodels compiled with --embedding_disk have no
        # "embedding" net; the table lives in config/embedding.bin as
        # [VOCAB_SIZE, HIDDEN_SIZE] bf16, gathered on CPU here).
        if embedding_bin_path is None:
            bd = os.path.dirname(os.path.abspath(bmodel_path))
            for cand in (os.path.join(bd, "config", "embedding.bin"),
                         os.path.join(bd, "..", "config", "embedding.bin")):
                if os.path.exists(cand):
                    embedding_bin_path = cand; break
        self.embedding_bin_path = embedding_bin_path
        self._embedding_table = None  # lazily memmap'd in _setup_nets
        self.tokenizer = None
        _tj = os.path.join(tokenizer_path, "tokenizer.json")
        if os.path.exists(_tj):
            try:
                self.tokenizer = _TokWrapper(_tj)
            except Exception as e:
                print(f"[warn] tokenizers load failed ({e}); falling back to AutoTokenizer")
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False, use_fast=True)
        self.ngram = (SlidingWindowNoRepeatNgramProcessor(no_repeat_ngram_size, ngram_window)
                      if no_repeat_ngram_size > 0 and ngram_window > 0 else None)

        # vit extras (image_newline / view_seperator) for forward_vision
        self.vit_extras = {}
        if vit_extras_path and os.path.exists(vit_extras_path):
            d = np.load(vit_extras_path)
            self.vit_extras = {k: d[k] for k in d.files}

        # LLM engine (loaded on demand; released while vit runs — SE9 gmem
        # cannot hold LLM (3.58GB) + vit (1.0GB) simultaneously, see README §4.1.5)
        self.model = None
        self._load_llm()

        # KV cache buffers (io_alone blocks reuse engine tensors; we keep refs
        # so the persistent history survives across decode steps).
        self.past_k = [None] * self.NUM_LAYERS
        self.past_v = [None] * self.NUM_LAYERS
        self.token_length = 0

    def _load_llm(self):
        if self.model is not None:
            return
        self.model = sail.EngineLLM(self.bmodel_path, [self.dev_id])
        self.graph_names = list(self.model.get_graph_names())
        self._setup_nets()

    def _release_llm(self):
        self.model = None
        self.tensors = {}
        self.graph_names = []

    def _load_vit(self):
        return sail.EngineLLM(self.vit_bmodel_path, [self.dev_id])

    def _setup_nets(self):
        """Discover net names by pattern (the combined bmodel's graph names)."""
        def find(pred):
            return [n for n in self.graph_names if pred(n)]

        self.name_embed = self._one(find(lambda n: n == "embedding"), optional=True)
        self.name_embed_cache = self._one(find(lambda n: n == "embedding_cache"), optional=True)
        blocks = sorted(find(lambda n: n.startswith("block_") and "cache" not in n),
                        key=lambda n: int(n.split("_")[1]))
        blocks_cache = sorted(find(lambda n: n.startswith("block_cache_")),
                              key=lambda n: int(n.split("_")[2]))
        self.NUM_LAYERS = len(blocks)
        self.name_blocks = blocks
        self.name_blocks_cache = blocks_cache
        self.name_lm = self._one(find(lambda n: n == "lm_head"))
        self.name_greedy = self._one(find(lambda n: n == "greedy_head"), optional=True)
        self.name_vit = self._one(find(lambda n: n in ("vit", "visual")), optional=True)

        if self.name_embed is not None:
            # SEQLEN from the embedding output shape [1, SEQLEN, HIDDEN_SIZE]
            shp = self.model.get_output_shape(self.name_embed, 0)
            self.SEQLEN = int(shp[1]) if len(shp) >= 3 else int(shp[0])
        else:
            # --embedding_disk: no embedding net. Load CPU table and infer SEQLEN
            # from block_0's hidden input [1, SEQLEN, H].
            if not self.embedding_bin_path or not os.path.exists(self.embedding_bin_path):
                raise RuntimeError(
                    "bmodel has no 'embedding' net (--embedding_disk) and no "
                    "embedding.bin found; pass embedding_bin_path= explicitly.")
            self._embedding_table = np.memmap(
                self.embedding_bin_path, dtype=np.uint16, mode='r'
            ).reshape(VOCAB_SIZE, HIDDEN_SIZE)
            shp = self.model.get_input_shape(blocks[0], 0)   # [1, SEQLEN, H]
            self.SEQLEN = int(shp[1])

        # SE9 sophon.sail: create_max_input/output_tensors works for both
        # addr_mode 0 (shared mem) and io_alone (persistent buffers reused
        # across calls). Fill via update_data each step. Lazy-create per net
        # (some bmodels have nets whose max buffers don't all fit alongside
        # the engine coeff; create on first use, skip unused).
        self.tensors = {}
        self._addr_mode = {n: self.model.get_addr_mode(n) for n in self.graph_names}

    def _t(self, net):
        """Lazily create + cache input/output tensors for a net on first use."""
        t = self.tensors.get(net)
        if t is None:
            t = {
                "input": self.model.create_max_input_tensors(net),
                "output": self.model.create_max_output_tensors(net),
                "addr_mode": self._addr_mode[net],
            }
            self.tensors[net] = t
        return t

    @staticmethod
    def _one(names, optional=False):
        if not names:
            if optional:
                return None
            raise RuntimeError(f"required net not found in bmodel (got none)")
        return names[0]

    # ---------------------------------------------------------------- vision
    def _run_vit_inference(self, image_batch, is_in_bmodel, vit_engine=None):
        """Run one or more images through the vit bmodel.

        image_batch: [B, 3, 1024, 1024] float32 (must be 1024x1024; 640 tiles
                     already padded).
        is_in_bmodel: True if vit is a net ('vit') in the combined bmodel.
        vit_engine: separate sail.EngineLLM (only when is_in_bmodel=False).

        Returns [B, 256, 1280] float32.
        """
        if is_in_bmodel:
            g = self.name_vit
            inp, out = self._t(g)["input"], self._t(g)["output"]
        else:
            g = "vit"
            inp = vit_engine.create_max_input_tensors(g)
            out = vit_engine.create_max_output_tensors(g)

        results = []
        for i in range(image_batch.shape[0]):
            img = np.ascontiguousarray(image_batch[i:i + 1], dtype=np.float32)
            inp[0].update_data(img.reshape(inp[0].shape()))
            if is_in_bmodel:
                self.model.process(g, inp, out)
            else:
                vit_engine.process(g, inp, out)
            results.append(_from_bf16(out[0].asnumpy()))   # [1,256,1280]
        return np.concatenate(results, axis=0)              # [B,256,1280]

    def forward_vision(self, images_crop, images_ori, crop_ratio, image_mode="gundam"):
        """Run the deeplip vision tower (CLIP-L + SAM ViT-B + projector).

        Gundam tiling (crop_ratio > (1,1)): processes global view + 640x640
        tiles (padded to 1024x1024 for the fixed-size bmodel, then cropped
        back to 10x10 query output per tile).

        Base mode (image_mode='base'): single 640x640 view padded to
        1024x1024, output cropped to 10x10 (111 image tokens).

        Returns image embeddings [num_image_tokens, 1280] float32, in the
        HF-native order: [tiles ...] [global ...] [view_seperator].
        (The token-grid structure in build_image_tokens is [global+newlines]
        [sep] [tiles+newlines]; the order swap is deliberate — the HF model
        was trained with this.)
        """
        if self.name_vit is None and not self.vit_bmodel_path:
            raise NotImplementedError(
                "Vision tower (deeplip) bmodel not provided. Text-only mode. "
                "Either compile with vit in-bmodel (net 'vit') or pass --vit_bmodel.")

        w, h = crop_ratio
        has_tiles = w > 1 or h > 1
        nq = P.num_queries_for(640)     # 10 — tile queries per side
        # Global query side: 16 for gundam (true 1024x1024), 10 for base (640 padded→1024)
        nqb_global = nq if image_mode == "base" else NQB

        nl_1d = self.vit_extras["model.image_newline"].reshape(HIDDEN_SIZE)  # [1280]
        vs_1d = self.vit_extras["model.view_seperator"].reshape(HIDDEN_SIZE)

        # --- helper: turn [B, nq_side, nq_side, H] into [B, nq_side*(nq_side+1), H] with newlines ---
        def _assemble_global(emb, nq_side):
            """emb: [B, nq_side, nq_side, H] -> [B, nq_side*(nq_side+1), H]"""
            B = emb.shape[0]
            rows = []
            for r in range(nq_side):
                rows.append(emb[:, r, :, :])                                        # [B, nq_side, H]
                rows.append(np.broadcast_to(nl_1d, (B, 1, HIDDEN_SIZE)))           # [B, 1, H]
            return np.concatenate(rows, axis=1)                                     # [B, nq_side*(nq_side+1), H]

        # --- 1. decide vit backend ---
        is_in_bmodel = self.name_vit is not None
        vit_engine = None   # only used when not in-bmodel

        if not is_in_bmodel:
            vit_engine = self._load_vit()

        try:
            # --- 2. global view (always) ---
            # images_ori is [1,3,1024,1024] (base-mode 640 view already padded)
            global_out = self._run_vit_inference(images_ori, is_in_bmodel, vit_engine)
            # global_out: [1, 256, 1280] -> [1, NQB, NQB, 1280] -> crop to [nqb_global, nqb_global, 1280]
            global_grid = global_out.reshape(1, NQB, NQB, HIDDEN_SIZE)[:, :nqb_global, :nqb_global, :]

            # --- 3. tile views (if any) ---
            tile_embs = []  # each [10, 10, 1280]
            if has_tiles:
                n_tiles = images_crop.shape[0]  # == w * h
                # Pad each [3, 640, 640] tile to [3, 1024, 1024] with zeros
                # (PAD_COLOR 127,127,127 normalises to ~0 in [-1,1] space)
                tiles_padded = np.zeros((n_tiles, 3, 1024, 1024), dtype=np.float32)
                tiles_padded[:, :, :640, :640] = images_crop[:, :, :640, :640]
                tile_out = self._run_vit_inference(tiles_padded, is_in_bmodel, vit_engine)
                # tile_out: [N, 256, 1280] -> [N, NQB, NQB, 1280] -> crop [N, nq, nq, 1280]
                tile_out = tile_out.reshape(n_tiles, NQB, NQB, HIDDEN_SIZE)[:, :nq, :nq, :]
                tile_embs = [tile_out[i] for i in range(n_tiles)]

            # --- 4. assemble HF-order embeddings: [tiles] [global] [sep] ---
            if not has_tiles:
                # Single view: [global+newlines] [sep]
                global_flat = _assemble_global(global_grid, nqb_global)  # [1, nqb_global*(nqb_global+1), 1280]
                vs_batch = vs_1d.reshape(1, 1, HIDDEN_SIZE)
                emb = np.concatenate([global_flat, vs_batch], axis=1)
                return emb[0]

            # Multi-tile: [tiles...] [global...] [sep]
            # 4a. Tile grid: arrange N tiles into [nq*h, nq*w, 1280] spatial grid
            tile_grid = np.zeros((nq * h, nq * w, HIDDEN_SIZE), dtype=np.float32)
            for idx, temb in enumerate(tile_embs):
                tr = idx // w
                tc = idx % w
                tile_grid[tr * nq:(tr + 1) * nq, tc * nq:(tc + 1) * nq, :] = temb

            # 4b. Flatten tile grid with newlines: [(nq*w+1)*(nq*h), 1280]
            tile_rows = []
            for r in range(nq * h):
                tile_rows.append(tile_grid[r])                              # [nq*w, 1280]
                tile_rows.append(np.broadcast_to(nl_1d, (1, HIDDEN_SIZE)))  # [1, 1280]
            tile_flat = np.concatenate(tile_rows, axis=0)                   # [tile_tokens, 1280]

            # 4c. Global: [nqb_global*(nqb_global+1), 1280]
            global_flat = _assemble_global(global_grid, nqb_global)[0]

            # 4d. Concatenate: [tiles][global][sep]  (HF line 540 order)
            emb = np.concatenate([tile_flat, global_flat, vs_1d.reshape(1, HIDDEN_SIZE)], axis=0)
            return emb

        finally:
            if not is_in_bmodel:
                del vit_engine

    # ----------------------------------------------------------------- prefill
    def forward_first(self, input_ids, image_embeddings=None, image_seq_mask=None):
        length = len(input_ids)
        self.token_length = length
        ids = np.zeros(self.SEQLEN, dtype=np.int32)
        ids[:length] = input_ids

        # embedding — keep bf16 uint16 throughout (matches tf test)
        if self.name_embed is not None:
            ti_e = self._t(self.name_embed)
            ti_e["input"][0].update_data(ids.reshape(ti_e["input"][0].shape()))
            self.model.process(self.name_embed, ti_e["input"], ti_e["output"])
            hu16 = np.ascontiguousarray(ti_e["output"][0].asnumpy()).reshape(1, self.SEQLEN, HIDDEN_SIZE)
        else:
            hu16 = np.array(self._embedding_table[ids]).reshape(1, self.SEQLEN, HIDDEN_SIZE)
            hu16 = np.ascontiguousarray(hu16)

        # merge vision embeddings (in float32 then back to bf16 for precision)
        if image_embeddings is not None and image_seq_mask is not None:
            mask = np.asarray(image_seq_mask, dtype=bool)
            mask = np.concatenate([mask, np.zeros(self.SEQLEN - len(mask), dtype=bool)])
            true_idx = np.where(mask[:self.SEQLEN])[0]
            n_img = min(len(true_idx), image_embeddings.shape[0])
            hf32 = _from_bf16(hu16)
            hf32[0, true_idx[:n_img], :] = image_embeddings[:n_img]
            hu16 = _to_bf16(hf32).reshape(1, self.SEQLEN, HIDDEN_SIZE)
            hu16 = np.ascontiguousarray(hu16)

        # position ids and attention mask — identical to verified tf test
        pos = np.arange(self.SEQLEN, dtype=np.int32).reshape(1, self.SEQLEN)
        attn = np.ones((self.SEQLEN, self.SEQLEN), dtype=np.float32) * -10000.0
        for i in range(length):
            attn[i, :i + 1] = 0
        for i in range(length, self.SEQLEN):
            attn[i, :] = -10000.0
            attn[i, i] = 0
        au16 = _to_bf16(attn.reshape(1, 1, self.SEQLEN, self.SEQLEN))

        # run blocks — identical to tf test (bf16 uint16, contiguous arrays)
        for i in range(self.NUM_LAYERS):
            bi = self.name_blocks[i]
            inp = self.model.create_max_input_tensors(bi)
            out = self.model.create_max_output_tensors(bi)
            inp[0].update_data(np.ascontiguousarray(hu16).reshape(inp[0].shape()))
            inp[1].update_data(pos.reshape(inp[1].shape()))
            inp[2].update_data(np.ascontiguousarray(au16).reshape(inp[2].shape()))
            self.model.process(bi, inp, out)
            hu16 = np.ascontiguousarray(out[0].asnumpy()).reshape(1, self.SEQLEN, HIDDEN_SIZE)
            self.past_k[i] = np.ascontiguousarray(out[1].asnumpy()).copy()
            self.past_v[i] = np.ascontiguousarray(out[2].asnumpy()).copy()

        # lm_head on last position
        last_hu16 = np.ascontiguousarray(hu16[:, length - 1:length, :])
        return self._lmhead(last_hu16)

    # ------------------------------------------------------------------ decode
    def forward_next(self, last_token):
        self.token_length += 1
        pos = self.token_length - 1
        ids = np.array([last_token], dtype=np.int32)

        emb_net = self.name_embed_cache or self.name_embed
        if emb_net is not None:
            einp = self.model.create_max_input_tensors(emb_net)
            eout = self.model.create_max_output_tensors(emb_net)
            einp[0].update_data(ids.reshape(einp[0].shape()))
            self.model.process(emb_net, einp, eout)
            hu16 = np.ascontiguousarray(eout[0].asnumpy()).reshape(1, 1, HIDDEN_SIZE)
        else:
            hu16 = np.array(self._embedding_table[ids]).reshape(1, 1, HIDDEN_SIZE)
            hu16 = np.ascontiguousarray(hu16)

        # decode attention mask [1,1,1,SEQLEN+1]: 0 for past tokens, -10000 future
        attn = np.zeros((1, 1, 1, self.SEQLEN + 1), dtype=np.float32)
        attn[0, 0, 0, pos + 1:] = -10000.0
        pos_id = np.array([pos], dtype=np.int32)

        for i in range(self.NUM_LAYERS):
            ci = self.name_blocks_cache[i]
            inp = self.model.create_max_input_tensors(ci)
            out = self.model.create_max_output_tensors(ci)
            inp[0].update_data(hu16.reshape(inp[0].shape()))
            inp[1].update_data(pos_id.reshape(inp[1].shape()))
            inp[2].update_data(_to_bf16(attn).reshape(inp[2].shape()))
            inp[3].update_data(self.past_k[i].reshape(inp[3].shape()))
            inp[4].update_data(self.past_v[i].reshape(inp[4].shape()))
            self.model.process(ci, inp, out)
            hu16 = np.ascontiguousarray(out[0].asnumpy()).reshape(1, 1, HIDDEN_SIZE)
            # append this step's new k/v into the history at position `pos`
            new_k = np.ascontiguousarray(out[1].asnumpy())
            new_v = np.ascontiguousarray(out[2].asnumpy())
            self.past_k[i][:, pos:pos + 1, :, :] = new_k
            self.past_v[i][:, pos:pos + 1, :, :] = new_v

        return self._lmhead(hu16)

    # ----------------------------------------------------------------- lm_head
    def _lmhead(self, hu16):
        """hu16: [1,1,H] bf16 uint16. Returns next token id (greedy)."""
        inp = self.model.create_max_input_tensors(self.name_lm)
        out = self.model.create_max_output_tensors(self.name_lm)
        inp[0].update_data(hu16.reshape(inp[0].shape()))
        self.model.process(self.name_lm, inp, out)
        out_data = out[0].asnumpy()
        if self.name_greedy is None:
            # lm_head already did topk -> returns token id (int32)
            return int(out_data.reshape(-1)[0])
        # logits mode: f32 [1,129280]; apply ngram then greedy_head argmax
        if self.ngram is not None:
            seq = np.array([self._generated_ids], dtype=np.int64)
            logits = out_data.reshape(1, -1).astype(np.float32)
            logits = self.ngram(seq, logits)
            return int(np.argmax(logits[0]))
        ginp = self.model.create_max_input_tensors(self.name_greedy)
        gout = self.model.create_max_output_tensors(self.name_greedy)
        ginp[0].update_data(out_data.reshape(ginp[0].shape()))
        self.model.process(self.name_greedy, ginp, gout)
        return int(gout[0].asnumpy().reshape(-1)[0])

    # --------------------------------------------------------------- generate
    def generate(self, prompt, image=None, image_mode="gundam",
                 max_new_tokens=2048, verbose=True):
        """prompt must contain '<image>' if image is given."""
        img_emb = None
        img_mask = None
        if image is not None:
            img = image if isinstance(image, Image.Image) else Image.open(image)
            images_crop, images_ori, crop_ratio = P.preprocess_image(
                img, image_mode=image_mode)
            img_tokens = P.build_image_tokens(image_mode=image_mode,
                                              crop_ratio=crop_ratio)
            prompt = prompt if "<image>" in prompt else "<image>" + prompt
            input_ids, img_mask = P.build_input_ids(self.tokenizer, prompt, img_tokens, BOS_ID)
            # Vision: if vit is a net in the combined bmodel (name_vit), run it
            # on the same engine (no release/reload — both already in gmem via
            # the memory_edit npu-5120 layout). Else (separate vit_bmodel_path)
            # do SE9 staged loading: release LLM -> run vit -> reload LLM.
            t0 = time.time()
            if self.name_vit is None:
                if verbose:
                    print(f"[vision] release LLM -> run vit -> reload LLM ...", flush=True)
                self._release_llm()
            img_emb = self.forward_vision(images_crop, images_ori, crop_ratio,
                                           image_mode=image_mode)
            if self.name_vit is None:
                self._load_llm()
            if verbose:
                print(f"[vision] vit done in {time.time()-t0:.1f}s, {img_emb.shape[0]} img tokens", flush=True)
        else:
            input_ids = [BOS_ID] + self.tokenizer.encode(prompt, add_special_tokens=False)

        if len(input_ids) >= self.SEQLEN:
            raise ValueError(f"input too long: {len(input_ids)} >= SEQLEN {self.SEQLEN}")

        self._generated_ids = list(input_ids)
        t0 = time.time()
        tok = self.forward_first(input_ids, img_emb, img_mask)
        self._generated_ids.append(tok)
        out = self.tokenizer.decode([tok], skip_special_tokens=True)
        if verbose:
            print(out, end="", flush=True)
        n = 1
        while n < max_new_tokens and tok != EOS_ID and self.token_length < self.SEQLEN:
            tok = self.forward_next(tok)
            self._generated_ids.append(tok)
            if tok == EOS_ID:
                break
            if verbose:
                print(self.tokenizer.decode([tok], skip_special_tokens=True), end="", flush=True)
            n += 1
        if verbose:
            print(f"\n[generated {n} tokens in {time.time()-t0:.1f}s]")
        return self.tokenizer.decode(self._generated_ids[len(input_ids):], skip_special_tokens=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bmodel", required=True)
    ap.add_argument("--vit_bmodel", default=None,
                    help="path to the deeplip vit bmodel (bf16). Required for image OCR.")
    ap.add_argument("--vit_extras", default=None,
                    help="path to vit_extras.npz (image_newline/view_seperator)")
    ap.add_argument("--tokenizer", required=True, help="path to the Unlimited-OCR weights dir (has tokenizer.json)")
    ap.add_argument("--dev", type=int, default=0)
    ap.add_argument("--image", default=None)
    ap.add_argument("--prompt", default="<image>document parsing.")
    ap.add_argument("--image_mode", choices=("gundam", "base"), default="gundam")
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--ngram_size", type=int, default=35)
    ap.add_argument("--ngram_window", type=int, default=128)
    ap.add_argument("--embedding_bin", default=None,
                    help="path to config/embedding.bin (auto-detected from bmodel dir if omitted; needed for --embedding_disk bmodels)")
    args = ap.parse_args()

    engine = UnlimitedOCR(args.bmodel, args.tokenizer, dev_id=args.dev,
                          no_repeat_ngram_size=args.ngram_size,
                          ngram_window=args.ngram_window,
                          vit_bmodel_path=args.vit_bmodel,
                          vit_extras_path=args.vit_extras,
                          embedding_bin_path=args.embedding_bin)
    if args.image is None:
        print("NOTE: no --image; running text-only (vision tower not exercised).")
    engine.generate(args.prompt, image=args.image, image_mode=args.image_mode,
                    max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
