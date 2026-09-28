#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3.5 Web UI: minimal Flask front-end around the SAIL inference engine.

Runs the same Qwen3_5 engine as qwen3_5.py's interactive CLI, exposed over
HTTP so it can be driven from a browser (textarea handles multi-line input
natively; no input() prompt sequencing involved).

Usage (on the board):
    python3 webui.py -m <bmodel> -c <config_dir> [--host 0.0.0.0] [--port 8000]

Then open http://<board-ip>:8000. To reach the board through a jump host:
    ssh -L 8000:<board-ip>:8000 linaro@<jump-host>   # then browse localhost:8000

Endpoints:
    GET  /            UI page (webui.html, same directory as this script)
    GET  /api/info    model / session info (JSON)
    POST /api/upload  multipart "file" upload -> saved under /tmp/qwen35_web/
    POST /api/chat    {"question": str, "media_path": str} -> NDJSON stream
                      events: {"type":"delta"|"info"|"stats"|"error"|"done", ...}
    POST /api/stop    ask the running generation to stop (checked between
                      tokens, so KV history stays valid - unlike Ctrl-C)
    POST /api/clear   clear chat history

Dependencies beyond the CLI ones: flask (pip3 install flask).
"""

import argparse
import json
import os
import re
import threading
import time
import traceback
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, request, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)

MODEL = None  # Qwen3_5 instance, created in main()
GEN_LOCK = threading.Lock()
GEN_STATE = {"running": False, "stop": False}
UPLOAD_DIR = Path("/tmp/qwen35_web")
WEBUI_HTML = Path(__file__).resolve().parent / "webui.html"


def build_model(args):
    """Import lazily so this module can be unit-tested without sail."""
    from qwen3_5 import Qwen3_5
    return Qwen3_5(args)


# ---------------------------------------------------------------------------
# streaming generation (mirrors Qwen3_5.run_round, printing -> NDJSON events)
# ---------------------------------------------------------------------------

def stream_round(question, media_path):
    m = MODEL
    m.input_str = question
    if media_path:
        media_type = m.get_media_type(media_path) or "image"
        if media_type == "image":
            messages = m.image_message(media_path)
        else:
            messages = m.video_message(media_path)
    else:
        media_type = "text"
        messages = m.text_message()

    def ev(obj):
        return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")

    try:
        inputs = m.process(messages, media_type)
        token_len = inputs.input_ids.numel()
        # seq_len - 1 for history bmodels: forward_first_with_kv asserts
        # history_length + token_len < seq_len (same rule as the CLI).
        max_input_tokens = (m.seq_len - 1) if m.support_history else m.MAX_INPUT_LENGTH
        if token_len > max_input_tokens:
            msg = "问题过长：{} > {} tokens，请缩短后重试".format(token_len, max_input_tokens)
            if media_type in ("image", "video"):
                grid = (inputs.image_grid_thw if media_type == "image"
                        else inputs.video_grid_thw).tolist()
                msg = "输入过长（grid_thw={}）：{} > {} tokens".format(grid, token_len,
                                                                     max_input_tokens)
            yield ev({"type": "error", "message": msg})
            return

        if m.support_history and ((token_len + m.history_length > m.seq_len - 128)
                                  or (m.history_length > m.PREFILL_KV_LENGTH)):
            m.clear_history()
            m.history_max_posid = 0
            yield ev({"type": "info", "message": "历史已满，自动清空后继续。"})

        first_start = time.time()
        m.forward_embed(inputs.input_ids.numpy())
        vision_sec = None
        grid_thw = None
        if media_type == "image":
            vit_start = time.time()
            m.vit_process_image(inputs)
            vision_sec = time.time() - vit_start
            grid_thw = inputs.image_grid_thw.tolist()
            position_ids = m.get_rope_index(inputs.input_ids, inputs.image_grid_thw,
                                            m.ID_IMAGE_PAD)
            m.max_posid = int(position_ids.max())
            token = m.forward_prefill(position_ids.numpy())
        elif media_type == "video":
            vit_start = time.time()
            m.vit_process_video(inputs)
            vision_sec = time.time() - vit_start
            grid_thw = inputs.video_grid_thw.tolist()
            position_ids = m.get_rope_index(inputs.input_ids, inputs.video_grid_thw,
                                            m.ID_VIDEO_PAD)
            m.max_posid = int(position_ids.max())
            token = m.forward_prefill(position_ids.numpy())
        else:
            position_ids = 3 * [i for i in range(token_len)]
            m.max_posid = token_len - 1
            token = m.forward_prefill(np.array(position_ids, dtype=np.int32))
        first_end = time.time()

        tok_num = 0
        full_word_tokens = []
        while token not in [m.ID_IM_END, m.ID_END] and m.history_length < m.seq_len:
            full_word_tokens.append(token)
            word = m.tokenizer.decode(full_word_tokens, skip_special_tokens=True)
            if "�" not in word:
                if len(full_word_tokens) == 1:
                    pre_word = word
                    word = m.tokenizer.decode([token, token],
                                              skip_special_tokens=True)[len(pre_word):]
                yield ev({"type": "delta", "text": word})
                full_word_tokens = []
            # stop is checked between tokens: KV state is consistent here, so
            # the history stays usable for the next round (unlike Ctrl-C).
            if GEN_STATE["stop"]:
                break
            m.max_posid += 1
            position_ids = np.array([m.max_posid, m.max_posid, m.max_posid],
                                    dtype=np.int32)
            token = m.forward_next(position_ids)
            tok_num += 1
        m.history_max_posid = m.max_posid + 2
        next_end = time.time()

        stats = {
            "type": "stats",
            "ftl": round(first_end - first_start, 3),
            "tps": round(tok_num / (next_end - first_end), 3) if next_end > first_end else 0.0,
            "tok_num": tok_num,
            "total_tokens": (m.history_length if m.support_history
                             else token_len + tok_num),
            "stopped": bool(GEN_STATE["stop"]),
        }
        if vision_sec is not None:
            stats["vision"] = round(vision_sec, 3)
            stats["grid_thw"] = grid_thw
        yield ev(stats)
        yield ev({"type": "done"})
    except Exception:
        # mirror the CLI: any inference failure clears history to recover
        m.clear_history()
        m.history_max_posid = 0
        yield ev({"type": "error",
                  "message": "推理失败，已清空历史：" + traceback.format_exc(limit=3)})
        yield ev({"type": "done"})


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return send_file(WEBUI_HTML)


@app.get("/api/info")
def api_info():
    m = MODEL
    return jsonify(model=os.path.basename(m.args.model_path),
                   seq_len=m.seq_len,
                   support_history=m.support_history)


@app.post("/api/upload")
def api_upload():
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify(error="没有收到文件"), 400
    # secure_filename() may strip non-ASCII names down to an extension-less
    # string (e.g. "图片.jpg" -> "jpg"), so preserve the original suffix.
    original = Path(f.filename)
    suffix = original.suffix if re.fullmatch(r"\.[A-Za-z0-9]+", original.suffix) else ""
    stem = secure_filename(original.stem) or "upload"
    stem = re.sub(r"[^\w.\-]", "_", stem).strip("._") or "upload"
    name = stem + suffix
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / "{}_{}".format(int(time.time() * 1000), name)
    f.save(dest)
    media_type = MODEL.get_media_type(str(dest))
    if media_type is None:
        dest.unlink(missing_ok=True)
        return jsonify(error="不支持的文件类型（仅图片/视频）"), 400
    return jsonify(path=str(dest), media_type=media_type, name=f.filename)


@app.post("/api/chat")
def api_chat():
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()
    media_path = (data.get("media_path") or "").strip()
    if not question:
        return jsonify(error="问题为空"), 400
    if media_path:
        if not os.path.exists(media_path):
            return jsonify(error="找不到文件：{}".format(media_path)), 400
        if MODEL.get_media_type(media_path) is None:
            return jsonify(error="不支持的文件类型：{}".format(media_path)), 400
    if not GEN_LOCK.acquire(blocking=False):
        return jsonify(error="上一轮还在生成，请先停止或等待"), 409
    GEN_STATE["stop"] = False
    GEN_STATE["running"] = True

    def gen():
        try:
            yield from stream_round(question, media_path)
        except GeneratorExit:
            # Browser refresh/close or client timeout stops the response iterator
            # without going through /api/stop. The KV cache may then represent a
            # partial round, so clear it before releasing the single-session lock.
            MODEL.clear_history()
            MODEL.history_max_posid = 0
            raise
        except Exception:  # generator-level safety net
            MODEL.clear_history()
            MODEL.history_max_posid = 0
        finally:
            GEN_STATE["running"] = False
            GEN_LOCK.release()

    return Response(gen(), mimetype="application/x-ndjson")


@app.post("/api/stop")
def api_stop():
    GEN_STATE["stop"] = True
    return jsonify(ok=True)


@app.post("/api/clear")
def api_clear():
    if GEN_STATE["running"]:
        return jsonify(error="请先停止当前生成"), 409
    MODEL.clear_history()
    MODEL.history_max_posid = 0
    return jsonify(ok=True)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='path to the bmodel file')
    parser.add_argument('-c', '--config_path', type=str, default="../config",
                        help='path to the processor file')
    parser.add_argument('-vr', '--video_ratio', type=float, default=0.25,
                        help='Set video ratio, default is 0.25')
    parser.add_argument('-d', '--devid', type=int, default=0,
                        help='device ID to use')
    parser.add_argument('-ll', '--log_level', type=str,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO",
                        help='log level, default: INFO')
    parser.add_argument('--do_sample', action='store_true',
                        help="if set, generate tokens by sample parameters")
    parser.add_argument('--host', type=str, default="0.0.0.0",
                        help='HTTP listen host, default 0.0.0.0')
    parser.add_argument('--port', type=int, default=8000,
                        help='HTTP listen port, default 8000')
    return parser


def main():
    global MODEL
    args = make_parser().parse_args()
    args_ns = args
    MODEL = build_model(args_ns)
    MODEL.args = args_ns  # api_info reads model_path from here
    print("\nWeb UI: http://{}:{}  (Ctrl-C to quit)".format(
        "localhost" if args.host in ("127.0.0.1",) else args.host, args.port))
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
