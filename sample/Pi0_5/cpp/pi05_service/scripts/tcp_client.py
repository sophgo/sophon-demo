#!/usr/bin/env python3
"""tcp_client.py — host 侧 pi0.5 SE7 TCP 推理客户端.
obs 2 图 -> JPG(q85, 4:2:0) -> TCP 发送 -> 收 [10,32] unnorm traj.
协议: [1B tidx][1B steps][8B seed][4B len0][4B len1][jpg0][jpg1] -> [1B ok][1280B f32]
"""
import io, socket, struct, time
import numpy as np
from PIL import Image

class TcpClient:
    def __init__(self, host, port=9200, quality=85):
        self.sock = socket.create_connection((host, port), timeout=120)
        self._buf = b""          # 持久接收缓冲: 一次 recv 可能带回多个字段
        self.q = quality

    def _to_jpg(self, img224_u8):
        buf = io.BytesIO()
        Image.fromarray(img224_u8).save(buf, format="JPEG", quality=self.q, subsampling=2)
        return buf.getvalue()

    def infer(self, img0_u8, img1_u8, tidx, steps, seed, noise=None):
        j0 = self._to_jpg(img0_u8)
        j1 = self._to_jpg(img1_u8)
        # noise=None 时保持原语义: 每次新噪声(对齐 C 列 serve_c 的 torch.normal 语义).
        # 2026-09-14: 新增可选 noise 注入, 供 A/B 对照固定噪声、逐位复现闭环.
        if noise is None:
            noise = np.random.RandomState().standard_normal((10, 32)).astype(np.float32)
        else:
            noise = np.ascontiguousarray(noise, dtype=np.float32)
        hdr = struct.pack("<BBqII", tidx, steps, seed, len(j0), len(j1))
        self.sock.sendall(hdr + noise.tobytes() + j0 + j1)
        ok = self._recv_exact(1)
        if ok[0] != 1:
            raise RuntimeError("server err")
        traj = self._recv_exact(10*32*4)
        return np.frombuffer(traj, dtype=np.float32).reshape(1, 10, 32).copy()

    def _recv_exact(self, n):
        """轮询式收满 n 字节, 带**持久**内部缓冲.

        实测: 直接 blocking recv(小 n) 在长请求下会被中间网络 RST(原因未明),
        改 0.5s 超时轮询 + 大缓冲稳定。
        2026-09-15: 缓冲改为实例级 —— 一次 recv 可能同时带回 ok(1B) 与
        traj(1280B), 旧实现返回 buf[:n] 后丢弃余量会导致 traj 丢失。"""
        import time as _t
        self.sock.settimeout(0.5)
        deadline = _t.time() + 300
        while len(self._buf) < n and _t.time() < deadline:
            try:
                k = self.sock.recv(1 << 20)
                if not k:
                    raise RuntimeError("conn closed at %d/%d" % (len(self._buf), n))
                self._buf += k
            except socket.timeout:
                continue
        if len(self._buf) < n:
            raise RuntimeError("recv timeout %d/%d" % (len(self._buf), n))
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

def main():
    import sys
    host = sys.argv[1] if len(sys.argv) > 1 else "172.26.166.88"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    # 用真实 obs: 从 float dat 转回 u8
    img0 = np.fromfile("/mnt/sda1/zzt/pi05-s1/devdata/realobs/loop_obs0.dat", dtype=np.float32).reshape(3,224,224)
    img1 = np.fromfile("/mnt/sda1/zzt/pi05-s1/devdata/realobs/loop_obs1.dat", dtype=np.float32).reshape(3,224,224)
    u0 = ((img0.transpose(1,2,0)+1.0)*127.5).clip(0,255).astype(np.uint8)
    u1 = ((img1.transpose(1,2,0)+1.0)*127.5).clip(0,255).astype(np.uint8)
    cli = TcpClient(host)
    # 对齐旧协议 seed=7
    t0 = time.time()
    traj = cli.infer(u0, u1, tidx=0, steps=steps, seed=7)
    t1 = time.time()
    print(f"steps={steps} 全链 {t1-t0:.3f}s  chunk0[:7]={np.round(traj[0,0,:7],4).tolist()}")
    np.save("/tmp/tcp_traj.npy", traj)
    # 连续 5 次吞吐
    ts = []
    for k in range(5):
        t0 = time.time()
        cli.infer(u0, u1, tidx=0, steps=steps, seed=7)
        ts.append(time.time()-t0)
    print("连续5次:", [f"{t:.3f}" for t in ts])
    print("TCP_CLIENT_OK")

if __name__ == "__main__":
    main()
