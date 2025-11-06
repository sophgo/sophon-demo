# FAQ

## SE7安装python3.10
之前的方法无法实现了，必须手动编译
```bash
sudo apt install -y build-essential checkinstall zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev curl
wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz
tar -xvf Python-3.10.0.tgz
cd Python-3.10.0
./configure --enable-optimizations
sudo make altinstall

cd /data
# 创建名为myenv的虚拟环境
python3.10 -m venv myenv

# 进入虚拟环境
source myenv/bin/activate

```