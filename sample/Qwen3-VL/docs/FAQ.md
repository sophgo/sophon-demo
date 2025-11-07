# FAQ

## SE7安装python3.10

之前的方法无法实现了，必须手动编译

```bash

sudo apt update
sudo apt install build-essential checkinstall zlib1g-dev libncurses5-dev libncursesw5-dev libsqlite3-dev liblzma-dev tk-dev uuid-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libbz2-dev wget curl -y
wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz 
tar -xaf Python-3.10.12.tgz
cd Python-3.10.12/
ls
./configure --enable-optimizations --enable-shared --prefix=/usr 
make -j4
sudo make altinstall

# 创建虚拟环境 py310env
cd /data
python3.10 -m venv py310env
source py310env/bin/activate

```
