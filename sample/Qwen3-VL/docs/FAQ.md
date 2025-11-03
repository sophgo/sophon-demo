# FAQ

## SE7安装python3.10
之前的方法无法实现了，必须手动编译
```bash
sudo apt install -y libbz2-dev
sudo apt-get purge python3.8 python3.8-* -y
sudo apt install -y liblzma-dev -y
sudo apt-get install -y --no-install-recommends libssl-dev curl wget
sudo apt install -y build-essential checkinstall zlib1g-dev libncurses5 libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev curl 
sudo apt-get update 
sudo apt-get install -y build-essential wget curl libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libffi-dev liblzma-dev tk-dev 
sudo wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz 
sudo tar -xvf Python-3.10.12.tgz 
cd Python-3.10.12 
sudo ./configure --enable-optimizations --enable-shared --prefix=/usr 
sudo make -j$(nproc) altinstall 
sudo mkdir -p /usr/lib/aarch64-linux-gnu 
sudo ln -sf /usr/local/lib/libpython3.10.so.1.0 /usr/lib/aarch64-linux-gnu/libpython3.10.so.1.0 
sudo ln -sf /usr/local/lib/libpython3.10.so /usr/lib/aarch64-linux-gnu/libpython3.10.so 
sudo ldconfig 
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2 
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.10 2 
sudo update-alternatives --set python3 /usr/bin/python3.10 
sudo update-alternatives --set python /usr/bin/python3.10 
sudo curl -sS https://bootstrap.pypa.io/get-pip.py | python3 
sudo ln -sf /usr/bin/pip3.10 /usr/bin/pip3 
sudo ln -sf /usr/bin/pip3.10 /usr/bin/pip 

```