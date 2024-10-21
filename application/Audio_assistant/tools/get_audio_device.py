import pyaudio
 
# 创建一个PyAudio对象
p = pyaudio.PyAudio()
 
# 打印所有可用的音频输入和输出设备
print('音频输入设备:')
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f"detail: {info}")
        print(f"Index: {i}, Name: {info['name']}, Channels: {info['maxInputChannels']}")
 
print('\n音频输出设备:')
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxOutputChannels'] > 0:
        print(f"detail: {info}")
        print(f"Index: {i}, Name: {info['name']}, Channels: {info['maxOutputChannels']}")
 
# 关闭PyAudio对象
p.terminate()