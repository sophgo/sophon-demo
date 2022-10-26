
import os
##请修改为您对应的预测结果文件
source_txt = 'retinaface_mobilenet0.25_fp32_1b.bmodel_opencv_WIDERVAL_python_result.txt'

with open(source_txt, 'r') as f:
    lines = f.readlines()
    i = 0
    while i < len(lines):
        filename = lines[i]
        filename = filename.split('/')[-1].split('.')[0]
        no = filename.split('_')[0] + '--'
        for dir in os.listdir('widerface_txt'):
            if dir.startswith(no):
                no = dir
                break
        number = lines[i+1]
        contents = lines[i+2: i+2+int(number)]
        with open(os.path.join('widerface_txt', no, filename + '.txt'), 'w') as w:
            w.writelines([filename+'\n', number] + contents)
        i += 2 + int(number)
