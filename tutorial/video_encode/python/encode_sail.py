import sophon.sail as sail
import argparse
import time 

# 图片编解码例程
def picture_decode_encode(handle:sail.Handle, decoder:sail.Decoder, encoder:sail.Encoder,bmcv:sail.Bmcv):
    # 初始化bm_image
    img = sail.BMImage()
    # decode
    decode_ret = decoder.read(handle, img)
    if decode_ret != 0:
        print("image decode failed!")
        return -1
    
    ## you can use bmcv to do something here
    time.sleep(0.01)

    # encode
    img_data = encoder.pic_encode(".jpg", img)

    # imwrite
    with open('picture_encode.jpg', mode='wb+') as f:
        f.writelines(img_data)

    return 0

# 视频推流
def video_push_stream(handle:sail.Handle, decoder:sail.Decoder, encoder:sail.Encoder,bmcv:sail.Bmcv):
    # 循环检测
    count = 0
    while(True):
        # 初始化bm_image
        img = sail.BMImage()
        # decode
        decode_ret = decoder.read(handle, img)
        count += 1
        if(decode_ret!=0):
            decoder.reconnect()
            continue
        
        ## you can use bmcv to do something here
        time.sleep(0.01)

        ## encode   (push rtsp/rtmp stream or write local video file)
        encode_ret = encoder.video_write(img)
        while(encode_ret != 0):
            time.sleep(0.01)
            encode_ret = encoder.video_write(img)



def is_image(filename):
    image_extensions = ['.jpg', '.png', '.bmp']
    return any(filename.lower().endswith(ext) for ext in image_extensions)

if __name__ == "__main__":
    # 参数解析
    parse = argparse.ArgumentParser(description="Demo for media basic interface/push_stream.py")
    parse.add_argument('--input_path', default="../datasets/test_car_person_1080P.mp4", type=str, help="Path or rtsp url to the video/image file.")#文件夹所在目录
    parse.add_argument('--output_path', type=str,help="Local file path or stream url")
    parse.add_argument('--device_id', default=0, type=int,help="Device id")   
    parse.add_argument('--compressed_nv12', default=True,type=bool,help="Whether the format of decoded output is compressed NV12.")
    parse.add_argument('--height', default=1080, type=int, help="The height of the encoded video")
    parse.add_argument('--width', default=1920, type=int, help="The width of the encoded video")
    parse.add_argument('--enc_fmt', default='h264_bm', type=str, help="encoded video format, h264_bm/hevc_bm")
    parse.add_argument('--bitrate', default=2000, type=int, help="encoded bitrate")
    parse.add_argument('--pix_fmt', default='NV12', type=str, help="encoded pixel format")
    parse.add_argument('--gop', default=32, type=int, help="gop size")
    parse.add_argument('--gop_preset', default=2, type=int, help="gop_preset")
    parse.add_argument('--framerate', default=25, type=int, help="encode frame rate")
    opt = parse.parse_args()

    # 初始化硬件
    handle = sail.Handle(opt.device_id)

    # 初始化bmcv的控制指令
    bmcv = sail.Bmcv(handle)

    # 初始化解码器
    decoder = sail.Decoder(opt.input_path, opt.compressed_nv12, opt.device_id)

    # 初始化编码器,开始推流测试
    if is_image(opt.input_path):
        encoder = sail.Encoder()
        picture_decode_encode(handle,decoder,encoder,bmcv)

    else:
        enc_params = f"width={opt.width}:height={opt.height}:bitrate={opt.bitrate}:gop={opt.gop}:gop_preset={opt.gop_preset}:framerate={opt.framerate}"
        encoder = sail.Encoder(opt.output_path, opt.device_id, opt.enc_fmt, opt.pix_fmt, enc_params, 10)
        video_push_stream(handle,decoder,encoder,bmcv)

    # 资源释放
    decoder.release()
    encoder.release()

