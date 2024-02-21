QT       += core gui
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets
TARGET = yolov5_app
TEMPLATE = app
DEFINES += QT_DEPRECATED_WARNINGS USE_BMCV USE_FFMPEG USE_OPENCV

CONFIG += debug
CONFIG += soc

contains(CONFIG, pcie){
        message("pcie mode")
        INCLUDEPATH += /opt/sophon/libsophon-current/include
        INCLUDEPATH += /opt/sophon/sophon-ffmpeg-latest/include
        INCLUDEPATH += /opt/sophon/sophon-opencv-latest/include/opencv4
        INCLUDEPATH += /opt/sophon/sophon-opencv-latest/include/opencv4/opencv2
        LIBS += -lbmlib -lbmrt -lavcodec -lavformat -lavutil -lbmcv -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs 
        LIBS += -L/opt/sophon/libsophon-current/lib
        LIBS += -L/opt/sophon/sophon-ffmpeg-latest/lib
        LIBS += -L/opt/sophon/sophon-opencv-latest/lib
}else{
        message("soc mode")

        SOC_SDK_PATH = /home/linaro/sdk/v23.10.01/soc-sdk
        LIBSOPHON_DIR_PATH = $$SOC_SDK_PATH/libsophon-0.5.0
        FFMPEG_DIR_PATH = $$SOC_SDK_PATH/sophon-ffmpeg_0.7.3
        OPENCV_DIR_PATH = $$SOC_SDK_PATH/sophon-opencv_0.7.3

        INCLUDEPATH += $$LIBSOPHON_DIR_PATH/include
        INCLUDEPATH += $$FFMPEG_DIR_PATH/include
        INCLUDEPATH += $$OPENCV_DIR_PATH/include/opencv4
        INCLUDEPATH += $$OPENCV_DIR_PATH/include/opencv4/opencv2
        LIBS += -lbmlib -lbmrt -lavcodec -lavformat -lavutil -lbmcv -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -Wl,-rpath=./:$$LIBSOPHON_DIR_PATH/lib:$$FFMPEG_DIR_PATH/lib:$$OPENCV_DIR_PATH/lib
        LIBS += -L$$LIBSOPHON_DIR_PATH/lib
        LIBS += -L$$FFMPEG_DIR_PATH/lib
        LIBS += -L$$OPENCV_DIR_PATH/lib

}



INCLUDEPATH += $$PWD/include \
        $$PWD/include/ff_codec \
        $$PWD/include/qtgui \
        $$PWD/include/common
HEADERS += $$PWD/include/*.h $$PWD/include/ff_codec/*.h $$PWD/include/qtgui/*.h $$PWD/include/common/*.h
SOURCES += $$PWD/src/*.cpp $$PWD/src/ff_codec/*.cpp $$PWD/src/qtgui/*.cpp $$PWD/src/common/*.cpp


