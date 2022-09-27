#include <condition_variable>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <sstream>
#include <chrono>
#include <mutex>
#include <thread>
#include <iomanip>
#include "ssd.hpp"
#include "utils.hpp"

using namespace std;
using time_stamp_t = time_point<steady_clock, microseconds>;

#define FRAMES 5

static void detect(SSD &net, cv::Mat &image, string name, TimeStamp *ts) {
  vector<ObjRect> detections;

  ts->save("detection");
  net.preForward(image);
  net.forward();
  net.postForward(image, detections);
  ts->save("detection");

  if (detections.size() == 0) {
    std::cout << "WARNING: Nothing was found in the specified image!" << std::endl;
  }

  for (size_t i = 0; i < detections.size(); i++) {
    ObjRect rect = detections[i];
    cv::rectangle(image, 
       cv::Rect(rect.x1, rect.y1, rect.x2 - rect.x1 + 1, rect.y2 - rect.y1 + 1), 
       cv::Scalar(255, 0, 0), 2);
  }

  // check result directory
  if (access("results", 0) != F_OK)
    mkdir("results", S_IRWXU);

  if (net.getPrecision()) {
    cv::imwrite("results/out-int8-" + name, image);
  } else {
    cv::imwrite("results/out-fp32-" + name, image);
  }
}

int main(int argc, char **argv) {

  cout.setf(ios::fixed);

  if (argc != 6) {
    cout << "USAGE:" << endl;
    cout << "  " << argv[0] << " image <image file> <bmodel file> <test count> <device id>" << endl;
    cout << "  " << argv[0] << " video <video url>  <bmodel file> <test count> <device id>" << endl;
    exit(1);
  }

  bool is_video;
  string input_url;
  string bmodel_file;
  struct stat info;
  bmodel_file = argv[3];

  if (stat(bmodel_file.c_str(), &info) != 0) {
    cout << "Cannot find valid model file." << endl;
    exit(1);
  }

  if (strcmp(argv[1], "video") != 0 && strcmp(argv[1], "image") != 0){
    cout << "mode must be image or video" << endl;
    exit(1);
  }

  is_video = false;
  if (strcmp(argv[1], "video") == 0)
    is_video = true;

  input_url = argv[2];
  if (stat(input_url.c_str(), &info) != 0) {
    cout << "Cannot find input image path." << endl;
    exit(1);
  }

  uint32_t test_loop;
  test_loop = stoull(string(argv[4]), nullptr, 0);
  if (test_loop < 1) {
    cout << "test loop must large 0." << endl;
    exit(1);
  }

  // set device id
  std::string dev_str = argv[5];
  std::stringstream checkdevid(dev_str);
  double t;
  if (!(checkdevid >> t)) {
    std::cout << "Is not a valid dev ID: " << dev_str << std::endl;
    exit(1);
  }
  int dev_id = std::stoi(dev_str);
  std::cout << "set device id: " << dev_id << std::endl;

  int max_dev_id = 0;
  bm_dev_getcount(&max_dev_id);
  if (dev_id >= max_dev_id) {
        std::cout << "ERROR: Input device id="<< dev_id
        << " exceeds the maximum number " << max_dev_id << std::endl;
        exit(-1);
  }

  TimeStamp main_ts;
  TimeStamp ssd_ts;
  TimeStamp *ts = &ssd_ts;

  SSD net(bmodel_file, dev_id);
  net.enableProfile(ts);
  if (!is_video) {
      for (uint32_t i = 0; i < test_loop; i++) {
        ts->save("ssd overall");
        ts->save("read image");
        cv::Mat img = cv::imread(input_url, cv::IMREAD_COLOR, dev_id);
        ts->save("read image");
        detect(net, img, to_string(i) + "_image.jpg", ts);
        ts->save("ssd overall");
      }
    
  } else {
    cv::VideoCapture cap(input_url, cv::CAP_ANY, dev_id);
    if (cap.isOpened()) {
      int w = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
      int h = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
      uint32_t c = 0;
      uint32_t f = 0;
      while(true) {
        if (c == test_loop) break;
        cv::Mat img;
        cap >> img;

        if (img.rows != h || img.cols != w)
          continue;

        if ((f % FRAMES) == 0) {
          detect(net, img, to_string(c) + "_dev_" + to_string(dev_id) + "_video.jpg", ts);
          c++;
        }
        f++;
      }
    }
  }

  time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
  ssd_ts.calbr_basetime(base_time);
  ssd_ts.build_timeline("ssd detect");
  ssd_ts.show_summary("detect ");
  ssd_ts.clear();

  std::cout << std::endl;

  return 0;
}
