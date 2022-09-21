#include <boost/filesystem.hpp>
#include <sys/types.h>
#include <sys/stat.h>
#include <sstream>
#include "ssd.hpp"

namespace fs = boost::filesystem;
using namespace std;

static void detect(bm_handle_t         &bm_handle,
                   SSD                 &net,
                   cv::Mat             &image,
                   string              name,
                   TimeStamp           *ts) {

  vector<vector<ObjRect>> detections;
  vector<cv::Mat> images;
  images.push_back (image);

  vector<bm_image> input_img_bmcv;
  ts->save("attach input");
  bm_image_from_mat(bm_handle, images, input_img_bmcv);
  ts->save("attach input");

  ts->save("detection");
  net.preForward (input_img_bmcv);

  // do inference
  net.forward();

  net.postForward (input_img_bmcv , detections);
  ts->save("detection");
  // destory bm_image
  for (size_t i = 0; i < input_img_bmcv.size();i++) {
    bm_image_destroy (input_img_bmcv[i]);
  }

  // save results to jpg file
  for (size_t i = 0; i < detections.size(); i++) {
    for (size_t j = 0; j < detections[i].size(); j++) {
      ObjRect rect = detections[i][j];
      cv::rectangle(image, cv::Rect(rect.x1, rect.y1, rect.x2 - rect.x1 + 1,
                                    rect.y2 - rect.y1 + 1), cv::Scalar(255, 0, 0), 2);
    }

    // check result directory
    if (!fs::exists("./results")) {
	    fs::create_directory("results");
    }
    std::cout<<"write"<<std::endl;
    // jpg encode
    if (net.getPrecision()) {
      cv::imwrite("results/out-batch-int8-" + name, image);
    } else {
      cv::imwrite("results/out-batch-fp32-" + name, image);
    }
  }
}

int main(int argc, char **argv) {

  cout.setf(ios::fixed);

  // sanity check
  if (argc != 6) {
    cout << "USAGE:" << endl;
    cout << "  " << argv[0] << " image <image file> <bmodel path> <test count> <device id>" << endl;
    cout << "  " << argv[0] << " video <video url>  <bmodel path> <test count> <device id>" << endl;
    exit(1);
  }

  string bmodel_file = argv[3];
  if (!fs::exists(bmodel_file)) {
    cout << "Cannot find valid model file." << endl;
    exit(1);
  }
  
  if (strcmp(argv[1], "video") != 0 && strcmp(argv[1], "image") != 0){
    cout << "mode must be image or video" << endl;
    exit(1);
  }

  bool is_video = false;
  if (strcmp(argv[1], "video") == 0)
    is_video = true;

  string input_url = argv[2];
  if (!is_video && !fs::exists(input_url)) {
    cout << "Cannot find input image file." << endl;
    exit(1);
  }

  unsigned long test_loop = stoul(string(argv[4]), nullptr, 0);
  if (test_loop < 1) {
    std::cout << "test_loop must large 0" << std::endl;
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
  std::cout << "set device id:"  << dev_id << std::endl;

  // profiling
  TimeStamp ssd_ts;
  TimeStamp *ts = &ssd_ts;

  // initialize handle of low level device
    int max_dev_id = 0;
    bm_dev_getcount(&max_dev_id);
    if (dev_id >= max_dev_id) {
        std::cout << "ERROR: Input device id=" << dev_id
                  << " exceeds the maximum number " << max_dev_id << std::endl;
        exit(-1);
    }
    bm_handle_t  bm_handle;
  bm_status_t ret = bm_dev_request (&bm_handle, dev_id);
  if (ret != BM_SUCCESS) {
    cout << "Initialize bm handle failed, ret = " << ret << endl;
    exit(-1);
  }

  // initialize SSD class
  SSD net(bm_handle , bmodel_file);

  // for profiling
  net.enableProfile(ts);

  // decode and detect
  if (!is_video) {

    fs::path image_file(input_url);
    string name = image_file.filename().string();
    for (uint32_t i = 0; i < test_loop; i++) {
      ts->save("ssd overall");
      ts->save("read image");

      // decode jpg file to Mat object
      cv::Mat img = cv::imread(input_url, cv::IMREAD_COLOR, dev_id);
      //cv::Mat img;
      //cv::resize(input,img,cv::Size(300,300));
      ts->save("read image");

      // do detect
      string img_out = "t_" + to_string(i) + "_dev_" + to_string(dev_id) + "_" +name;
      detect(bm_handle, net, img, img_out, ts);
      ts->save("ssd overall");
    }

  } else {

    // open stream
    cv::VideoCapture cap(input_url, cv::CAP_ANY, dev_id);
    if (!cap.isOpened()) {
      cout << "open stream " << input_url << " failed!" << endl;
      exit(1);
    }

    // get resolution
    int w = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int h = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    cout << "resolution of input stream: " << h << "," << w << endl;
    // set output format to YUVi420
    cap.set(cv::CAP_PROP_OUTPUT_YUV, 1.0);

    for (uint32_t c = 0; c < test_loop; c++) {

      // get one frame from decoder
      cv::Mat *p_img = new cv::Mat;
      cap.read(*p_img);

      // sanity check
      if (p_img->avRows() != h || p_img->avCols() != w) {
        if (p_img != nullptr) delete p_img;
        continue;
      }
      //cv::Mat *input;

      //cv::resize(*p_img,*input,cv::Size(300,300));
      // do detct
      string img_out = "t_" + to_string(c) + "_dev_" + to_string(dev_id)  + "_video.jpg";
      detect(bm_handle, net, *p_img, img_out, ts);
      // release Mat object
      if (p_img != nullptr) delete p_img;
    }
  }

  time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
  ssd_ts.calbr_basetime(base_time);
  ssd_ts.build_timeline("ssd detect");
  ssd_ts.show_summary("detect ");
  ssd_ts.clear();

  bm_dev_free(bm_handle);

  cout << endl;

  return 0;
}
