#include <boost/filesystem.hpp>
#include <sys/types.h>
#include <sys/stat.h>
#include <sstream>
#include "resnet.hpp"
#include <string>
#include "opencv2/opencv.hpp"
#include <jsoncpp/json/json.h>

namespace fs = boost::filesystem;
using namespace std;

static vector<pair<int, float>> infer(RESNET              &net,
                             vector<cv::Mat>     &images,
                             TimeStamp           *ts) {

  vector<pair<int, float>> results;

  ts->save("infer");
  net.preForward(images);

  // do inference
  net.forward();

  net.postForward(results);
  ts->save("infer");

  return results;
}


int main(int argc, char** argv) {
  //system("chcp 65001");
  cout.setf(ios::fixed);

  // sanity check
  if (argc != 4) {
    cout << "USAGE:" << endl;
    cout << "  " << argv[0] << " <input path> <bmodel path> <device id>" << endl;
    exit(1);
  }

  string input_url = argv[1];
  if (!fs::exists(input_url)) {
    cout << "Cannot find input image path." << endl;
    exit(1);
  }

  string bmodel_file = argv[2];
  if (!fs::exists(bmodel_file)) {
    cout << "Cannot find valid model file." << endl;
    exit(1);
  }

  // set device id
  string dev_str = argv[3];
  stringstream checkdevid(dev_str);
  double t;
  if (!(checkdevid >> t)) {
    cout << "Is not a valid dev ID: " << dev_str << endl;
    exit(1);
  }
  int dev_id = stoi(dev_str);
  cout << "set device id:"  << dev_id << endl;

  // initialize handle of low level device
  int max_dev_id = 0;
  bm_dev_getcount(&max_dev_id);
  if (dev_id >= max_dev_id) {
      cout << "ERROR: Input device id=" << dev_id
                << " exceeds the maximum number " << max_dev_id << endl;
      exit(-1);
  }
  // profiling
  TimeStamp resnet_ts;
  TimeStamp *ts = &resnet_ts;

  // initialize RESNET class
  RESNET net(bmodel_file, dev_id);

  // for profiling
  net.enableProfile(ts);
  int batch_size = net.batch_size();
  vector<cv::Mat> batch_imgs;
  vector<string> batch_names;
  
  if (fs::is_regular_file(input_url)){
    if (batch_size != 1){
      cout << "ERROR: batch_size of model is  " << batch_size << endl;
      exit(-1);
    }
    ts->save("resnet overall");
    ts->save("read image");
    // decode jpg file to Mat object
    cv::Mat img = cv::imread(input_url);
    ts->save("read image");
    size_t index = input_url.rfind("/");
    string img_name = input_url.substr(index + 1);
    batch_imgs.push_back(img);
    batch_names.push_back(img_name);
    // do infer
    vector<pair<int, float>> results = infer(net, batch_imgs, ts);
    ts->save("resnet overall");
    // output results
    cout << input_url << " pred: " << results[0].first <<", score:" << results[0].second << endl;
  }
  else if (fs::is_directory(input_url)){
    fs::recursive_directory_iterator beg_iter(input_url);
    fs::recursive_directory_iterator end_iter;
    vector<string> files_vector;
    for (; beg_iter != end_iter; ++beg_iter){
      if (fs::is_directory(*beg_iter)) continue;
      else {
        string img_file = beg_iter->path().string();
        files_vector.push_back(img_file);
      }
    }
    std::sort(files_vector.begin(),files_vector.end());
    
    map<string, pair<int, float>> d_result;
    vector<string>::iterator iter;
    ts->save("resnet overall");
    for (iter = files_vector.begin(); iter != files_vector.end(); iter++){
      string img_file = *iter;
      ts->save("read image");
      cv::Mat img = cv::imread(img_file, cv::IMREAD_COLOR, dev_id);
      ts->save("read image");
      size_t index = img_file.rfind("/");
      string img_name = img_file.substr(index + 1);
      batch_imgs.push_back(img);
      batch_names.push_back(img_name);
      if ((int)batch_imgs.size() == batch_size) {
        vector<pair<int,float>> results = infer(net, batch_imgs, ts);
        for(int i = 0; i < batch_size; i++){
          fs::path img_name(batch_names[i]);
          cout << img_name.string() << " pred: " << results[i].first <<", score:" << results[i].second << endl;
          d_result[img_name.string()] = results[i];
        }
        batch_imgs.clear();
        batch_names.clear();
      }        
    }
    ts->save("resnet overall");
    if (!fs::exists("results")) {
      fs::create_directories("results");
    }
    size_t index = input_url.rfind("/");
    string dataset_name = input_url.substr(index + 1);
    index = bmodel_file.rfind("/");
    string model_name = bmodel_file.substr(index + 1);
    string result_file = "results/" + model_name + "_" + dataset_name + "_opencv_cpp" + "_result.txt";
    cout << "================" << endl;
    cout << "result saved in " << result_file << endl;
    ofstream desFile(result_file, ios::out);
    for (auto result : d_result) {
      desFile << result.first.c_str() <<"\t" << result.second.first << "\t" << result.second.second << endl;

    }
    desFile.close();
  }else {
    cout << "Is not a valid path: " << input_url << endl;
    exit(1);
  }
  
  time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
  resnet_ts.calbr_basetime(base_time);
  resnet_ts.build_timeline("resnet infer");
  resnet_ts.show_summary("resnet infer");
  resnet_ts.clear();
  return 0;
}
