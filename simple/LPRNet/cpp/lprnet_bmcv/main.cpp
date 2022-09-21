#include <fstream>
#include <string>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include "json.hpp"
#include "lprnet.hpp"

using json = nlohmann::json;
using namespace std;

static vector<string> detect(bm_handle_t         &bm_handle,
                             LPRNET              &net,
                             vector<cv::Mat>     &images,
                             TimeStamp           *ts) {

  vector<string> detections;
  vector<bm_image> input_img_bmcv;
  ts->save("detection");
  ts->save("attach input");
  bm_image_from_mat(bm_handle, images, input_img_bmcv);
  ts->save("attach input");
  
  net.preForward(input_img_bmcv);

  // do inference
  net.forward();

  net.postForward(input_img_bmcv , detections);
  ts->save("detection");

  // destory bm_image
  for (size_t i = 0; i < input_img_bmcv.size();i++) {
    bm_image_destroy(input_img_bmcv[i]);
  }
  return detections;
}


int main(int argc, char** argv) {
  cout.setf(ios::fixed);

  // sanity check
  if (argc != 4) {
    cout << "USAGE:" << endl;
    cout << "  " << argv[0] << " <input path> <bmodel path> <device id>" << endl;
    exit(1);
  }

  struct stat info;
  string input_url = argv[1];

  string bmodel_file = argv[2];
  if (stat(bmodel_file.c_str(), &info) != 0) {
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

  // profiling
  TimeStamp lprnet_ts;
  TimeStamp *ts = &lprnet_ts;

  // initialize handle of low level device
  int max_dev_id = 0;
  bm_dev_getcount(&max_dev_id);
  if (dev_id >= max_dev_id) {
      cout << "ERROR: Input device id=" << dev_id
                << " exceeds the maximum number " << max_dev_id << endl;
      exit(-1);
  }
  bm_handle_t  bm_handle;
  bm_status_t ret = bm_dev_request (&bm_handle, dev_id);
  if (ret != BM_SUCCESS) {
    cout << "Initialize bm handle failed, ret = " << ret << endl;
    exit(-1);
  }

  // initialize LPRNET class
  LPRNET net(bm_handle , bmodel_file);

  // for profiling
  net.enableProfile(ts);
  int batch_size = net.batch_size();

  vector<cv::Mat> batch_imgs;
  vector<string> batch_names;
  int cn;
  if (stat(input_url.c_str(), &info) != 0) {
    cout << "Cannot find input image path." << endl;
    exit(1);
  }
  else if (info.st_mode & S_IFDIR){
    vector<string> files_vector;
    DIR *pDir;
    struct dirent* ptr;
    pDir = opendir(input_url.c_str());
    while((ptr = readdir(pDir))!=0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0){
            files_vector.push_back(input_url + "/" + ptr->d_name);
        }
    }
    closedir(pDir);

    std::sort(files_vector.begin(),files_vector.end());

    json result_json;
    cn = files_vector.size();
    vector<string>::iterator iter;
    ts->save("lprnet overall");
    for (iter = files_vector.begin(); iter != files_vector.end(); iter++){
      string img_file = *iter;
      //cout << img_file << endl;
      ts->save("read image");
      cv::Mat img = cv::imread(img_file, cv::IMREAD_COLOR, dev_id);
      ts->save("read image");
      size_t index = img_file.rfind("/");
      string img_name = img_file.substr(index + 1);
      batch_imgs.push_back(img);
      batch_names.push_back(img_name);
      if ((int)batch_imgs.size() == batch_size) {
        vector<string> results = detect(bm_handle, net, batch_imgs, ts);
        for(int i = 0; i < batch_size; i++){
          img_name = batch_names[i];
          cout << img_name << " pred:" << results[i].c_str() << endl;
          result_json[img_name] = results[i].c_str();
        }
        batch_imgs.clear();
        batch_names.clear();
      }
    }
    ts->save("lprnet overall");
    if (access("results", 0) != F_OK)
        mkdir("results", S_IRWXU);
    size_t index = input_url.rfind("/");
    string dataset_name = input_url.substr(index + 1);
    index = bmodel_file.rfind("/");
    string model_name = bmodel_file.substr(index + 1);
    string json_file = "results/" + model_name + "_" + dataset_name + "_bmcv_cpp" + "_result.json";
    cout << "================" << endl;
    cout << "result saved in " << json_file << endl;
    ofstream(json_file)<<std::setw(4)<<result_json;
  }else {
    cout << "Is not a valid path: " << input_url << endl;
    exit(1);
  }
  cout << "================" << endl;
  vector<time_stamp_t> t_infer = *lprnet_ts.records_["lprnet inference"];
  microseconds sum_infer(0);
  for (size_t j = 0; j < t_infer.size(); j += 2) {
    microseconds duration = duration_cast<microseconds>(t_infer[j + 1] - t_infer[j]);
    sum_infer += duration;
  }
  cout << "infer_time = " << float((sum_infer / (t_infer.size() / 2)).count())/1000/batch_size << "ms" << endl;

  vector<time_stamp_t> t_overall = *lprnet_ts.records_["lprnet overall"];
  microseconds sum_overall(0);
  for (size_t j = 0; j < t_overall.size(); j += 2) {
    microseconds duration = duration_cast<microseconds>(t_overall[j + 1] - t_overall[j]);
    sum_overall += duration;
  }
  cout << "QPS = " << cn * 1000000 / int((sum_overall / (t_overall.size() / 2)).count()) << endl;

  time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
  lprnet_ts.calbr_basetime(base_time);
  lprnet_ts.build_timeline("lprnet detect");
  lprnet_ts.show_summary("lprnet detect");
  lprnet_ts.clear();

  


  bm_dev_free(bm_handle);
  cout << endl;
  return 0;
}
