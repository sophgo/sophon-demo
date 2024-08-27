//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-STREAM is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "decoder.h"

bool check_path(std::string file_path,
                std::vector<std::string> correct_postfixes) {
  auto index = file_path.rfind('.');
  std::string postfix = file_path.substr(index + 1);
  if (find(correct_postfixes.begin(), correct_postfixes.end(), postfix) !=
      correct_postfixes.end()) {
    return true;
  } else {
    return false;
  }
};

void getAllFiles(std::string path, std::vector<std::string>& files,
                 std::vector<std::string> correct_postfixes) {
  DIR* dir;
  struct dirent* ptr;
  if ((dir = opendir(path.c_str())) == NULL) {
    perror("Open dri error...");
    exit(1);
  }
  while ((ptr = readdir(dir)) != NULL) {
    if (std::strcmp(ptr->d_name, ".") == 0 ||
        std::strcmp(ptr->d_name, "..") == 0)
      continue;
    else if (ptr->d_type == 8 &&
             check_path(path + "/" + ptr->d_name, correct_postfixes))  // file
      files.push_back(path + "/" + ptr->d_name);
    else if (ptr->d_type == 10)  // link file
      continue;
    else if (ptr->d_type == 4) {
      // files.push_back(ptr->d_name);//dir
      getAllFiles(path + "/" + ptr->d_name, files, correct_postfixes);
    }
  }
  closedir(dir);
}

void bm_image2Frame(std::shared_ptr<Frame>& f, bm_image& img) {
  f->mWidth = img.width;
  f->mHeight = img.height;
  f->mDataType = img.data_type;
  f->mFormatType = img.image_format;
  f->mChannel = 3;
  f->mDataSize = img.width * img.height * f->mChannel * sizeof(uchar);
}

Decoder::Decoder() {}

Decoder::~Decoder() {
  for (auto& thread : threads) {
    if (thread.joinable()) thread.join();
  }
}

int Decoder::init(const std::string& json) {
  decoders.resize(2);

  auto configure = nlohmann::json::parse(json, nullptr, false);
  if (!configure.is_object()) {
    return -1;
  }

  int id = 0;
  for (const auto& channel : configure["channels"]) {
    mUrl = channel["url"];
    int ret = bm_dev_request(&m_handle, mDeviceId);
    assert(BM_SUCCESS == ret);
    ret = decoders[id++].openDec(&m_handle, mUrl.c_str());
    if (ret < 0) {
      return -1;
    }
  }

  // std::vector<std::thread> threads;
  for (int i = 0; i < decoders.size(); i++) {
    threads.emplace_back(&Decoder::start, this, i);
  }
  return 1;
}

int Decoder::process(std::shared_ptr<Frame>& mFrame, int decode_id) {
  int frame_id = 0;
  int eof = 0;
  std::shared_ptr<bm_image> spBmImage = nullptr;
  int64_t pts = 0;

  {  // 在所有线程等待执行decoder.grab前添加一个等待点
    std::unique_lock<std::mutex> lock(decoder_mutex);
    numThreadsReady++;
    if (numThreadsReady == numThreadsTotal) {
      numThreadsReady = 0;
      lock.unlock();
      decoder_cv.notify_all();
    } else {
      decoder_cv.wait(lock);
    }
  }
  spBmImage = decoders[decode_id].grab(frame_id, eof, pts, mSampleInterval);

  mFrame->mHandle = m_handle;
  mFrame->mFrameId = frame_id;
  mFrame->mSpData = spBmImage;
  mFrame->mTimestamp = pts;
  if (eof) {
    mFrame->mEndOfStream = true;

  } else {
    if (spBmImage != nullptr) bm_image2Frame(mFrame, *spBmImage);
  }

  return 1;
}

void Decoder::uninit() {}

void Decoder::start(int decoder_id) {
  while (true) {
    std::shared_ptr<Frame> mframe = std::make_shared<Frame>();
    process(mframe, decoder_id);
    if (mframe->mEndOfStream) break;
    {
      while (output_frames->frames[decoder_id].size() == 20) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(100));  // 可以添加短暂延迟以避免忙等待
      }
      std::unique_lock<std::mutex> lock(*output_queue_lock);
      output_frames->frames[decoder_id].push(std::move(mframe));
      std::cout << "decode_size" << output_frames->frames[decoder_id].size()
                << std::endl;
    }
  }
  return;
}