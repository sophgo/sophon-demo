#ifndef RetinaFace_Post_H__
#define RetinaFace_Post_H__

#include <iostream>
#include <vector>
#include <math.h>
#include <map>
#include "bmodel_base.hpp"
#include "bmruntime_interface.h"

typedef struct _tag__stFaceRect{
  int top;
  int bottom;
  int left;
  int right;
  float points_x[5];
  float points_y[5];
  float score;
} stFaceRect;

struct anchor_win {
  float x_ctr;
  float y_ctr;
  float w;
  float h;
};

struct anchor_box {
  float x1;
  float y1;
  float x2;
  float y2;
};

struct FacePts {
  float x[5];
  float y[5];
};

struct FaceDetectInfo {
  float score;
  anchor_box rect;
  FacePts pts;
};

struct anchor_cfg {
public:
  int STRIDE;
  std::vector<int> SCALES;
  int BASE_SIZE;
  std::vector<float> RATIOS;
  int ALLOWED_BORDER;

  anchor_cfg() {
    STRIDE = 0;
    SCALES.clear();
    BASE_SIZE = 0;
    RATIOS.clear();
    ALLOWED_BORDER = 0;
  }
};

class RetinaFacePostProcess : public BmodelBase {
public:
  RetinaFacePostProcess(std::string network = "net3", float nms = 0.4);
  ~RetinaFacePostProcess();

  void run(const bm_net_info_t& net_info, float** preds, std::vector<stFaceRect>& results,
   int img_h, int img_w, float ratio_, int max_face_count = 20, float threshold = 0.9, float scales = 1.0);

private:
  std::vector<FaceDetectInfo> postProcess(int inputW, int inputH, float threshold);
  anchor_box bbox_pred(anchor_box anchor, std::vector<float> regress);
  std::vector<anchor_box> bbox_pred(std::vector<anchor_box> anchors, std::vector<std::vector<float> >regress);
  std::vector<FacePts> landmark_pred(std::vector<anchor_box> anchors, std::vector<FacePts> facePts);
  FacePts landmark_pred(anchor_box anchor, FacePts facePt);
  static bool CompareBBox(const FaceDetectInfo &a, const FaceDetectInfo &b);
  std::vector<FaceDetectInfo> nms(std::vector<FaceDetectInfo> &bboxes, float threshold);
private:
  std::string network;
  float nms_threshold;
  std::vector<float> _ratio;
  std::vector<anchor_cfg> cfg;

  std::vector<int> _feat_stride_fpn;
  std::map<std::string, std::vector<anchor_box> > _anchors_fpn;
  std::map<std::string, std::vector<anchor_box> > _anchors;
  std::map<std::string, int> _num_anchors;

};

#endif // RetinaFace_Post_H__
