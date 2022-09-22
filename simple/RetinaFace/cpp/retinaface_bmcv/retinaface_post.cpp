#include <algorithm>
#include <assert.h>
#include "retinaface_post.hpp"

using namespace std;


anchor_win  _whctrs(anchor_box anchor) {
  anchor_win win;
  win.w = anchor.x2 - anchor.x1 + 1;
  win.h = anchor.y2 - anchor.y1 + 1;
  win.x_ctr = anchor.x1 + 0.5 * (win.w - 1);
  win.y_ctr = anchor.y1 + 0.5 * (win.h - 1);

  return win;
}

anchor_box _mkanchors(anchor_win win) {
  anchor_box anchor;
  anchor.x1 = win.x_ctr - 0.5 * (win.w - 1);
  anchor.y1 = win.y_ctr - 0.5 * (win.h - 1);
  anchor.x2 = win.x_ctr + 0.5 * (win.w - 1);
  anchor.y2 = win.y_ctr + 0.5 * (win.h - 1);

  return anchor;
}

vector<anchor_box> _ratio_enum(anchor_box anchor, vector<float> ratios) {
  vector<anchor_box> anchors;
  for(size_t i = 0; i < ratios.size(); i++) {
    anchor_win win = _whctrs(anchor);
    float size = win.w * win.h;
    float scale = size / ratios[i];

    win.w = round(sqrt(scale));
    win.h = round(win.w * ratios[i]);

    anchor_box tmp = _mkanchors(win);
    anchors.push_back(tmp);
  }
  return anchors;
}

vector<anchor_box> _scale_enum(anchor_box anchor, vector<int> scales) {
  //Enumerate a set of anchors for each scale wrt an anchor.
  vector<anchor_box> anchors;
  for(size_t i = 0; i < scales.size(); i++) {
    anchor_win win = _whctrs(anchor);

    win.w = win.w * scales[i];
    win.h = win.h * scales[i];

    anchor_box tmp = _mkanchors(win);
    anchors.push_back(tmp);
  }

  return anchors;
}

vector<anchor_box> generate_anchors(int base_size = 16, vector<float> ratios = {0.5, 1, 2},
  vector<int> scales = {8, 64}, int stride = 16, bool dense_anchor = false) {
  anchor_box base_anchor;
  base_anchor.x1 = 0;
  base_anchor.y1 = 0;
  base_anchor.x2 = base_size - 1;
  base_anchor.y2 = base_size - 1;

  vector<anchor_box> ratio_anchors;
  ratio_anchors = _ratio_enum(base_anchor, ratios);

  vector<anchor_box> anchors;
  for(size_t i = 0; i < ratio_anchors.size(); i++) {
    vector<anchor_box> tmp = _scale_enum(ratio_anchors[i], scales);
    anchors.insert(anchors.end(), tmp.begin(), tmp.end());
  }

  if(dense_anchor) {
  assert(stride % 2 == 0);
  vector<anchor_box> anchors2 = anchors;
  for(size_t i = 0; i < anchors2.size(); i++) {
    anchors2[i].x1 += stride / 2;
    anchors2[i].y1 += stride / 2;
    anchors2[i].x2 += stride / 2;
    anchors2[i].y2 += stride / 2;
  }
  anchors.insert(anchors.end(), anchors2.begin(), anchors2.end());
  }

  return anchors;
}

vector<vector<anchor_box>> generate_anchors_fpn(bool dense_anchor = false, vector<anchor_cfg> cfg = {}) {
  vector<vector<anchor_box>> anchors;
  for(size_t i = 0; i < cfg.size(); i++) {
    anchor_cfg tmp = cfg[i];
    int bs = tmp.BASE_SIZE;
    vector<float> ratios = tmp.RATIOS;
    vector<int> scales = tmp.SCALES;
    int stride = tmp.STRIDE;

    vector<anchor_box> r = generate_anchors(bs, ratios, scales, stride, dense_anchor);
    anchors.push_back(r);
  }

  return anchors;
}

vector<anchor_box> anchors_plane(int height, int width, int stride, vector<anchor_box> base_anchors) {
  vector<anchor_box> all_anchors;
  for(size_t k = 0; k < base_anchors.size(); k++) {
    for(int ih = 0; ih < height; ih++) {
      int sh = ih * stride;
      for(int iw = 0; iw < width; iw++) {
        int sw = iw * stride;
        anchor_box tmp;
        tmp.x1 = base_anchors[k].x1 + sw;
        tmp.y1 = base_anchors[k].y1 + sh;
        tmp.x2 = base_anchors[k].x2 + sw;
        tmp.y2 = base_anchors[k].y2 + sh;
        all_anchors.push_back(tmp);
      }
    }
  }
  return all_anchors;
}

void clip_boxes(vector<anchor_box> &boxes, int width, int height) {
  for(size_t i = 0; i < boxes.size(); i++) {
    if(boxes[i].x1 < 0) {
      boxes[i].x1 = 0;
    }
    if(boxes[i].y1 < 0) {
      boxes[i].y1 = 0;
    }
    if (boxes[i].x2 > width - 1) {
      boxes[i].x2 = width - 1;
    }
    if(boxes[i].y2 > height - 1) {
      boxes[i].y2 = height -1;
    }
  }
}

void clip_boxes(anchor_box &box, int width, int height) {
  if(box.x1 < 0) {
    box.x1 = 0;
  }
  if(box.y1 < 0) {
    box.y1 = 0;
  }
  if(box.x2 > width - 1) {
    box.x2 = width - 1;
  }
  if(box.y2 > height - 1) {
    box.y2 = height -1;
  }
}

RetinaFacePostProcess::RetinaFacePostProcess(string network, float nms)
  : network(network), nms_threshold(nms) {
  int fmc = 3;
  if(network == "net3") {
    _ratio = {1.0};
  }
  else if(network == "net3a") {
    _ratio = {1.0, 1.5};
  }
  else if(network == "net6") {
    fmc = 6;
  }
  else if(network == "net5") {
    fmc = 5;
  }
  else if(network == "net5a") {
    fmc = 5;
    _ratio = {1.0, 1.5};
  }
  else if(network == "net4") {
    fmc = 4;
  }
  else if(network == "net5a") {
    fmc = 4;
    _ratio = {1.0, 1.5};
  }
  else {
    std::cout << "network setting error" << network << std::endl;
  }

  if(fmc == 3) {
    _feat_stride_fpn = {32, 16, 8};
    anchor_cfg tmp;
    tmp.SCALES = {32, 16};
    tmp.BASE_SIZE = 16;
    tmp.RATIOS = _ratio;
    tmp.ALLOWED_BORDER = 9999;
    tmp.STRIDE = 32;
    cfg.push_back(tmp);

    tmp.SCALES = {8, 4};
    tmp.BASE_SIZE = 16;
    tmp.RATIOS = _ratio;
    tmp.ALLOWED_BORDER = 9999;
    tmp.STRIDE = 16;
    cfg.push_back(tmp);

    tmp.SCALES = {2, 1};
    tmp.BASE_SIZE = 16;
    tmp.RATIOS = _ratio;
    tmp.ALLOWED_BORDER = 9999;
    tmp.STRIDE = 8;
    cfg.push_back(tmp);
  } else {
    std::cout << "please reconfig anchor_cfg" << network << std::endl;
  }

  bool dense_anchor = false;
  vector<vector<anchor_box>> anchors_fpn = generate_anchors_fpn(dense_anchor, cfg);
  for(size_t i = 0; i < anchors_fpn.size(); i++) {
    string key = "stride" + std::to_string(_feat_stride_fpn[i]);
    _anchors_fpn[key] = anchors_fpn[i];
    _num_anchors[key] = static_cast<int>(anchors_fpn[i].size());
  }
}

RetinaFacePostProcess::~RetinaFacePostProcess() {
}

vector<anchor_box> RetinaFacePostProcess::bbox_pred(vector<anchor_box> anchors, vector<vector<float> > regress) {
  vector<anchor_box> rects(anchors.size());
  for(size_t i = 0; i < anchors.size(); i++) {
    float width = anchors[i].x2 - anchors[i].x1 + 1;
    float height = anchors[i].y2 - anchors[i].y1 + 1;
    float ctr_x = anchors[i].x1 + 0.5 * (width - 1.0);
    float ctr_y = anchors[i].y1 + 0.5 * (height - 1.0);

    float pred_ctr_x = regress[i][0] * width + ctr_x;
    float pred_ctr_y = regress[i][1] * height + ctr_y;
    float pred_w = exp(regress[i][2]) * width;
    float pred_h = exp(regress[i][3]) * height;

    rects[i].x1 = pred_ctr_x - 0.5 * (pred_w - 1.0);
    rects[i].y1 = pred_ctr_y - 0.5 * (pred_h - 1.0);
    rects[i].x2 = pred_ctr_x + 0.5 * (pred_w - 1.0);
    rects[i].y2 = pred_ctr_y + 0.5 * (pred_h - 1.0);
  }

  return rects;
}

anchor_box RetinaFacePostProcess::bbox_pred(anchor_box anchor, vector<float> regress) {
  anchor_box rect;

  float width = anchor.x2 - anchor.x1 + 1;
  float height = anchor.y2 - anchor.y1 + 1;
  float ctr_x = anchor.x1 + 0.5 * (width - 1.0);
  float ctr_y = anchor.y1 + 0.5 * (height - 1.0);

  float pred_ctr_x = regress[0] * width + ctr_x;
  float pred_ctr_y = regress[1] * height + ctr_y;
  float pred_w = exp(regress[2]) * width;
  float pred_h = exp(regress[3]) * height;

  rect.x1 = pred_ctr_x - 0.5 * (pred_w - 1.0);
  rect.y1 = pred_ctr_y - 0.5 * (pred_h - 1.0);
  rect.x2 = pred_ctr_x + 0.5 * (pred_w - 1.0);
  rect.y2 = pred_ctr_y + 0.5 * (pred_h - 1.0);

  return rect;
}

vector<FacePts> RetinaFacePostProcess::landmark_pred(vector<anchor_box> anchors, vector<FacePts> facePts) {
  vector<FacePts> pts(anchors.size());
  for(size_t i = 0; i < anchors.size(); i++) {
    float width = anchors[i].x2 - anchors[i].x1 + 1;
    float height = anchors[i].y2 - anchors[i].y1 + 1;
    float ctr_x = anchors[i].x1 + 0.5 * (width - 1.0);
    float ctr_y = anchors[i].y1 + 0.5 * (height - 1.0);

    for(size_t j = 0; j < 5; j ++) {
      pts[i].x[j] = facePts[i].x[j] * width + ctr_x;
      pts[i].y[j] = facePts[i].y[j] * height + ctr_y;
    }
  }

  return pts;
}

FacePts RetinaFacePostProcess::landmark_pred(anchor_box anchor, FacePts facePt) {
  FacePts pt;
  float width = anchor.x2 - anchor.x1 + 1;
  float height = anchor.y2 - anchor.y1 + 1;
  float ctr_x = anchor.x1 + 0.5 * (width - 1.0);
  float ctr_y = anchor.y1 + 0.5 * (height - 1.0);

  for(size_t j = 0; j < 5; j ++) {
    pt.x[j] = facePt.x[j] * width + ctr_x;
    pt.y[j] = facePt.y[j] * height + ctr_y;
  }

  return pt;
}

bool RetinaFacePostProcess::CompareBBox(const FaceDetectInfo & a, const FaceDetectInfo & b) {
  return a.score > b.score;
}

vector<FaceDetectInfo> RetinaFacePostProcess::nms(vector<FaceDetectInfo>& bboxes, float threshold) {
  vector<FaceDetectInfo> bboxes_nms;
  std::sort(bboxes.begin(), bboxes.end(), CompareBBox);

  int32_t select_idx = 0;
  int32_t num_bbox = static_cast<int32_t>(bboxes.size());
  vector<int32_t> mask_merged(num_bbox, 0);
  bool all_merged = false;

  while (!all_merged) {
    while (select_idx < num_bbox && mask_merged[select_idx] == 1)
      select_idx++;

    if (select_idx == num_bbox) {
      all_merged = true;
      continue;
    }

    bboxes_nms.push_back(bboxes[select_idx]);
    mask_merged[select_idx] = 1;

    anchor_box select_bbox = bboxes[select_idx].rect;
    float area1 = static_cast<float>((select_bbox.x2 - select_bbox.x1 + 1) * (select_bbox.y2 - select_bbox.y1 + 1));
    float x1 = static_cast<float>(select_bbox.x1);
    float y1 = static_cast<float>(select_bbox.y1);
    float x2 = static_cast<float>(select_bbox.x2);
    float y2 = static_cast<float>(select_bbox.y2);

    select_idx++;
    for (int32_t i = select_idx; i < num_bbox; i++) {
      if (mask_merged[i] == 1) {
        continue;
      }
      anchor_box& bbox_i = bboxes[i].rect;
      float x = std::max<float>(x1, static_cast<float>(bbox_i.x1));
      float y = std::max<float>(y1, static_cast<float>(bbox_i.y1));
      float w = std::min<float>(x2, static_cast<float>(bbox_i.x2)) - x + 1;
      float h = std::min<float>(y2, static_cast<float>(bbox_i.y2)) - y + 1;
      if (w <= 0 || h <= 0) {
        continue;
      }
      float area2 = static_cast<float>((bbox_i.x2 - bbox_i.x1 + 1) * (bbox_i.y2 - bbox_i.y1 + 1));
      float area_intersect = w * h;

      if (static_cast<float>(area_intersect) / (area1 + area2 - area_intersect) > threshold) {
        mask_merged[i] = 1;
      }
    }
  }
  return bboxes_nms;
}


void RetinaFacePostProcess::run(
            const bm_net_info_t& net_info,
            float** preds, 
            vector<stFaceRect>& results, int img_h, int img_w, int max_face_count,
            float threshold, float scales) {
  auto &input_shape = net_info.stages[0].input_shapes[0];
  int hs = input_shape.dims[2];
  int ws = input_shape.dims[3];

  map<string, int> output_names_map;
  vector<int> output_sizes;
  for (int i = 0; i < net_info.output_num; i++) {
    output_names_map.insert(pair<string, int>(
                          net_info.output_names[i], i));
    auto &output_shape = net_info.stages[0].output_shapes[i];
    auto count = bmrt_shape_count(&output_shape);
    output_sizes.push_back(count / output_shape.dims[0]);
  }
  float* cls_data = preds[output_names_map["cls"]];
  float *land_data = preds[output_names_map["land"]];
  float *loc_data = preds[output_names_map["loc"]];

  const int num_layer = 3;
  const size_t steps[] = {8, 16, 32};
  const int num_anchor = 2;
  const size_t anchor_sizes[][2] = {
        {16, 32},
        {64, 128},
        {256, 512}};
  const float variances[] = {0.1, 0.2};

  size_t index = 0, min_size;
  const float *loc, *land;
  float x, y, w, h, conf;
  float anchor_w, anchor_h, anchor_x, anchor_y;
  FaceDetectInfo obj;
  vector<FaceDetectInfo> faceInfo;
  for (int il = 0; il < num_layer; ++il){
    int feature_width = (ws + steps[il] - 1) / steps[il];
    int feature_height = (hs + steps[il] - 1) / steps[il];
    for (int iy = 0; iy < feature_height; ++iy){
      for (int ix = 0; ix < feature_width; ++ix){
        for (int ia = 0; ia < num_anchor; ++ia){
          conf = cls_data[index * 2 + 1];
          if (conf < threshold) goto cond;
          min_size = anchor_sizes[il][ia];
          anchor_x = (ix + 0.5) * steps[il] / ws;
          anchor_y = (iy + 0.5) * steps[il] / hs;
          anchor_w = min_size * 1. / ws;
          anchor_h = min_size * 1. / hs;
          obj.score = conf;
          loc = loc_data + index * 4;
          w = exp(loc[2] * variances[1]) * anchor_w;
          h = exp(loc[3] * variances[1]) * anchor_h;
          x = anchor_x + loc[0] * variances[0] * anchor_w;
          y = anchor_y + loc[1] * variances[0] * anchor_h;
          obj.rect.x1 = (x - w / 2) * img_w;
          obj.rect.x2 = (x + w / 2) * img_w;
          obj.rect.y1 = (y - h / 2) * img_h;
          obj.rect.y2 = (y + h / 2) * img_h;
          land = land_data + index * 10;
          for (int i = 0; i < 5; ++i){
            obj.pts.x[i] = (anchor_x +
                land[i * 2] * variances[0] * anchor_w) * img_w;
            obj.pts.y[i] = (anchor_y +
                land[i * 2 + 1] * variances[0] * anchor_h) * img_h;
          }
          // cout << "obj.score = " << obj.score << endl;
          // cout << "obj.rect.x1 = " << obj.rect.x1 << endl;
          // cout << "obj.rect.x2 = " << obj.rect.x2 << endl;
          // cout << "obj.rect.y1 = " << obj.rect.y1 << endl;
          // cout << "obj.rect.y2 = " << obj.rect.y2 << endl;
          faceInfo.push_back(obj);
cond:
          ++index;
        }
      }
    }
  }
  // cout << "faceInfo.size:" << faceInfo.size() << endl;
  // auto objs = nms_(boxes, nms_threshold);
  faceInfo = nms(faceInfo, nms_threshold);
  // cout << "faceInfo.size:" << faceInfo.size() << endl;
  int face_num = 
       max_face_count > static_cast<int>(faceInfo.size()) ? static_cast<int>(faceInfo.size()) : max_face_count;
  for (int i = 0; i < face_num; i++) {
    stFaceRect det_result;
    det_result.left = faceInfo[i].rect.x1;
    det_result.right = faceInfo[i].rect.x2;
    det_result.top = faceInfo[i].rect.y1;
    det_result.bottom = faceInfo[i].rect.y2;
    det_result.score = faceInfo[i].score;
    for(size_t k = 0; k < 5; k++) {
      det_result.points_x[k] = faceInfo[i].pts.x[k];
      det_result.points_y[k] = faceInfo[i].pts.y[k];
    }
    results.push_back(det_result);
  }
}
