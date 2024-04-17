#include <sys/types.h>
#include <unistd.h>
#include <sys/wait.h>
#include <errno.h>
#include <string.h>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <iostream>
#include "json.hpp"
#include "ppocr_det.hpp"
#include "ppocr_cls.hpp"
#include "ppocr_rec.hpp"
#include "ff_decode.hpp"
using json = nlohmann::json;
using namespace std;
#define USE_ANGLE_CLS 0
#define USE_OPENCV_WARP 0
#define USE_OPENCV_DECODE 0
//from PPaddleOCR github.
cv::Mat GetRotateCropImage(const cv::Mat &srcimage,
                                    std::vector<std::vector<int>> box) {
    cv::Mat image;
    srcimage.copyTo(image);
    std::vector<std::vector<int>> points = box;

    int x_collect[4] = {box[0][0], box[1][0], box[2][0], box[3][0]};
    int y_collect[4] = {box[0][1], box[1][1], box[2][1], box[3][1]};
    int left = int(*std::min_element(x_collect, x_collect + 4));
    int right = int(*std::max_element(x_collect, x_collect + 4));
    int top = int(*std::min_element(y_collect, y_collect + 4));
    int bottom = int(*std::max_element(y_collect, y_collect + 4));

    cv::Mat img_crop;
    image(cv::Rect(left, top, right - left, bottom - top)).copyTo(img_crop);

    for (int i = 0; i < points.size(); i++) {
        points[i][0] -= left;
        points[i][1] -= top;
    }

    int img_crop_width = int(sqrt(pow(points[0][0] - points[1][0], 2) +
                                    pow(points[0][1] - points[1][1], 2)));
    int img_crop_height = int(sqrt(pow(points[0][0] - points[3][0], 2) +
                                    pow(points[0][1] - points[3][1], 2)));
    
    //VPP only accept > 16 width or height.
    img_crop_width = MAX(16, img_crop_width);
    img_crop_height = MAX(16, img_crop_height);

    cv::Point2f pts_std[4];
    pts_std[0] = cv::Point2f(0., 0.);
    pts_std[1] = cv::Point2f(img_crop_width, 0.);
    pts_std[2] = cv::Point2f(img_crop_width, img_crop_height);
    pts_std[3] = cv::Point2f(0.f, img_crop_height);

    cv::Point2f pointsf[4];
    pointsf[0] = cv::Point2f(points[0][0], points[0][1]);
    pointsf[1] = cv::Point2f(points[1][0], points[1][1]);
    pointsf[2] = cv::Point2f(points[2][0], points[2][1]);
    pointsf[3] = cv::Point2f(points[3][0], points[3][1]);

    cv::Mat M = cv::getPerspectiveTransform(pointsf, pts_std);

    cv::Mat dst_img;
    cv::warpPerspective(img_crop, dst_img, M,
                        cv::Size(img_crop_width, img_crop_height),
                        cv::BORDER_REPLICATE);

    if (float(dst_img.rows) >= float(dst_img.cols) * 1.5) {
        cv::Mat srcCopy = cv::Mat(dst_img.rows, dst_img.cols, dst_img.depth());
        cv::transpose(dst_img, srcCopy);
        cv::flip(srcCopy, srcCopy, 0);
        return srcCopy;
    } else {
        return dst_img;
    }
}

#if USE_OPENCV_WARP
    bm_image get_rotate_crop_image(bm_handle_t handle, bm_image input_bmimg_planar, OCRBox box) {
        cv::Mat cv_img;
        assert(BM_SUCCESS == cv::bmcv::toMAT(&input_bmimg_planar, cv_img));

        std::vector<std::vector<int>> detectionBox;
        detectionBox.push_back({box.x1, box.y1});
        detectionBox.push_back({box.x2, box.y2});
        detectionBox.push_back({box.x3, box.y3});
        detectionBox.push_back({box.x4, box.y4});
        cv::Mat cv_crop = GetRotateCropImage(cv_img, detectionBox);
        bm_image crop_bmimg;
        assert(BM_SUCCESS == cv::bmcv::toBMI(cv_crop, &crop_bmimg)); //BGR_PACKED.
        int stride1[3], stride2[3];
        assert(BM_SUCCESS == bm_image_get_stride(crop_bmimg, stride1));
        stride2[0] = FFALIGN(stride1[0], 64);
        stride2[1] = FFALIGN(stride1[1], 64);
        stride2[2] = FFALIGN(stride1[2], 64);
        bm_image crop_planar;
        bm_image_create(handle, crop_bmimg.height, crop_bmimg.width, input_bmimg_planar.image_format, input_bmimg_planar.data_type, &crop_planar, stride2);
        bmcv_image_storage_convert(handle, 1, &crop_bmimg, &crop_planar); //BGR_PLANNAR
        bm_image_destroy(crop_bmimg);
        return crop_planar;
    }
#else
    bm_image get_rotate_crop_image(bm_handle_t handle, bm_image input_bmimg_planar, OCRBox box) { 
        int crop_width = max((int)sqrt(pow(box.x1 - box.x2, 2) + pow(box.y1 - box.y2, 2)),
                            (int)sqrt(pow(box.x3 - box.x4, 2) + pow(box.y3 - box.y4, 2)));
        int crop_height = max((int)sqrt(pow(box.x1 - box.x4, 2) + pow(box.y1 - box.y4, 2)),
                            (int)sqrt(pow(box.x3 - box.x2, 2) + pow(box.y3 - box.y2, 2)));
        // legality bounding
        crop_width = min(max(16, crop_width), input_bmimg_planar.width);
        crop_height = min(max(16, crop_height), input_bmimg_planar.height);

        bmcv_perspective_image_coordinate coord;
        coord.coordinate_num = 1;
        shared_ptr<bmcv_perspective_coordinate> coord_data = make_shared<bmcv_perspective_coordinate>();
        coord.coordinate = coord_data.get();
        coord.coordinate->x[0] = box.x1;
        coord.coordinate->y[0] = box.y1;
        coord.coordinate->x[1] = box.x2;
        coord.coordinate->y[1] = box.y2;
        coord.coordinate->x[2] = box.x4;
        coord.coordinate->y[2] = box.y4;
        coord.coordinate->x[3] = box.x3;
        coord.coordinate->y[3] = box.y3;

        bm_image crop_bmimg;
        bm_image_create(handle, crop_height, crop_width, input_bmimg_planar.image_format, input_bmimg_planar.data_type, &crop_bmimg);
        assert(BM_SUCCESS == bmcv_image_warp_perspective_with_coordinate(handle, 1, &coord, &input_bmimg_planar, &crop_bmimg, 0));//bilinear interpolation.

        if ((float)crop_height / crop_width < 1.5) {
            return crop_bmimg;
        } else {
            bm_image rot_bmimg;
            bm_image_create(handle, crop_width, crop_height, input_bmimg_planar.image_format, input_bmimg_planar.data_type,
                            &rot_bmimg);

            cv::Point2f center = cv::Point2f(crop_width / 2.0, crop_height / 2.0);
            cv::Mat rot_mat = cv::getRotationMatrix2D(center, -90.0, 1.0);
            bmcv_affine_image_matrix matrix_image;
            matrix_image.matrix_num = 1;
            std::shared_ptr<bmcv_affine_matrix> matrix_data = std::make_shared<bmcv_affine_matrix>();
            matrix_image.matrix = matrix_data.get();
            matrix_image.matrix->m[0] = rot_mat.at<double>(0, 0);
            matrix_image.matrix->m[1] = rot_mat.at<double>(0, 1);
            matrix_image.matrix->m[2] = rot_mat.at<double>(0, 2) - crop_height / 2.0 + crop_width / 2.0;
            matrix_image.matrix->m[3] = rot_mat.at<double>(1, 0);
            matrix_image.matrix->m[4] = rot_mat.at<double>(1, 1);
            matrix_image.matrix->m[5] = rot_mat.at<double>(1, 2) - crop_height / 2.0 + crop_width / 2.0;

            assert(BM_SUCCESS == bmcv_image_warp_affine(handle, 1, &matrix_image, &crop_bmimg, &rot_bmimg, 0));//bilinear interpolation
            bm_image_destroy(crop_bmimg);
            return rot_bmimg;
        }
    }
#endif

void visualize_boxes(bm_image input_bmimg, OCRBoxVec& ocr_result, const string& save_path, float rec_thresh) {
    cv::Mat img_src;
    cv::bmcv::toMAT(&input_bmimg, img_src);
    // cv::Mat img_res(input_bmimg.height, input_bmimg.width, CV_8UC3, cv::Scalar(255, 255, 255));

    for (int n = 0; n < ocr_result.size(); n++) {
        if (ocr_result[n].rec_res.empty())
            continue;
        cv::Point rook_points[4];
        rook_points[0] = cv::Point(int(ocr_result[n].x1), int(ocr_result[n].y1));
        rook_points[1] = cv::Point(int(ocr_result[n].x2), int(ocr_result[n].y2));
        rook_points[2] = cv::Point(int(ocr_result[n].x3), int(ocr_result[n].y3));
        rook_points[3] = cv::Point(int(ocr_result[n].x4), int(ocr_result[n].y4));

        const cv::Point* ppt[1] = {rook_points};
        int npt[] = {4};
        string label = ocr_result[n].rec_res;
        if (label != "###" && ocr_result[n].score > rec_thresh) {
            cv::polylines(img_src, ppt, npt, 1, 1, CV_RGB(0, 255, 0), 2, 8, 0);
            std::cout << label << "; ";
            // cv::putText(img_src, label, cv::Point(int(ocr_result[n].x1), int(ocr_result[n].y1)),
            // cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 1);
        }
    }
    cv::imwrite(save_path, img_src);
}

int main(int argc, char* argv[]) {
    cout.setf(ios::fixed);
    // get params
    const char* keys =
        "{input | ../../datasets/cali_set_det | input path, images directory}"
        "{bmodel_det | ../../models/BM1684X/ch_PP-OCRv3_det_fp32.bmodel | det bmodel file path}"
        "{bmodel_rec | ../../models/BM1684X/ch_PP-OCRv3_rec_fp32.bmodel | rec bmodel file path}"
        "{bmodel_cls | ../../models/BM1684X/ch_PP-OCRv3_cls_fp32.bmodel | cls bmodel file path, unsupport now.}"
        "{batch_size | 4 | ppocr system batchsize}"
        "{rec_thresh | 0.5 | recognize threshold}"
        "{labelnames | ../../datasets/ppocr_keys_v1.txt | class names file path}"
        "{dev_id | 0 | TPU device id}"
        "{help | 0 | print help information.}"
        "{use_beam_search | false | beam search trigger}"
        "{beam_size | 3 | beam size, default 3, available 1-40, only valid when using beam search}";

    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }
    string input = parser.get<string>("input");
    string bmodel_det = parser.get<string>("bmodel_det");
    string bmodel_cls = parser.get<string>("bmodel_cls");
    string bmodel_rec = parser.get<string>("bmodel_rec");
    string label_names = parser.get<string>("labelnames");
    int batch_size = parser.get<int>("batch_size");
    float rec_thresh = parser.get<float>("rec_thresh");
    assert(batch_size > 0);
    int dev_id = parser.get<int>("dev_id");
    bool beam_search = parser.get<bool>("use_beam_search");
    int beam_size = parser.get<int>("beam_size");

    if(beam_size < 1 || beam_size > 40){
        cout << "ERROR!!beam_size out of range, should be integer in range(1, 41)" << endl;
        exit(1);
    }
    // check params
    struct stat info;
    if (stat(bmodel_det.c_str(), &info) != 0) {
        cout << "Cannot find valid det bmodel file." << endl;
        exit(1);
    }
#if USE_ANGLE_CLS
    if (stat(bmodel_cls.c_str(), &info) != 0) {
        cout << "Cannot find valid cls bmodel file." << endl;
        exit(1);
    }
#endif
    if (stat(bmodel_rec.c_str(), &info) != 0) {
        cout << "Cannot find valid rec bmodel file." << endl;
        exit(1);
    }
    if (stat(label_names.c_str(), &info) != 0) {
        cout << "Cannot find labelnames file." << endl;
        exit(1);
    }
    if (stat(input.c_str(), &info) != 0) {
        cout << "Cannot find input path." << endl;
        exit(1);
    }

    // creat handle
    BMNNHandlePtr handle = std::make_shared<BMNNHandle>(dev_id);
    cout << "set device id: " << dev_id << endl;
    bm_handle_t h = handle->handle();

    // Load bmodel
    std::shared_ptr<BMNNContext> bm_ctx_det = std::make_shared<BMNNContext>(handle, bmodel_det.c_str());
    PPOCR_Detector ppocr_det(bm_ctx_det);
    CV_Assert(0 == ppocr_det.Init());

    std::shared_ptr<BMNNContext> bm_ctx_rec = std::make_shared<BMNNContext>(handle, bmodel_rec.c_str());
    PPOCR_Rec ppocr_rec(bm_ctx_rec);
    CV_Assert(0 == ppocr_rec.Init(label_names.c_str()));

#if USE_ANGLE_CLS
    std::shared_ptr<BMNNContext> bm_ctx_cls = std::make_shared<BMNNContext>(handle, bmodel_cls.c_str());
    PPOCR_Cls ppocr_cls(bm_ctx_det);
    CV_Assert(0 == ppocr_cls.Init());
#endif


    // Save results
    TimeStamp ts;
    ppocr_det.enableProfile(&ts);
    ppocr_rec.enableProfile(&ts);
    int total_crop_num = 0;
    int total_frame_num = 0;
    json results_json;

    // for cv decode;
#if USE_OPENCV_DECODE
    std::vector<cv::Mat> batch_cvmats;
#endif
    // creat save path
    if (access("results", 0) != F_OK)
        mkdir("results", S_IRWXU);
    if (access("results/images", 0) != F_OK)
        mkdir("results/images", S_IRWXU);

    if (info.st_mode & S_IFDIR) {
        vector<string> files_vector;
        DIR* pDir;
        struct dirent* ptr;
        pDir = opendir(input.c_str());
        while ((ptr = readdir(pDir)) != 0) {
            if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
                files_vector.push_back(input + "/" + ptr->d_name);
            }
        }
        closedir(pDir);

        std::sort(files_vector.begin(), files_vector.end());

        vector<bm_image> batch_imgs;
        vector<string> batch_names;
        vector<OCRBoxVec> batch_boxes;
        vector<pair<int, int>> batch_ids;
        vector<bm_image> batch_crops;
        int id = 0;
        int cn = files_vector.size();
        for (vector<string>::iterator iter = files_vector.begin(); iter != files_vector.end(); iter++) {
            string img_file = *iter;
            id++;
            cout << id << "/" << cn << ", img_file: " << img_file << endl;
            // Read image
            ts.save("decode time", 1);
            bm_image bmimg;
        #if USE_OPENCV_DECODE        
            cv::Mat cvmat = cv::imread(img_file, IMREAD_COLOR, dev_id);
            batch_cvmats.push_back(cvmat);//so that cvmat will not be released.
            cv::bmcv::toBMI(cvmat, &bmimg);
        #else    
            picDec(h, img_file.c_str(), bmimg);
        #endif
            ts.save("decode time", 1);
            size_t index = img_file.rfind("/");
            string img_name = img_file.substr(index + 1);
            batch_imgs.push_back(bmimg);
            batch_names.push_back(img_name);
            total_frame_num++; 

            iter++;
            bool end_flag = (iter == files_vector.end());
            iter--;
            
            if ((batch_imgs.size() == batch_size || end_flag) && !batch_imgs.empty()) {
                CV_Assert(0 == ppocr_det.run(batch_imgs, batch_boxes));
                for (int i = 0; i < batch_imgs.size(); i++) {
                    bm_image input_bmimg_planar;
                    bm_image_create(h, batch_imgs[i].height, batch_imgs[i].width, FORMAT_BGR_PLANAR,
                                    batch_imgs[i].data_type, &input_bmimg_planar);
                    auto ret = bmcv_image_vpp_convert(h, 1, batch_imgs[i], &input_bmimg_planar);

                    bm_image_destroy(batch_imgs[i]);
                    batch_imgs[i] = input_bmimg_planar;

#if DEBUG
                    std::cout << "original image: " << batch_imgs[i].height << " " << batch_imgs[i].width << std::endl;
#endif
                    for (int j = 0; j < batch_boxes[i].size(); j++) {
                    #if DEBUG
                        batch_boxes[i][j].printInfo();
                    #endif
                        LOG_TS(&ts, "(per crop)get crop time");
                        bm_image crop_bmimg = get_rotate_crop_image(h, input_bmimg_planar, batch_boxes[i][j]);
                        LOG_TS(&ts, "(per crop)get crop time");
                        batch_crops.push_back(crop_bmimg);
                        batch_ids.push_back(std::make_pair(i, j));
                        total_crop_num += 1;
                    }
                }

#if 0
                for(int i = 0; i < batch_imgs.size(); i++)
                    if(batch_names[i] == "gt_4743.jpg"){
                        for(int j = 0; j < batch_boxes[i].size(); j++)
                        {
                            cv::Mat resized_img;
                            cv::bmcv::toMAT(&batch_crops[j], resized_img);
                            static int ii = 0;
                            std::string fname = cv::format("results/%s_crop_img_%d.jpg", batch_names[i].c_str(), ii++);
                            cv::imwrite(fname, resized_img);
                        }
                    }
#endif

#if USE_ANGLE_CLS
                //
                exit(1);
#endif
                CV_Assert(0 == ppocr_rec.run(batch_crops, batch_boxes, batch_ids, beam_search, beam_size));
                for (int i = 0; i < batch_boxes.size(); i++) {
                    string save_file = "results/images/" + batch_names[i];
                    std::cout << "detect results: ";
                    visualize_boxes(batch_imgs[i], batch_boxes[i], save_file.c_str(), rec_thresh);
                    std::cout << std::endl;
                    size_t index = batch_names[i].rfind(".");
                    string striped_name = batch_names[i].substr(0, index);
                    vector<json> ocrinfo_vec;
                    for (auto& b : batch_boxes[i]) {
                        if (b.rec_res != "###" && b.score > rec_thresh) {
                            json ocr_info;
                            ocr_info["illegibility"] = false;
                            ocr_info["score"] = b.score;
                            ocr_info["points"] = {{b.x1, b.y1}, {b.x2, b.y2}, {b.x3, b.y3}, {b.x4, b.y4}};
                            ocr_info["transcription"] = b.rec_res;
                            ocrinfo_vec.push_back(ocr_info);
                        }
                    }
                    results_json[striped_name] = ocrinfo_vec;
                }

                for (auto& crop_ : batch_crops) {
                    bm_image_destroy(crop_);
                }
                for (auto& origin_ : batch_imgs) {
                    bm_image_destroy(origin_);
                }
                batch_crops.clear();
                batch_ids.clear();
                batch_imgs.clear();
                batch_names.clear();
                batch_boxes.clear();
            #if USE_OPENCV_DECODE
                batch_cvmats.clear();
            #endif
            }
        }
        string json_file = "results/ppocr_system_results_b" + std::to_string(batch_size) + ".json";
        cout << "================" << endl;
        cout << "result saved in " << json_file << endl;
        ofstream(json_file) << std::setw(4) << results_json;

        std::cout << "avg_crop_num: " << (float)total_crop_num / (float)total_frame_num << std::endl;

        time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
        ts.calbr_basetime(base_time);
        ts.build_timeline("ppocr test");
        ts.show_summary("ppocr test");
        ts.clear();
    }
}