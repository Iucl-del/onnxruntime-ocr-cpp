#pragma once
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <map>

struct TextBoxExpandConfig {
    float pad_w_ratio = 1.0f / 15.0f;  // 宽度扩展比例
    float pad_h_ratio = 1.0f / 6.0f;   // 高度扩展比例
    int min_pad_w = 1.5;               // 最小横向扩展像素
    int min_pad_h = 4.0;               // 最小纵向扩展像素
};

struct TextDirectionResult {
    cv::Rect rect;
    int angle; //旋转角度
    float score;//分类可信度
};

class PaddleOCR {
public:
    PaddleOCR(std::string_view model_path);
    cv::Mat detectTextInImage(cv::Mat image, const TextBoxExpandConfig& config = TextBoxExpandConfig());//文本检测函数
    std::vector<TextDirectionResult> detectTextInImages();
    void get_text_boxes(std::vector<cv::Rect>& text_crops){ text_crops = text_crops_; }
private:
    std::vector<float> matToCHW(cv::Mat& img);
    cv::Mat Prcocess(cv::Mat& img, int64_t& height, int64_t& width);
    cv::Mat preprocessClsImage(cv::Mat& img, int64_t& height, int64_t& width);

private:
    Ort::Env env_;
    Ort::SessionOptions options_;
    Ort::Session det_session_;
    Ort::Session cls_session_;

    std::string model_path_;

    cv::Mat image_;
    std::vector<cv::Rect> text_crops_;
    std::map<std::string, cv::Rect> text_to_rect_map_;
    std::map<cv::Rect,std::pair<int,float>> rect_to_direction_map_;
};
