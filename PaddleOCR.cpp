#include "PaddleOCR.hpp"

PaddleOCR::PaddleOCR(std::string_view model_path) : env_(ORT_LOGGING_LEVEL_WARNING, "det"),det_session_(nullptr),cls_session_(nullptr),model_path_(model_path) {
    options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = 0;
    options_.AppendExecutionProvider_CUDA(cuda_options);
    std::string det_model_path = model_path_ + "det_server.onnx";
    det_session_ = Ort::Session{env_, det_model_path.c_str(), options_};
    std::string cls_model_path = model_path_ + "cls_server.onnx";
    cls_session_ = Ort::Session{env_, cls_model_path.c_str(), options_};
}

cv::Mat PaddleOCR::detectTextInImage(cv::Mat image, const TextBoxExpandConfig& config) {
    image_ = image.clone();
    int64_t height,width;
    cv::Mat out_img = Prcocess(image,height,width);
    std::vector<float> input_tensor = matToCHW(out_img);

    std::vector<int64_t> input_shape = {1, 3, height, width}; // 定义输入tensor的形状（NCHW）
    Ort::MemoryInfo memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault));
    Ort::Value input= Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor.data(),
        input_tensor.size(),
        input_shape.data(),
        input_shape.size()
    );

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_holder = det_session_.GetInputNameAllocated(0, allocator);
    auto output_name_holder = det_session_.GetOutputNameAllocated(0, allocator);
    const char* input_names[] = {input_name_holder.get()};
    const char* output_names[] = {output_name_holder.get()};

    auto output = det_session_.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input,
        1,
        output_names,
        1
    );

    float* output_data = output[0].GetTensorMutableData<float>();
    auto out_shape = output[0].GetTensorTypeAndShapeInfo().GetShape(); // 获取输出shape
    int out_h = out_shape[2];                          // 获取输出高度
    int out_w = out_shape[3];                          // 获取输出宽度

    cv::Mat prob_map(out_h, out_w, CV_32F, output_data); // 构造概率图（不拷贝数据）

    cv::Mat binary;                                    // 定义二值图
    cv::threshold(prob_map, binary, 0.25, 255, cv::THRESH_BINARY); // 二值化处理（略降阈值减少漏检）
    binary.convertTo(binary, CV_8U);                   // 转换为8位图像
    cv::morphologyEx(
        binary,
        binary,
        cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 3))
    ); // 闭运算优先连通断裂文本，避免开运算过度腐蚀
    cv::dilate(
        binary,
        binary,
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3))
    ); // 轻微膨胀，提升文本区域覆盖

    std::vector<std::vector<cv::Point>> contours;                // 定义轮廓容器
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE); // 查找外轮廓

    for (auto& contour : contours) {                   // 遍历所有轮廓
        if (cv::contourArea(contour) < 100) continue;  // 过滤小面积区域

        cv::Rect rect = cv::boundingRect(contour);     // 获取最小外接矩形

        // 对检测框做安全扩展，避免框贴边导致“没完整圈住”
        int pad_w = std::max(config.min_pad_w, static_cast<int>(std::round(rect.width  * config.pad_w_ratio)));
        int pad_h = std::max(config.min_pad_h, static_cast<int>(std::round(rect.height * config.pad_h_ratio)));
        rect.x = std::max(0, rect.x - pad_w);
        rect.y = std::max(0, rect.y - pad_h);
        rect.width = std::min(image.cols - rect.x,  rect.width  + pad_w * 2);
        rect.height = std::min(image.rows - rect.y, rect.height + pad_h * 2);

        text_crops_.emplace_back(rect);

        cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 1); // 在原图画框
    }
    return image;
}

std::vector<float> PaddleOCR::matToCHW(cv::Mat& img) {
    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    std::vector<float> data;
    data.reserve(static_cast<size_t>(img.rows * img.cols * img.channels()));
    for (size_t i = 0; i < 3; ++i) {
        const float* begin = channels[i].ptr<float>(0);
        const float* end = begin + channels[i].total();
        data.insert(data.end(), begin, end);
    }
    return data;
}


cv::Mat PaddleOCR::Prcocess(cv::Mat& img, int64_t& height, int64_t& width) {
    // 按32对齐到更大尺寸，使用padding而非resize，避免拉伸变形
    height = ((img.rows + 31) / 32) * 32;
    width  = ((img.cols + 31) / 32) * 32;

    cv::Mat padded;
    cv::copyMakeBorder(
        img,
        padded,
        0, height - img.rows,
        0, width - img.cols,
        cv::BORDER_CONSTANT,
        cv::Scalar(0, 0, 0)
    );

    // 模型按RGB均值方差归一化，先转色彩空间
    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

    cv::Mat float_img;
    rgb.convertTo(float_img, CV_32F, 1.0 / 255);

    std::vector<float> mean {0.485, 0.456, 0.406};         // 定义每个通道的均值（RGB）
    std::vector<float> std {0.229, 0.224, 0.225};          // 定义每个通道的标准差

    std::vector<cv::Mat> channels;
    cv::split(float_img, channels);

    for (size_t i = 0; i < 3; ++i) {
        channels[i] = (channels[i]-mean[i]) / std[i];
    }

    cv::merge(channels, float_img);
    return float_img;
}

cv::Mat PaddleOCR::preprocessClsImage(int64_t& height, int64_t& width) {

}