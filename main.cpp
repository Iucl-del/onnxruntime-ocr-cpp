#include <iostream>
#include "PaddleOCR.hpp"



int main() {
    namespace fs = std::filesystem;
    const fs::path run_dir = fs::current_path();
    fs::path model_path = run_dir.parent_path() / "model" / "det_server.onnx";
    fs::path image_path = run_dir.parent_path() / "1.jpg";

    PaddleOCR ocr(model_path.string());
    cv::Mat img = cv::imread(image_path.string());
    cv::Mat annotated_image = ocr.detectTextInImage(img.clone());

    std::vector<cv::Rect> boxs;
    ocr.get_text_boxes(boxs);
    for (auto& box : boxs) {
        cv::imshow("result",img(box).clone());
        cv::waitKey(0);
    }

    cv::imwrite("img.png", annotated_image);
                                       // 等待按键
    return 0;
}
