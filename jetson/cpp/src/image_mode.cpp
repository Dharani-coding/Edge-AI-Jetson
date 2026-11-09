#include "image_mode.hpp"

#include "preprocess.hpp"
#include "trt_engine.hpp"
#include "visualizer.hpp"
#include "yolo_seg.hpp"

#include <opencv2/imgcodecs.hpp>

#include <sys/stat.h>
#include <cerrno>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace edge_ai {

namespace {

bool isDirectory(const std::string& path)
{
    struct stat info {};
    return ::stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
}

// mkdir -p (std::filesystem needs extra linking on GCC 7).
void makeDirectories(const std::string& path)
{
    for (size_t pos = 0; pos != std::string::npos;)
    {
        pos = path.find('/', pos + 1);
        const std::string part = path.substr(0, pos);
        if (::mkdir(part.c_str(), 0755) != 0 && errno != EEXIST)
            throw std::runtime_error("Cannot create directory: " + part);
    }
}

std::string lowercaseExtension(const std::string& path)
{
    const size_t dot = path.find_last_of('.');
    if (dot == std::string::npos)
        return "";
    std::string ext = path.substr(dot + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
}

std::string fileName(const std::string& path)
{
    const size_t slash = path.find_last_of('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

std::string stem(const std::string& name)
{
    const size_t dot = name.find_last_of('.');
    return dot == std::string::npos ? name : name.substr(0, dot);
}

std::vector<std::string> listImages(const std::string& path)
{
    if (!isDirectory(path))
        return {path};   // a single image file

    std::vector<cv::String> all;
    cv::glob(path + "/*", all, false);

    std::vector<std::string> images;
    for (const cv::String& file : all)
    {
        const std::string ext = lowercaseExtension(file);
        if (ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp")
            images.push_back(file);
    }
    std::sort(images.begin(), images.end());
    return images;
}

// "2 person, 1 dog", like the Ultralytics console output.
std::string summarize(const std::vector<Detection>& detections)
{
    if (detections.empty())
        return "no detections";

    std::map<std::string, int> counts;
    for (const Detection& d : detections)
        ++counts[cocoClassName(d.class_id)];

    std::string text;
    for (const auto& entry : counts)
        text += (text.empty() ? "" : ", ") + std::to_string(entry.second) + " " + entry.first;
    return text;
}

void writeDetections(const std::string& path, const std::vector<Detection>& detections,
                     const LetterboxInfo& lb, const cv::Size& image_size)
{
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("Cannot write " + path);

    // Map model-input boxes back to original pixels without integer rounding,
    // so the numbers are directly comparable with Ultralytics' xyxy output.
    auto clampX = [&](float x) { return std::min(std::max(x, 0.0f), static_cast<float>(image_size.width)); };
    auto clampY = [&](float y) { return std::min(std::max(y, 0.0f), static_cast<float>(image_size.height)); };

    char line[160];
    for (const Detection& d : detections)
    {
        const cv::Rect2f& b = d.input_box;
        const float x1 = clampX((b.x - lb.pad_x) / lb.scale);
        const float y1 = clampY((b.y - lb.pad_y) / lb.scale);
        const float x2 = clampX((b.x + b.width - lb.pad_x) / lb.scale);
        const float y2 = clampY((b.y + b.height - lb.pad_y) / lb.scale);

        std::snprintf(line, sizeof(line), "%d %.4f %.1f %.1f %.1f %.1f %s\n",
                      d.class_id, d.confidence, x1, y1, x2, y2, cocoClassName(d.class_id));
        out << line;
    }
}

}  // namespace

int runOnImages(const AppConfig& config)
{
    const std::vector<std::string> images = listImages(config.images_path);
    if (images.empty())
        throw std::runtime_error("No images (.jpg/.jpeg/.png/.bmp) found in " + config.images_path);

    makeDirectories(config.output_dir);

    TrtEngine engine(config.detector.engine_path);
    const SegModelInfo& model = engine.info();
    Letterbox letterbox(model.input_width, model.input_height);
    YoloSegDecoder decoder(model, config.detector);
    Visualizer visualizer(model.num_classes, config.detector.draw_masks);

    // Warm-up, so the first image's timing is not dominated by CUDA/TensorRT start-up.
    engine.infer();

    std::cout << "\nProcessing " << images.size() << " image(s) -> " << config.output_dir << "\n" << std::endl;

    int failures = 0;
    for (size_t i = 0; i < images.size(); ++i)
    {
        const std::string& path = images[i];
        cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
        if (image.empty())
        {
            std::cerr << "Cannot read " << path << std::endl;
            ++failures;
            continue;
        }

        const auto start = std::chrono::steady_clock::now();
        const LetterboxInfo lb = letterbox.apply(image, engine.input());
        const auto t_pre = std::chrono::steady_clock::now();
        engine.infer();
        const auto t_infer = std::chrono::steady_clock::now();
        const std::vector<Detection>& detections =
            decoder.process(engine.detections(), engine.prototypes(), lb, image.size());
        const auto t_post = std::chrono::steady_clock::now();

        auto ms = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };

        const std::string name = fileName(path);
        writeDetections(config.output_dir + "/" + stem(name) + ".txt", detections, lb, image.size());

        visualizer.drawDetections(image, detections);
        if (!cv::imwrite(config.output_dir + "/" + name, image))
        {
            std::cerr << "Cannot write " << config.output_dir << "/" << name << std::endl;
            ++failures;
        }

        std::printf("image %zu/%zu %s: %dx%d %s | pre %.1f ms, trt %.1f ms, post %.1f ms\n",
                    i + 1, images.size(), name.c_str(), image.cols, image.rows,
                    summarize(detections).c_str(),
                    ms(start, t_pre), ms(t_pre, t_infer), ms(t_infer, t_post));
    }

    std::cout << "\nResults saved to " << config.output_dir << std::endl;
    return failures == 0 ? 0 : 1;
}

}  // namespace edge_ai
