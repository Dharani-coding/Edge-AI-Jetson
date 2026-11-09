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
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace edge_ai {

namespace {

using Clock = std::chrono::steady_clock;

double msBetween(Clock::time_point a, Clock::time_point b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// ---------------------------------------------------------------- file helpers

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

    if (images.empty())
        throw std::runtime_error("No images (.jpg/.jpeg/.png/.bmp) found in " + path);
    return images;
}

// ---------------------------------------------------------------- detection helpers

// Model-input box -> original image pixels, without integer rounding, clipped.
cv::Rect2f toImageBox(const Detection& d, const LetterboxInfo& lb, const cv::Size& size)
{
    auto clampX = [&](float x) { return std::min(std::max(x, 0.0f), static_cast<float>(size.width)); };
    auto clampY = [&](float y) { return std::min(std::max(y, 0.0f), static_cast<float>(size.height)); };

    const cv::Rect2f& b = d.input_box;
    const float x1 = clampX((b.x - lb.pad_x) / lb.scale);
    const float y1 = clampY((b.y - lb.pad_y) / lb.scale);
    const float x2 = clampX((b.x + b.width - lb.pad_x) / lb.scale);
    const float y2 = clampY((b.y + b.height - lb.pad_y) / lb.scale);
    return cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
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

    char line[160];
    for (const Detection& d : detections)
    {
        const cv::Rect2f b = toImageBox(d, lb, image_size);
        std::snprintf(line, sizeof(line), "%d %.4f %.1f %.1f %.1f %.1f %s\n",
                      d.class_id, d.confidence, b.x, b.y, b.x + b.width, b.y + b.height,
                      cocoClassName(d.class_id));
        out << line;
    }
}

// ---------------------------------------------------------------- COCO helpers

// YOLO class index (0-79) -> COCO category id (1-90, with gaps).
constexpr int kCocoCategoryIds[80] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90};

// COCO file names are the zero-padded image id: 000000397133.jpg -> 397133.
long cocoImageId(const std::string& name)
{
    const std::string digits = stem(name);
    if (digits.empty() || !std::all_of(digits.begin(), digits.end(), ::isdigit))
        throw std::runtime_error("Not a COCO image file name (expected digits only): " + name);
    return std::stol(digits);
}

// Full-image mask as COCO RLE, compressed exactly like pycocotools' rleToString.
// RLE = run lengths over the image in column-major order, starting with a run of 0s.
// The mask is only non-zero inside `box`, so columns outside it are one run each.
std::string encodeRle(const cv::Mat& box_mask, const cv::Rect& box, const cv::Size& image_size)
{
    const int64_t h = image_size.height;
    std::vector<int64_t> counts;
    counts.reserve(4 * static_cast<size_t>(box.width) + 2);

    uint8_t current = 0;
    int64_t run = 0;
    auto push = [&](uint8_t value, int64_t n) {
        if (n <= 0)
            return;
        if (value != current)
        {
            counts.push_back(run);
            run = 0;
            current = value;
        }
        run += n;
    };

    const cv::Mat columns = box_mask.t();   // row i = column i of the box, contiguous

    for (int x = 0; x < image_size.width; ++x)
    {
        if (box_mask.empty() || x < box.x || x >= box.x + box.width)
        {
            push(0, h);
            continue;
        }
        push(0, box.y);
        const uchar* column = columns.ptr<uchar>(x - box.x);
        for (int y = 0; y < box.height; ++y)
            push(column[y] ? 1 : 0, 1);
        push(0, h - box.y - box.height);
    }
    counts.push_back(run);

    // pycocotools compression: delta to the count two back, 5-bit groups, offset by '0'.
    std::string text;
    for (size_t i = 0; i < counts.size(); ++i)
    {
        int64_t x = counts[i];
        if (i > 2)
            x -= counts[i - 2];
        bool more = true;
        while (more)
        {
            int64_t c = x & 0x1f;
            x >>= 5;   // arithmetic shift, as in maskApi.c
            more = (c & 0x10) ? x != -1 : x != 0;
            if (more)
                c |= 0x20;
            text.push_back(static_cast<char>(c + 48));
        }
    }
    return text;
}

// The RLE alphabet ('0'..'o') contains '\', which must be escaped in JSON.
void writeJsonString(FILE* out, const std::string& text)
{
    std::fputc('"', out);
    for (char c : text)
    {
        if (c == '\\' || c == '"')
            std::fputc('\\', out);
        std::fputc(c, out);
    }
    std::fputc('"', out);
}

std::string precisionFromEngineName(const std::string& path)
{
    const std::string name = fileName(path);
    for (const char* tag : {"fp16", "fp32", "int8"})
        if (name.find(tag) != std::string::npos)
            return tag;
    return "unknown";
}

struct FileCloser
{
    void operator()(FILE* f) const { if (f) std::fclose(f); }
};

}  // namespace

// ---------------------------------------------------------------- image mode

int runOnImages(const AppConfig& config)
{
    const std::vector<std::string> images = listImages(config.images_path);
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

        const auto start = Clock::now();
        const LetterboxInfo lb = letterbox.apply(image, engine.input());
        const auto t_pre = Clock::now();
        engine.infer();
        const auto t_infer = Clock::now();
        const std::vector<Detection>& detections =
            decoder.process(engine.detections(), engine.prototypes(), lb, image.size());
        const auto t_post = Clock::now();

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
                    msBetween(start, t_pre), msBetween(t_pre, t_infer), msBetween(t_infer, t_post));
    }

    std::cout << "\nResults saved to " << config.output_dir << std::endl;
    return failures == 0 ? 0 : 1;
}

// ---------------------------------------------------------------- COCO evaluation

int runEvaluation(const AppConfig& config)
{
    const std::vector<std::string> images = listImages(config.eval_path);
    makeDirectories(config.output_dir);

    DetectorConfig detector = config.detector;
    detector.draw_masks = true;   // mask metrics need the masks

    TrtEngine engine(detector.engine_path);
    const SegModelInfo& model = engine.info();
    if (model.num_classes != 80)
        throw std::runtime_error("COCO evaluation needs an 80-class model");

    Letterbox letterbox(model.input_width, model.input_height);
    YoloSegDecoder decoder(model, detector);
    engine.infer();   // warm-up

    // The buffer is declared first so it outlives the FILE (fclose flushes into it).
    std::vector<char> buffer(1 << 20);
    const std::string predictions_path = config.output_dir + "/predictions.json";
    std::unique_ptr<FILE, FileCloser> out(std::fopen(predictions_path.c_str(), "w"));
    if (!out)
        throw std::runtime_error("Cannot write " + predictions_path);
    std::setvbuf(out.get(), buffer.data(), _IOFBF, buffer.size());

    std::cout << "\nEvaluating " << images.size() << " image(s) -> " << predictions_path
              << "\n(conf " << detector.conf_threshold << ", NMS IoU " << detector.iou_threshold
              << ", max_det " << detector.max_detections << ")\n" << std::endl;

    std::fputs("[", out.get());
    bool first = true;
    size_t num_predictions = 0;
    double sum_pre = 0.0, sum_infer = 0.0, sum_post = 0.0;
    const auto start = Clock::now();

    for (size_t i = 0; i < images.size(); ++i)
    {
        const std::string name = fileName(images[i]);
        const long image_id = cocoImageId(name);
        const cv::Mat image = cv::imread(images[i], cv::IMREAD_COLOR);
        if (image.empty())
            throw std::runtime_error("Cannot read " + images[i]);

        const auto t0 = Clock::now();
        const LetterboxInfo lb = letterbox.apply(image, engine.input());
        const auto t1 = Clock::now();
        engine.infer();
        const auto t2 = Clock::now();
        const std::vector<Detection>& detections =
            decoder.process(engine.detections(), engine.prototypes(), lb, image.size());
        const auto t3 = Clock::now();

        sum_pre += msBetween(t0, t1);
        sum_infer += msBetween(t1, t2);
        sum_post += msBetween(t2, t3);

        for (const Detection& d : detections)
        {
            const cv::Rect2f b = toImageBox(d, lb, image.size());
            std::fprintf(out.get(),
                         "%s\n{\"image_id\":%ld,\"category_id\":%d,\"bbox\":[%.2f,%.2f,%.2f,%.2f],"
                         "\"score\":%.5f,\"segmentation\":{\"size\":[%d,%d],\"counts\":",
                         first ? "" : ",", image_id, kCocoCategoryIds[d.class_id],
                         b.x, b.y, b.width, b.height, d.confidence, image.rows, image.cols);
            writeJsonString(out.get(), encodeRle(d.mask, d.box, image.size()));
            std::fputs("}}", out.get());
            first = false;
            ++num_predictions;
        }

        if ((i + 1) % 50 == 0 || i + 1 == images.size())
            std::printf("  %zu/%zu images, %zu predictions\n", i + 1, images.size(), num_predictions);
    }

    std::fputs("\n]\n", out.get());
    if (std::fflush(out.get()) != 0)
        throw std::runtime_error("Failed writing " + predictions_path);
    out.reset();

    const double total_s = msBetween(start, Clock::now()) / 1000.0;
    const double n = static_cast<double>(images.size());

    // Settings and timings; coco_metrics.py copies this into metrics.json.
    const std::string info_path = config.output_dir + "/run_info.json";
    std::unique_ptr<FILE, FileCloser> info(std::fopen(info_path.c_str(), "w"));
    if (!info)
        throw std::runtime_error("Cannot write " + info_path);
    std::fprintf(info.get(),
                 "{\n"
                 "  \"platform\": \"jetson\",\n"
                 "  \"backend\": \"TensorRT %d.%d.%d\",\n"
                 "  \"precision\": \"%s\",\n"
                 "  \"engine\": \"%s\",\n"
                 "  \"input\": \"%dx%d\",\n"
                 "  \"conf\": %g,\n"
                 "  \"iou\": %g,\n"
                 "  \"max_det\": %d,\n"
                 "  \"num_images\": %zu,\n"
                 "  \"mean_preprocess_ms\": %.2f,\n"
                 "  \"mean_inference_ms\": %.2f,\n"
                 "  \"mean_postprocess_ms\": %.2f,\n"
                 "  \"total_seconds\": %.1f\n"
                 "}\n",
                 NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH,
                 precisionFromEngineName(detector.engine_path).c_str(),
                 fileName(detector.engine_path).c_str(),
                 model.input_height, model.input_width,
                 detector.conf_threshold, detector.iou_threshold, detector.max_detections,
                 images.size(), sum_pre / n, sum_infer / n, sum_post / n, total_s);

    std::printf("\nSaved %s (%zu predictions) and run_info.json\n"
                "Mean per image: pre %.1f ms, trt %.1f ms, post %.1f ms\n"
                "Score with: python3 scripts/coco_metrics.py --pred %s\n",
                predictions_path.c_str(), num_predictions,
                sum_pre / n, sum_infer / n, sum_post / n, predictions_path.c_str());
    return 0;
}

}  // namespace edge_ai
