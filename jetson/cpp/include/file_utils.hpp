#pragma once

#include <string>

namespace edge_ai {

// Small path helpers (std::filesystem needs extra linking on GCC 7).

bool isDirectory(const std::string& path);

// mkdir -p
void makeDirectories(const std::string& path);

// "/a/b/clip.mp4" -> "clip.mp4"
std::string fileName(const std::string& path);

// "clip.mp4" -> "clip"
std::string stem(const std::string& name);

// "/a/b/Clip.JPG" -> "jpg"
std::string lowercaseExtension(const std::string& path);

}  // namespace edge_ai
