#include "file_utils.hpp"

#include <sys/stat.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <stdexcept>

namespace edge_ai {

bool isDirectory(const std::string& path)
{
    struct stat info {};
    return ::stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
}

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

}  // namespace edge_ai
