#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include "tensor.h"

class SafeTensorsLoader {
public:
    explicit SafeTensorsLoader(const std::string& path);
    bool has(const std::string& name) const;
    void load_into(const std::string& name, Tensor& dst, bool transpose);

private:
    struct TensorInfo {
        std::string dtype;
        std::vector<int64_t> shape;
        size_t data_start;
        size_t data_end;
        std::string file_path;
    };

    std::unordered_map<std::string, TensorInfo> tensors_;

    void add_file(const std::string& path);
    static std::string dir_name(const std::string& path);
    static std::string join_path(const std::string& dir, const std::string& file);
};
