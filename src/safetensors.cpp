#include "safetensors.h"
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace {

struct JsonValue {
    enum Type { Null, Bool, Number, String, Array, Object };
    Type type = Null;
    bool boolean = false;
    double number = 0.0;
    std::string str;
    std::vector<JsonValue> arr;
    std::unordered_map<std::string, JsonValue> obj;
};

class JsonParser {
public:
    explicit JsonParser(const std::string& s) : s_(s), pos_(0) {}

    JsonValue parse() {
        skip_ws();
        JsonValue v = parse_value();
        skip_ws();
        if (pos_ != s_.size()) {
            throw std::runtime_error("Trailing data in JSON");
        }
        return v;
    }

private:
    const std::string& s_;
    size_t pos_;

    std::string context() const {
        size_t start = pos_ > 32 ? pos_ - 32 : 0;
        size_t len = std::min<size_t>(64, s_.size() - start);
        return s_.substr(start, len);
    }

    void skip_ws() {
        while (pos_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[pos_]))) {
            ++pos_;
        }
    }

    char peek() const {
        return pos_ < s_.size() ? s_[pos_] : '\0';
    }

    char get() {
        if (pos_ >= s_.size()) {
            throw std::runtime_error("Unexpected end of JSON");
        }
        return s_[pos_++];
    }

    void expect(char c) {
        if (get() != c) {
            throw std::runtime_error("Unexpected JSON character near: " + context());
        }
    }

    JsonValue parse_value() {
        skip_ws();
        char c = peek();
        if (c == '{') {
            return parse_object();
        }
        if (c == '[') {
            return parse_array();
        }
        if (c == '"') {
            return parse_string();
        }
        if (c == 't' || c == 'f') {
            return parse_bool();
        }
        if (c == 'n') {
            return parse_null();
        }
        return parse_number();
    }

    JsonValue parse_object() {
        JsonValue v;
        v.type = JsonValue::Object;
        expect('{');
        skip_ws();
        if (peek() == '}') {
            get();
            return v;
        }
        while (true) {
            skip_ws();
            JsonValue key = parse_string();
            skip_ws();
            expect(':');
            JsonValue value = parse_value();
            v.obj[key.str] = std::move(value);
            skip_ws();
            char c = get();
            if (c == '}') {
                break;
            }
            if (c != ',') {
                throw std::runtime_error("Invalid JSON object near: " + context());
            }
            skip_ws();
        }
        return v;
    }

    JsonValue parse_array() {
        JsonValue v;
        v.type = JsonValue::Array;
        expect('[');
        skip_ws();
        if (peek() == ']') {
            get();
            return v;
        }
        while (true) {
            skip_ws();
            v.arr.push_back(parse_value());
            skip_ws();
            char c = get();
            if (c == ']') {
                break;
            }
            if (c != ',') {
                throw std::runtime_error("Invalid JSON array near: " + context());
            }
            skip_ws();
        }
        return v;
    }

    JsonValue parse_string() {
        JsonValue v;
        v.type = JsonValue::String;
        expect('"');
        while (pos_ < s_.size()) {
            char c = get();
            if (c == '"') {
                break;
            }
            if (c == '\\') {
                char esc = get();
                switch (esc) {
                    case '"': v.str.push_back('"'); break;
                    case '\\': v.str.push_back('\\'); break;
                    case '/': v.str.push_back('/'); break;
                    case 'b': v.str.push_back('\b'); break;
                    case 'f': v.str.push_back('\f'); break;
                    case 'n': v.str.push_back('\n'); break;
                    case 'r': v.str.push_back('\r'); break;
                    case 't': v.str.push_back('\t'); break;
                    case 'u':
                        for (int i = 0; i < 4; ++i) {
                            if (pos_ >= s_.size()) {
                                throw std::runtime_error("Invalid JSON escape");
                            }
                            ++pos_;
                        }
                        v.str.push_back('?');
                        break;
                    default:
                        throw std::runtime_error("Invalid JSON escape");
                }
            } else {
                v.str.push_back(c);
            }
        }
        return v;
    }

    JsonValue parse_bool() {
        JsonValue v;
        v.type = JsonValue::Bool;
        if (s_.compare(pos_, 4, "true") == 0) {
            pos_ += 4;
            v.boolean = true;
        } else if (s_.compare(pos_, 5, "false") == 0) {
            pos_ += 5;
            v.boolean = false;
        } else {
            throw std::runtime_error("Invalid JSON boolean");
        }
        return v;
    }

    JsonValue parse_null() {
        JsonValue v;
        v.type = JsonValue::Null;
        if (s_.compare(pos_, 4, "null") != 0) {
            throw std::runtime_error("Invalid JSON null");
        }
        pos_ += 4;
        return v;
    }

    JsonValue parse_number() {
        JsonValue v;
        v.type = JsonValue::Number;
        const char* start = s_.c_str() + pos_;
        char* end = nullptr;
        v.number = std::strtod(start, &end);
        if (start == end) {
            throw std::runtime_error("Invalid JSON number");
        }
        pos_ = static_cast<size_t>(end - s_.c_str());
        return v;
    }
};

static uint64_t read_u64_le(std::ifstream& f) {
    uint8_t buf[8];
    f.read(reinterpret_cast<char*>(buf), 8);
    if (!f) {
        throw std::runtime_error("Failed to read safetensors header");
    }
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) {
        v = (v << 8) | buf[i];
    }
    return v;
}

static std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    f.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    std::string data(size, '\0');
    f.read(&data[0], size);
    if (!f) {
        throw std::runtime_error("Failed to read file: " + path);
    }
    return data;
}

static size_t numel(const std::vector<int64_t>& shape) {
    size_t n = 1;
    for (int64_t d : shape) {
        n *= static_cast<size_t>(d);
    }
    return n;
}

static std::vector<int> to_int_shape(const std::vector<int64_t>& shape) {
    std::vector<int> out;
    out.reserve(shape.size());
    for (int64_t d : shape) {
        out.push_back(static_cast<int>(d));
    }
    return out;
}

static float bf16_to_f32(uint16_t v) {
    uint32_t u = static_cast<uint32_t>(v) << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

static float f16_to_f32(uint16_t v) {
    uint32_t sign = (static_cast<uint32_t>(v) & 0x8000u) << 16;
    uint32_t exp = (v >> 10) & 0x1fu;
    uint32_t mant = v & 0x3ffu;
    uint32_t out = 0;

    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            exp = 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3ffu;
            uint32_t exp32 = exp + (127 - 15);
            out = sign | (exp32 << 23) | (mant << 13);
        }
    } else if (exp == 0x1fu) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        uint32_t exp32 = exp + (127 - 15);
        out = sign | (exp32 << 23) | (mant << 13);
    }

    float f = 0.0f;
    std::memcpy(&f, &out, sizeof(float));
    return f;
}

static bool ends_with(const std::string& s, const std::string& suffix) {
    if (s.size() < suffix.size()) {
        return false;
    }
    return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static int64_t as_i64(const JsonValue& v) {
    if (v.type != JsonValue::Number) {
        throw std::runtime_error("Expected JSON number");
    }
    return static_cast<int64_t>(v.number);
}

static const JsonValue& get_obj_field(const JsonValue& obj, const std::string& key) {
    if (obj.type != JsonValue::Object) {
        throw std::runtime_error("Expected JSON object");
    }
    auto it = obj.obj.find(key);
    if (it == obj.obj.end()) {
        throw std::runtime_error("Missing JSON field: " + key);
    }
    return it->second;
}

} // namespace

SafeTensorsLoader::SafeTensorsLoader(const std::string& path) {
    if (ends_with(path, ".json")) {
        JsonValue root;
        try {
            std::string index_data = read_file(path);
            JsonParser parser(index_data);
            root = parser.parse();
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to parse index JSON '" + path + "': " + e.what());
        }
        const JsonValue& weight_map = get_obj_field(root, "weight_map");
        if (weight_map.type != JsonValue::Object) {
            throw std::runtime_error("Invalid weight_map in index file");
        }
        std::vector<std::string> files;
        files.reserve(weight_map.obj.size());
        for (const auto& kv : weight_map.obj) {
            if (kv.second.type != JsonValue::String) {
                throw std::runtime_error("Invalid weight_map entry");
            }
            files.push_back(join_path(dir_name(path), kv.second.str));
        }
        std::sort(files.begin(), files.end());
        files.erase(std::unique(files.begin(), files.end()), files.end());
        for (const auto& file : files) {
            add_file(file);
        }
    } else {
        add_file(path);
    }
}

bool SafeTensorsLoader::has(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

void SafeTensorsLoader::add_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Failed to open safetensors file: " + path);
    }

    uint64_t header_len = read_u64_le(f);
    std::string header(header_len, '\0');
    f.read(&header[0], static_cast<std::streamsize>(header_len));
    if (!f) {
        throw std::runtime_error("Failed to read safetensors header: " + path);
    }

    JsonValue root;
    try {
        JsonParser parser(header);
        root = parser.parse();
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to parse safetensors header '" + path + "': " + e.what());
    }
    if (root.type != JsonValue::Object) {
        throw std::runtime_error("Invalid safetensors header JSON");
    }

    for (const auto& kv : root.obj) {
        if (kv.first == "__metadata__") {
            continue;
        }
        const JsonValue& entry = kv.second;
        const JsonValue& dtype_v = get_obj_field(entry, "dtype");
        const JsonValue& shape_v = get_obj_field(entry, "shape");
        const JsonValue& offsets_v = get_obj_field(entry, "data_offsets");

        if (dtype_v.type != JsonValue::String || shape_v.type != JsonValue::Array || offsets_v.type != JsonValue::Array) {
            throw std::runtime_error("Invalid tensor entry in safetensors header");
        }
        if (offsets_v.arr.size() != 2) {
            throw std::runtime_error("Invalid data_offsets size");
        }

        std::vector<int64_t> shape;
        shape.reserve(shape_v.arr.size());
        for (const auto& dim : shape_v.arr) {
            shape.push_back(as_i64(dim));
        }

        int64_t offset_start = as_i64(offsets_v.arr[0]);
        int64_t offset_end = as_i64(offsets_v.arr[1]);
        size_t data_start = static_cast<size_t>(8 + header_len + offset_start);
        size_t data_end = static_cast<size_t>(8 + header_len + offset_end);

        if (tensors_.find(kv.first) != tensors_.end()) {
            throw std::runtime_error("Duplicate tensor in safetensors: " + kv.first);
        }

        TensorInfo info;
        info.dtype = dtype_v.str;
        info.shape = std::move(shape);
        info.data_start = data_start;
        info.data_end = data_end;
        info.file_path = path;
        tensors_.emplace(kv.first, std::move(info));
    }
}

void SafeTensorsLoader::load_into(const std::string& name, Tensor& dst, bool transpose) {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("Missing tensor: " + name);
    }
    const TensorInfo& info = it->second;

    std::vector<int> expected_shape;
    if (transpose) {
        if (info.shape.size() != 2) {
            throw std::runtime_error("Transpose expects 2D tensor: " + name);
        }
        expected_shape = {static_cast<int>(info.shape[1]), static_cast<int>(info.shape[0])};
    } else {
        expected_shape = to_int_shape(info.shape);
    }

    if (dst.shape != expected_shape) {
        throw std::runtime_error("Shape mismatch for tensor: " + name);
    }

    size_t n = numel(info.shape);
    size_t bytes_per_elem = 0;
    if (info.dtype == "F32") {
        bytes_per_elem = 4;
    } else if (info.dtype == "F16" || info.dtype == "BF16") {
        bytes_per_elem = 2;
    } else {
        throw std::runtime_error("Unsupported dtype: " + info.dtype);
    }

    size_t expected_bytes = n * bytes_per_elem;
    if (info.data_end < info.data_start || (info.data_end - info.data_start) != expected_bytes) {
        throw std::runtime_error("Size mismatch for tensor: " + name);
    }

    std::ifstream f(info.file_path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Failed to open safetensors file: " + info.file_path);
    }
    f.seekg(static_cast<std::streamoff>(info.data_start), std::ios::beg);

    std::vector<float> tmp(n);
    if (info.dtype == "F32") {
        f.read(reinterpret_cast<char*>(tmp.data()), static_cast<std::streamsize>(expected_bytes));
        if (!f) {
            throw std::runtime_error("Failed to read tensor data: " + name);
        }
    } else {
        std::vector<uint16_t> raw(n);
        f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(expected_bytes));
        if (!f) {
            throw std::runtime_error("Failed to read tensor data: " + name);
        }
        if (info.dtype == "BF16") {
            for (size_t i = 0; i < n; ++i) {
                tmp[i] = bf16_to_f32(raw[i]);
            }
        } else {
            for (size_t i = 0; i < n; ++i) {
                tmp[i] = f16_to_f32(raw[i]);
            }
        }
    }

    if (transpose) {
        int64_t out = info.shape[0];
        int64_t in = info.shape[1];
        for (int64_t i = 0; i < out; ++i) {
            for (int64_t j = 0; j < in; ++j) {
                dst.data[static_cast<size_t>(j * out + i)] = tmp[static_cast<size_t>(i * in + j)];
            }
        }
    } else {
        std::copy(tmp.begin(), tmp.end(), dst.data.begin());
    }
}

std::string SafeTensorsLoader::dir_name(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return "";
    }
    return path.substr(0, pos);
}

std::string SafeTensorsLoader::join_path(const std::string& dir, const std::string& file) {
    if (dir.empty()) {
        return file;
    }
    if (!file.empty() && (file[0] == '/' || file[0] == '\\')) {
        return file;
    }
    return dir + "/" + file;
}
