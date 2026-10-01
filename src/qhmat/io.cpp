#include "qhmat/core.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace qhmat {
namespace {

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;
static_assert(std::numeric_limits<double>::is_iec559 && sizeof(double) == 8,
              "qhmat files require IEEE-754 binary64");

std::size_t checked_size(std::uint64_t value) {
    if (value > std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("file dimension is too large");
    return static_cast<std::size_t>(value);
}

std::size_t words_for(std::size_t rows, int bits) {
    if (rows > (std::numeric_limits<std::size_t>::max() - 31) / std::size_t(bits))
        throw std::runtime_error("packed column size overflow");
    return (rows * std::size_t(bits) + 31) / 32;
}

class Writer {
public:
    explicit Writer(const std::filesystem::path& path) : stream_(path, std::ios::binary) {
        if (!stream_) throw std::runtime_error("cannot create qhmat file");
    }
    void byte(std::uint8_t value) {
        stream_.put(static_cast<char>(value));
        hash_ = (hash_ ^ value) * fnv_prime;
        if (!stream_) throw std::runtime_error("qhmat write failed");
    }
    void uint32(std::uint32_t value) {
        for (int i = 0; i < 4; ++i) byte(static_cast<std::uint8_t>(value >> (8 * i)));
    }
    void uint64(std::uint64_t value) {
        for (int i = 0; i < 8; ++i) byte(static_cast<std::uint8_t>(value >> (8 * i)));
    }
    void real(double value) {
        static_assert(sizeof(double) == sizeof(std::uint64_t), "IEEE-754 binary64 required");
        std::uint64_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        uint64(bits);
    }
    void finish() {
        for (int i = 0; i < 8; ++i) stream_.put(static_cast<char>(hash_ >> (8 * i)));
        stream_.flush();
        if (!stream_) throw std::runtime_error("qhmat write failed");
        stream_.close();
        if (!stream_) throw std::runtime_error("qhmat close failed");
    }
private:
    std::ofstream stream_;
    std::uint64_t hash_ = fnv_offset;
};

class Reader {
public:
    explicit Reader(const std::filesystem::path& path) : stream_(path, std::ios::binary) {
        if (!stream_) throw std::runtime_error("cannot open qhmat file");
        stream_.seekg(0, std::ios::end);
        const auto length = stream_.tellg();
        if (length < 0) throw std::runtime_error("cannot determine qhmat file size");
        remaining_ = static_cast<std::uint64_t>(length);
        stream_.seekg(0);
        if (remaining_ < 24) throw std::runtime_error("truncated qhmat file");
    }
    std::uint64_t remaining_data() const { return remaining_ < 8 ? 0 : remaining_ - 8; }
    std::uint8_t byte() {
        if (remaining_data() == 0) throw std::runtime_error("truncated qhmat file");
        const int value = stream_.get();
        if (value == EOF) throw std::runtime_error("truncated qhmat file");
        --remaining_;
        hash_ = (hash_ ^ static_cast<std::uint8_t>(value)) * fnv_prime;
        return static_cast<std::uint8_t>(value);
    }
    std::uint32_t uint32() {
        std::uint32_t result = 0;
        for (int i = 0; i < 4; ++i) result |= std::uint32_t(byte()) << (8 * i);
        return result;
    }
    std::uint64_t uint64() {
        std::uint64_t result = 0;
        for (int i = 0; i < 8; ++i) result |= std::uint64_t(byte()) << (8 * i);
        return result;
    }
    double real() {
        const auto bits = uint64();
        double value;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }
    void require_bytes(std::size_t count, std::size_t width) const {
        if (width != 0 && count > remaining_data() / width)
            throw std::runtime_error("truncated qhmat payload");
    }
    void finish() {
        if (remaining_ != 8) throw std::runtime_error("trailing qhmat data");
        std::uint64_t checksum = 0;
        for (int i = 0; i < 8; ++i) {
            const int value = stream_.get();
            if (value == EOF) throw std::runtime_error("truncated qhmat checksum");
            checksum |= std::uint64_t(static_cast<std::uint8_t>(value)) << (8 * i);
        }
        if (checksum != hash_) throw std::runtime_error("qhmat checksum mismatch");
    }
private:
    std::ifstream stream_;
    std::uint64_t remaining_ = 0;
    std::uint64_t hash_ = fnv_offset;
};

}  // namespace

void HMatrix::save(const std::string& path) const {
    if (!finalized_) throw std::invalid_argument("matrix must be finalized before saving");
    const auto target = std::filesystem::u8path(path);
    if (std::filesystem::exists(target))
        throw std::invalid_argument("refusing to replace an existing qhmat file");
    static std::atomic<std::uint64_t> sequence{0};
    auto temporary = target;
    temporary += ".partial-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
        "-" + std::to_string(sequence++);
    try {
        Writer writer(temporary);
        for (char c : std::string("QHMAT001")) writer.byte(static_cast<std::uint8_t>(c));
        writer.uint64(1);
        writer.uint64(order_);
        writer.uint64(blocks_.size());
        for (std::size_t index : permutation_) writer.uint64(index);
        for (const auto& block : blocks_) {
            const auto& r = block.rectangle;
            writer.uint64(r.row0); writer.uint64(r.row1);
            writer.uint64(r.col0); writer.uint64(r.col1);
            writer.byte(r.admissible ? 1 : 0);
            writer.byte(block.dense.empty() ? 0 : 1);
            if (!block.dense.empty()) {
                for (double value : block.dense) writer.real(value);
                continue;
            }
            writer.uint64(block.widths.size());
            writer.uint64(block.menu_size);
            writer.real(block.truncation_error);
            writer.real(block.quantization_bound);
            for (int bit : block.widths) writer.byte(static_cast<std::uint8_t>(bit));
            for (double scale : block.left.scales) writer.real(scale);
            for (double scale : block.right.scales) writer.real(scale);
            for (std::uint32_t word : block.left.words) writer.uint32(word);
            for (std::uint32_t word : block.right.words) writer.uint32(word);
        }
        writer.finish();
        std::filesystem::rename(temporary, target);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

HMatrix HMatrix::load(const std::string& path) {
    Reader reader(std::filesystem::u8path(path));
    std::string magic;
    for (int i = 0; i < 8; ++i) magic.push_back(static_cast<char>(reader.byte()));
    if (magic != "QHMAT001") throw std::runtime_error("invalid qhmat file magic");
    if (reader.uint64() != 1) throw std::runtime_error("unsupported qhmat file version");
    const auto order = checked_size(reader.uint64());
    const auto count = checked_size(reader.uint64());
    if (!order || !count || order > reader.remaining_data() / 8)
        throw std::runtime_error("invalid qhmat file dimensions");
    std::vector<std::size_t> permutation(order);
    for (auto& index : permutation) index = checked_size(reader.uint64());
    HMatrix matrix(order, std::move(permutation));
    if (count > reader.remaining_data() / 34)
        throw std::runtime_error("invalid qhmat block count");
    for (std::size_t k = 0; k < count; ++k) {
        Rectangle r;
        r.row0 = checked_size(reader.uint64()); r.row1 = checked_size(reader.uint64());
        r.col0 = checked_size(reader.uint64()); r.col1 = checked_size(reader.uint64());
        const auto admissible = reader.byte(), dense = reader.byte();
        if (admissible > 1 || dense > 1) throw std::runtime_error("invalid qhmat block flag");
        r.admissible = admissible == 1;
        matrix.check_rectangle(r, dense == 1);
        Block block;
        block.rectangle = r;
        const auto rows = r.row1 - r.row0, cols = r.col1 - r.col0;
        if (dense) {
            if (cols > std::numeric_limits<std::size_t>::max() / rows)
                throw std::runtime_error("dense block size overflow");
            const auto total = rows * cols;
            reader.require_bytes(total, 8);
            block.dense.resize(total);
            for (auto& value : block.dense) {
                value = reader.real();
                if (!std::isfinite(value)) throw std::runtime_error("nonfinite dense value");
            }
        } else {
            const auto rank = checked_size(reader.uint64());
            block.menu_size = checked_size(reader.uint64());
            if (rank > std::min(rows, cols) || block.menu_size == 0 || block.menu_size > 31)
                throw std::runtime_error("invalid qhmat rank or precision menu");
            block.truncation_error = reader.real();
            block.quantization_bound = reader.real();
            if (!std::isfinite(block.truncation_error) || block.truncation_error < 0 ||
                !std::isfinite(block.quantization_bound) || block.quantization_bound < 0)
                throw std::runtime_error("invalid qhmat error bound");
            reader.require_bytes(rank, 17);
            block.widths.resize(rank);
            for (auto& bit : block.widths) {
                bit = reader.byte();
                if (bit < 2 || bit > 32) throw std::runtime_error("invalid qhmat bit width");
            }
            block.left.scales.resize(rank);
            block.right.scales.resize(rank);
            for (auto& scale : block.left.scales) scale = reader.real();
            for (auto& scale : block.right.scales) scale = reader.real();
            for (const auto* scales : {&block.left.scales, &block.right.scales})
                for (double scale : *scales)
                    if (!std::isfinite(scale) || scale < 0)
                        throw std::runtime_error("invalid qhmat scale");
            std::size_t left_words = 0, right_words = 0;
            for (int bit : block.widths) {
                block.left.offsets.push_back(left_words);
                block.right.offsets.push_back(right_words);
                const auto left_count = words_for(rows, bit), right_count = words_for(cols, bit);
                if (left_count > std::numeric_limits<std::size_t>::max() - left_words ||
                    right_count > std::numeric_limits<std::size_t>::max() - right_words)
                    throw std::runtime_error("packed block size overflow");
                left_words += left_count;
                right_words += right_count;
            }
            if (right_words > std::numeric_limits<std::size_t>::max() - left_words)
                throw std::runtime_error("packed block size overflow");
            reader.require_bytes(left_words + right_words, 4);
            block.left.words.resize(left_words);
            block.right.words.resize(right_words);
            for (auto& word : block.left.words) word = reader.uint32();
            for (auto& word : block.right.words) word = reader.uint32();
        }
        matrix.blocks_.push_back(std::move(block));
    }
    reader.finish();
    matrix.finalize();
    return matrix;
}

}  // namespace qhmat
