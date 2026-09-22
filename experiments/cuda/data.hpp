#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace qmat {
inline void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
inline bool valid_bits(int b) { return b == 4 || b == 8 || b == 12 || b == 16; }
struct Data {
    int m, n, rank;
    std::vector<uint32_t> bits;
    std::vector<float> p, q; // Column-major factors P(m,r), Q(n,r).
};
template<class T> void read_values(std::istream& in, T* ptr, size_t count) {
    in.read(reinterpret_cast<char*>(ptr), sizeof(T) * count);
    require(bool(in), "truncated input");
}
inline Data read_data(const std::string& path) {
    const uint32_t endian = 1;
    require(*reinterpret_cast<const uint8_t*>(&endian) == 1, "little-endian host required");
    std::ifstream in(path, std::ios::binary);
    char magic[8]; read_values(in, magic, 8);
    require(std::string(magic, 8) == "QMATGPU1", "invalid data magic");
    uint32_t dims[3]; read_values(in, dims, 3);
    for (auto d : dims) require(d > 0 && d <= 65536, "invalid dimension");
    Data a{int(dims[0]), int(dims[1]), int(dims[2]), {}, {}, {}};
    const size_t count = size_t(a.m + a.n) * a.rank;
    const auto start = in.tellg();
    in.seekg(0, std::ios::end);
    require(in.tellg() - start == std::streamoff((count + a.rank) * 4), "invalid file size");
    in.seekg(start);
    a.bits.resize(a.rank); a.p.resize(size_t(a.m) * a.rank); a.q.resize(size_t(a.n) * a.rank);
    read_values(in, a.bits.data(), a.bits.size());
    read_values(in, a.p.data(), a.p.size()); read_values(in, a.q.data(), a.q.size());
    for (auto b : a.bits) require(valid_bits(b), "unsupported bit width");
    for (const auto* v : {&a.p, &a.q})
        for (float x : *v) require(std::isfinite(x), "nonfinite input");
    return a;
}
inline Data reorder(Data a, const std::string& mode) {
    if (mode != "adaptive" && mode != "mixed") {
        require(mode.rfind("uniform", 0) == 0, "unknown allocation mode");
        int b = std::stoi(mode.substr(7)); require(valid_bits(b), "unsupported bit width");
        std::fill(a.bits.begin(), a.bits.end(), b);
    } else if (mode == "mixed") {
        for (int j = 0; j < a.rank; ++j) a.bits[j] = 4 * (j % 4 + 1);
    }
    std::vector<int> order(a.rank); std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int x, int y) { return a.bits[x] < a.bits[y]; });
    Data out = a;
    for (int j = 0; j < a.rank; ++j) {
        out.bits[j] = a.bits[order[j]];
        std::copy_n(a.p.data() + size_t(order[j]) * a.m, a.m, out.p.data() + size_t(j) * a.m);
        std::copy_n(a.q.data() + size_t(order[j]) * a.n, a.n, out.q.data() + size_t(j) * a.n);
    }
    return out;
}
struct Packed {
    std::vector<uint32_t> words;
    std::vector<uint64_t> offsets;
    std::vector<double> scales, max_abs;
    std::vector<double> exact;
};
inline Packed pack(const std::vector<float>& x, int rows, const std::vector<uint32_t>& bits) {
    Packed out;
    out.exact.resize(x.size());
    for (size_t j = 0; j < bits.size(); ++j) {
        int b = bits[j]; require(valid_bits(b), "unsupported bit width");
        const int limit = (1 << (b - 1)) - 1;
        double largest = 0;
        for (int i = 0; i < rows; ++i) largest = std::max(largest, std::abs(double(x[j * rows + i])));
        double scale = largest == 0 ? 0 : largest / limit;
        out.max_abs.push_back(largest); out.scales.push_back(scale);
        size_t offset = out.words.size(); out.offsets.push_back(offset);
        out.words.resize(offset + (size_t(rows) * b + 31) / 32, 0);
        for (int i = 0; i < rows; ++i) {
            int value = scale == 0 ? 0 : int(std::nearbyint(double(x[j * rows + i]) / scale));
            value = std::clamp(value, -limit, limit);
            uint32_t raw = uint32_t(value) & ((1u << b) - 1);
            size_t bit = size_t(i) * b, word = offset + bit / 32;
            int shift = bit % 32;
            out.words[word] |= raw << shift;
            if (shift + b > 32) out.words[word + 1] |= raw >> (32 - shift);
            out.exact[j * rows + i] = value * scale;
        }
    }
    return out;
}
inline int unpack(const Packed& a, int i, int j, int b) {
    size_t bit = size_t(i) * b, word = a.offsets[j] + bit / 32;
    int shift = bit % 32;
    uint32_t raw = a.words[word] >> shift;
    if (shift + b > 32) raw |= a.words[word + 1] << (32 - shift);
    raw &= b == 32 ? UINT32_MAX : (1u << b) - 1;
    return raw & (1u << (b - 1)) ? int(int64_t(raw) - (int64_t(1) << b)) : int(raw);
}
inline double gamma(int n, double u) {
    require(n * u < 1, "invalid rounding bound");
    return n * u / (1 - n * u);
}
struct Bounds { double quant = 0, arithmetic = 0, reference_slack = 0; };
inline Bounds bounds(const Data& a, const Packed& p, const Packed& q) {
    Bounds out;
    int channels = 1;
    for (int j = 1; j < a.rank; ++j) channels += a.bits[j] != a.bits[j - 1];
    const double u = std::ldexp(1.0, -24), ud = std::ldexp(1.0, -53);
    for (int start = 0; start < a.rank;) {
        int end = start + 1;
        while (end < a.rank && a.bits[end] == a.bits[start]) ++end;
        double g = gamma(end - start, u), h = gamma(channels - 1, u);
        for (int j = start; j < end; ++j) {
            double limit = (1 << (a.bits[j] - 1)) - 1;
            double base = p.max_abs[j] * q.max_abs[j];
            double weight = base * std::pow(1 + 1 / (2 * limit), 2);
            out.quant += base * (1 / limit + 1 / (4 * limit * limit));
            // FP64 scale multiplication followed by FP32 conversion, conservatively two roundings.
            double repr = gamma(2, u);
            out.arithmetic += weight * ((2 * repr + repr * repr) +
                              std::pow(1 + repr, 2) * (g + h * (1 + g)));
            out.reference_slack += 16 * gamma(a.rank + 2, ud) * (base + weight);
        }
        start = end;
    }
    return out;
}
inline Data fixture(int m = 31, int n = 29, int rank = 13) {
    Data a{m, n, rank, std::vector<uint32_t>(rank, 8),
           std::vector<float>(size_t(m) * rank), std::vector<float>(size_t(n) * rank)};
    for (int j = 0; j < rank; ++j) {
        for (int i = 0; i < m; ++i) a.p[j * m + i] = j == 0 ? 0 : std::sin(0.3 * (i + 1) * j) / (j + 1);
        for (int i = 0; i < n; ++i) a.q[j * n + i] = std::cos(0.7 * (i + 1) * j);
    }
    return a;
}
} // namespace qmat
