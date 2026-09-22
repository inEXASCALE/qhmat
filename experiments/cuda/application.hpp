#pragma once
#include "data.hpp"
#include <array>
#include <map>

namespace qmat {
struct AppLeaf {
    int row0, row1, col0, col1, rank;
    std::vector<uint32_t> bits;
    std::vector<float> p, q, dense;
    Packed pp, pq;
    int m() const { return row1-row0; }
    int n() const { return col1-col0; }
};
struct Application {
    int n, rhs;
    std::vector<uint32_t> permutation;
    std::vector<float> dense, x;
    std::array<std::vector<double>, 4> references;
    std::array<std::vector<double>, 3> allowances, model_bounds;
    std::vector<AppLeaf> leaves;
};
class AppReader {
    std::ifstream in;
    size_t remaining;
public:
    explicit AppReader(const std::string& path) : in(path, std::ios::binary) {
        require(bool(in), "cannot open application input");
        in.seekg(0, std::ios::end); remaining = size_t(in.tellg()); in.seekg(0);
    }
    template<class T> std::vector<T> get(size_t count) {
        require(count <= remaining / sizeof(T), "truncated application input");
        std::vector<T> v(count); read_values(in, v.data(), count); remaining -= count*sizeof(T);
        return v;
    }
    void finish() { require(remaining == 0, "trailing application data"); }
};
inline Application read_application(const std::string& path) {
    const uint32_t endian = 1;
    require(*reinterpret_cast<const uint8_t*>(&endian) == 1, "little-endian host required");
    AppReader reader(path);
    auto magic = reader.get<char>(8);
    require(std::string(magic.begin(), magic.end()) == "QMATAPP1", "invalid application magic");
    auto dims = reader.get<uint32_t>(3);
    require(dims[0] > 0 && dims[0] <= 16384 && dims[1] > 0 && dims[1] <= 4096 &&
            dims[2] > 0 && dims[2] <= 1000000, "invalid application dimensions");
    Application a; a.n = int(dims[0]); a.rhs = int(dims[1]);
    a.permutation = reader.get<uint32_t>(a.n);
    auto sorted = a.permutation; std::sort(sorted.begin(), sorted.end());
    for (int i=0; i<a.n; ++i) require(sorted[i] == uint32_t(i), "invalid permutation");
    a.dense = reader.get<float>(size_t(a.n)*a.n);
    a.x = reader.get<float>(size_t(a.n)*a.rhs);
    for (auto& ref : a.references) ref = reader.get<double>(a.x.size());
    for (auto* group : {&a.allowances, &a.model_bounds})
        for (auto& bound : *group) bound = reader.get<double>(a.rhs);
    auto packed = [&](int rows, const std::vector<uint32_t>& bits) {
        Packed p;
        auto count = reader.get<uint64_t>(1)[0];
        p.offsets = reader.get<uint64_t>(bits.size());
        p.scales = reader.get<double>(bits.size());
        size_t expected = 0;
        for (size_t j=0; j<bits.size(); ++j) {
            require(p.offsets[j] == expected, "invalid packed column offset");
            require(std::isfinite(p.scales[j]) && p.scales[j] >= 0, "invalid packed scale");
            expected += (size_t(rows)*bits[j]+31)/32;
        }
        require(count == expected, "invalid packed word count");
        p.words = reader.get<uint32_t>(expected);
        return p;
    };
    for (uint32_t k=0; k<dims[2]; ++k) {
        auto v = reader.get<uint32_t>(5);
        require(v[0] < v[1] && v[1] <= dims[0] && v[2] < v[3] && v[3] <= dims[0] &&
                v[4] <= std::min(v[1]-v[0], v[3]-v[2]), "invalid leaf dimensions");
        AppLeaf l{int(v[0]), int(v[1]), int(v[2]), int(v[3]), int(v[4]), {}, {}, {}, {}, {}, {}};
        if (!l.rank) l.dense = reader.get<float>(size_t(l.m())*l.n());
        else {
            l.bits = reader.get<uint32_t>(l.rank);
            for (auto b : l.bits) require(valid_bits(b) || b == 24 || b == 32, "invalid application bit width");
            l.p = reader.get<float>(size_t(l.m())*l.rank);
            l.q = reader.get<float>(size_t(l.n())*l.rank);
            l.pp = packed(l.m(), l.bits); l.pq = packed(l.n(), l.bits);
        }
        a.leaves.push_back(std::move(l));
    }
    reader.finish();
    for (int row=0; row<a.n; ++row) {
        std::vector<std::pair<int,int>> intervals;
        for (auto& l : a.leaves) if (l.row0 <= row && row < l.row1) intervals.emplace_back(l.col0,l.col1);
        std::sort(intervals.begin(),intervals.end()); int end = 0;
        for (auto p : intervals) { require(p.first == end, "partition gap/overlap"); end = p.second; }
        require(end == a.n, "partition incomplete");
    }
    auto finite = [](const auto& values) { for (auto x : values) require(std::isfinite(x), "nonfinite application value"); };
    finite(a.dense); finite(a.x);
    for (auto& ref : a.references) finite(ref);
    for (auto* group : {&a.allowances, &a.model_bounds}) for (auto& b : *group) {
        finite(b); for (double x : b) require(x >= 0, "negative allowance");
    }
    for (auto& l : a.leaves) { finite(l.dense); finite(l.p); finite(l.q); }
    return a;
}
inline float decoded(const Packed& p, int row, int col, int bits) {
    float value = float(double(unpack(p,row,col,bits))*p.scales[col]);
    require(std::isfinite(value), "decoded factor overflow");
    return value;
}
inline std::vector<double> apply_host(const Application& a, int rhs, bool quantized) {
    require(rhs > 0 && rhs <= a.rhs, "invalid RHS count");
    std::vector<double> y(size_t(a.n)*rhs, 0);
    for (const auto& l : a.leaves) for (int j=0; j<rhs; ++j) {
        std::vector<double> z(l.rank, 0);
        for (int r=0; r<l.rank; ++r) for (int k=0; k<l.n(); ++k)
            z[r] += double(quantized ? decoded(l.pq,k,r,l.bits[r]) : l.q[k+size_t(r)*l.n()]) *
                    a.x[a.permutation[l.col0+k]+size_t(j)*a.n];
        for (int i=0; i<l.m(); ++i) {
            double value=0;
            if (l.rank) for (int r=0; r<l.rank; ++r)
                value += double(quantized ? decoded(l.pp,i,r,l.bits[r]) : l.p[i+size_t(r)*l.m()])*z[r];
            else for (int k=0; k<l.n(); ++k)
                value += double(l.dense[i+size_t(k)*l.m()])*a.x[a.permutation[l.col0+k]+size_t(j)*a.n];
            y[a.permutation[l.row0+i]+size_t(j)*a.n] += value;
        }
    }
    return y;
}
} // namespace qmat
