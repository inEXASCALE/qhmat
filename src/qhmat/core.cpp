#include "qhmat/core.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace qhmat {
namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a)
        throw std::overflow_error("storage count overflow");
    return a + b;
}

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b) {
    if (b != 0 && a > std::numeric_limits<std::uint64_t>::max() / b)
        throw std::overflow_error("storage count overflow");
    return a * b;
}

std::uint64_t limit(int bits) {
    require(bits >= 2 && bits <= 32, "bit width must be in [2, 32]");
    return (std::uint64_t{1} << (bits - 1)) - 1;
}

std::size_t label_bits(std::size_t menu_size) {
    require(menu_size > 0 && menu_size <= 31, "precision menu size must be in [1, 31]");
    std::size_t bits = 0;
    for (std::size_t capacity = 1; capacity < menu_size; capacity *= 2) ++bits;
    return bits;
}

double norm(const std::vector<double>& lower, const std::vector<double>& upper,
            std::size_t dimension) {
    double sum = 0.0;
    for (std::size_t d = 0; d < dimension; ++d) {
        double gap = upper[d] - lower[d];
        sum += gap * gap;
    }
    return std::sqrt(sum);
}

}  // namespace

double coefficient(int bits) {
    const double q = static_cast<double>(limit(bits));
    return 1.0 / q + 1.0 / (4.0 * q * q);
}

Allocation allocate(const std::vector<double>& singular_values,
                    const std::vector<int>& bit_widths, double budget,
                    std::size_t rows, std::size_t cols) {
    require(rows > 0 && cols > 0, "matrix dimensions must be positive");
    require(std::isfinite(budget) && budget >= 0.0, "budget must be finite and nonnegative");
    require(singular_values.size() <= std::min(rows, cols), "rank exceeds block dimension");
    std::vector<int> menu = bit_widths;
    require(!menu.empty(), "precision menu must not be empty");
    std::sort(menu.begin(), menu.end());
    require(std::adjacent_find(menu.begin(), menu.end()) == menu.end(),
            "precision menu contains duplicates");
    for (int bit : menu) limit(bit);
    for (std::size_t j = 0; j < singular_values.size(); ++j) {
        require(std::isfinite(singular_values[j]) && singular_values[j] >= 0.0,
                "singular values must be finite and nonnegative");
        if (j) require(singular_values[j - 1] >= singular_values[j],
                       "singular values must be sorted descending");
    }

    const std::size_t rank = singular_values.size();
    std::vector<long double> prefix(rank + 1, 0.0L);
    for (std::size_t j = 0; j < rank; ++j)
        prefix[j + 1] = prefix[j] + singular_values[j];
    const long double high_coefficient = coefficient(menu.back());
    Allocation out;
    out.bits.assign(rank, menu.back());
    if (high_coefficient * prefix[rank] <= static_cast<long double>(budget)) {
        std::size_t remaining = rank;
        long double available = budget;
        for (std::size_t level = 0; level + 1 < menu.size(); ++level) {
            const long double c = coefficient(menu[level]);
            long double channel_cost = 0.0L;
            while (remaining) {
                const long double next = channel_cost + c * singular_values[remaining - 1]
                                         + high_coefficient * prefix[remaining - 1];
                if (next > available) break;
                out.bits[remaining - 1] = menu[level];
                channel_cost += c * singular_values[remaining - 1];
                --remaining;
            }
            available -= channel_cost;
        }
    }

    long double bound = 0.0L;
    std::uint64_t sum_bits = 0;
    for (std::size_t j = 0; j < rank; ++j) {
        bound += static_cast<long double>(singular_values[j]) * coefficient(out.bits[j]);
        sum_bits = checked_add(sum_bits, static_cast<std::uint64_t>(out.bits[j]));
    }
    out.bound = static_cast<double>(bound);
    out.payload_bits = checked_mul(checked_add(rows, cols), sum_bits);
    out.feasible = bound <= static_cast<long double>(budget);
    return out;
}

Partition spatial_partition(const std::vector<double>& points,
                            const std::vector<double>& support_lower,
                            const std::vector<double>& support_upper,
                            std::size_t count, std::size_t dimension,
                            std::size_t leaf_size, double eta) {
    require(count > 0 && dimension > 0 && dimension <= 16,
            "invalid point-cloud dimensions");
    require(count <= std::numeric_limits<std::size_t>::max() / dimension,
            "point-cloud dimensions overflow");
    require(points.size() == count * dimension &&
            support_lower.size() == points.size() && support_upper.size() == points.size(),
            "point/support shapes differ");
    require(leaf_size > 0 && std::isfinite(eta) && eta > 0,
            "leaf size and admissibility must be positive");
    for (std::size_t i = 0; i < points.size(); ++i)
        require(std::isfinite(points[i]) && std::isfinite(support_lower[i]) &&
                std::isfinite(support_upper[i]) && support_lower[i] <= points[i] &&
                points[i] <= support_upper[i], "invalid point or support box");

    struct Cluster {
        std::size_t first, last;
        std::vector<double> lower, upper;
        int left = -1, right = -1;
    };
    Partition out;
    std::vector<Cluster> tree;
    std::vector<std::size_t> initial(count);
    std::iota(initial.begin(), initial.end(), 0);

    std::function<int(std::vector<std::size_t>)> build = [&](std::vector<std::size_t> indices) {
        Cluster c;
        c.first = out.permutation.size();
        c.lower.assign(dimension, std::numeric_limits<double>::infinity());
        c.upper.assign(dimension, -std::numeric_limits<double>::infinity());
        std::vector<double> min_point = c.lower, max_point = c.upper;
        for (std::size_t i : indices)
            for (std::size_t d = 0; d < dimension; ++d) {
                const std::size_t offset = i * dimension + d;
                c.lower[d] = std::min(c.lower[d], support_lower[offset]);
                c.upper[d] = std::max(c.upper[d], support_upper[offset]);
                min_point[d] = std::min(min_point[d], points[offset]);
                max_point[d] = std::max(max_point[d], points[offset]);
            }
        const int id = static_cast<int>(tree.size());
        tree.push_back(std::move(c));
        if (indices.size() <= leaf_size) {
            out.permutation.insert(out.permutation.end(), indices.begin(), indices.end());
        } else {
            std::size_t axis = 0;
            for (std::size_t d = 1; d < dimension; ++d)
                if (max_point[d] - min_point[d] > max_point[axis] - min_point[axis]) axis = d;
            std::stable_sort(indices.begin(), indices.end(), [&](std::size_t a, std::size_t b) {
                return points[a * dimension + axis] < points[b * dimension + axis];
            });
            const auto middle = indices.begin() + indices.size() / 2;
            tree[id].left = build(std::vector<std::size_t>(indices.begin(), middle));
            tree[id].right = build(std::vector<std::size_t>(middle, indices.end()));
        }
        tree[id].last = out.permutation.size();
        return id;
    };
    const int root = build(std::move(initial));

    std::function<void(int, int)> visit = [&](int ai, int bi) {
        const auto& a = tree[ai];
        const auto& b = tree[bi];
        double distance2 = 0.0;
        for (std::size_t d = 0; d < dimension; ++d) {
            const double gap = std::max({0.0, a.lower[d] - b.upper[d],
                                         b.lower[d] - a.upper[d]});
            distance2 += gap * gap;
        }
        const double distance = std::sqrt(distance2);
        const double diameter = std::max(norm(a.lower, a.upper, dimension),
                                         norm(b.lower, b.upper, dimension));
        const bool admissible = distance > 0.0 && distance >= eta * diameter;
        if (admissible || (a.left < 0 && b.left < 0)) {
            out.blocks.push_back({a.first, a.last, b.first, b.last, admissible});
            return;
        }
        const int aa[] = {a.left < 0 ? ai : a.left, a.left < 0 ? -1 : a.right};
        const int bb[] = {b.left < 0 ? bi : b.left, b.left < 0 ? -1 : b.right};
        for (int x : aa) for (int y : bb) if (x >= 0 && y >= 0) visit(x, y);
    };
    visit(root, root);
    return out;
}

HMatrix::HMatrix(std::size_t order, std::vector<std::size_t> permutation)
    : order_(order), permutation_(std::move(permutation)) {
    require(order_ > 0 && permutation_.size() == order_, "invalid matrix order/permutation");
    auto sorted = permutation_;
    std::sort(sorted.begin(), sorted.end());
    for (std::size_t i = 0; i < order_; ++i)
        require(sorted[i] == i, "permutation is not a bijection");
}

void HMatrix::check_rectangle(Rectangle r, bool diagonal) const {
    require(!finalized_, "matrix has been finalized");
    require(r.row0 < r.row1 && r.row1 <= order_ && r.col0 < r.col1 &&
            r.col1 <= order_, "invalid block rectangle");
    const bool same = r.row0 == r.col0 && r.row1 == r.col1;
    const bool disjoint = r.row1 <= r.col0 || r.col1 <= r.row0;
    require(diagonal ? same : disjoint,
            "only diagonal blocks may be dense; factorized blocks must be off-diagonal");
}

void HMatrix::add_dense(Rectangle r, const std::vector<double>& values) {
    check_rectangle(r, true);
    require(values.size() == checked_mul(r.row1 - r.row0, r.col1 - r.col0),
            "dense block shape mismatch");
    for (double value : values) require(std::isfinite(value), "nonfinite dense block");
    Block b;
    b.rectangle = r;
    b.dense = values;
    blocks_.push_back(std::move(b));
}

HMatrix::PackedFactor HMatrix::pack(const std::vector<double>& matrix,
                                    std::size_t rows, const std::vector<int>& widths) {
    PackedFactor out;
    out.scales.reserve(widths.size());
    out.offsets.reserve(widths.size());
    for (std::size_t j = 0; j < widths.size(); ++j) {
        const int bits = widths[j];
        const std::uint64_t q = limit(bits);
        double maximum = 0.0;
        for (std::size_t i = 0; i < rows; ++i)
            maximum = std::max(maximum, std::abs(matrix[j * rows + i]));
        const double scale = maximum == 0 ? 0 : maximum / static_cast<double>(q);
        out.scales.push_back(scale);
        out.offsets.push_back(out.words.size());
        const std::size_t payload_bits = checked_mul(rows, std::size_t(bits));
        const std::size_t word_count = checked_add(payload_bits, 31) / 32;
        out.words.resize(out.words.size() + word_count, 0);
        for (std::size_t i = 0; i < rows; ++i) {
            const double scaled = scale == 0 ? 0 : matrix[j * rows + i] / scale;
            const auto rounded = static_cast<std::int64_t>(std::nearbyint(scaled));
            const auto clipped = std::max(-static_cast<std::int64_t>(q),
                                          std::min(static_cast<std::int64_t>(q), rounded));
            const std::uint32_t mask = bits == 32 ? UINT32_MAX : (std::uint32_t{1} << bits) - 1;
            const std::uint32_t raw = static_cast<std::uint32_t>(clipped) & mask;
            const std::size_t position = i * std::size_t(bits);
            const std::size_t word = out.offsets[j] + position / 32;
            const unsigned shift = static_cast<unsigned>(position % 32);
            out.words[word] |= raw << shift;
            if (shift + bits > 32) out.words[word + 1] |= raw >> (32 - shift);
        }
    }
    return out;
}

double HMatrix::decode(const PackedFactor& factor, std::size_t row,
                       std::size_t column, std::size_t, int bits) {
    const std::size_t position = row * std::size_t(bits);
    const std::size_t word = factor.offsets[column] + position / 32;
    const unsigned shift = static_cast<unsigned>(position % 32);
    std::uint32_t raw = factor.words[word] >> shift;
    if (shift + bits > 32) raw |= factor.words[word + 1] << (32 - shift);
    const std::uint32_t mask = bits == 32 ? UINT32_MAX : (std::uint32_t{1} << bits) - 1;
    raw &= mask;
    const std::int64_t signed_value = (raw & (std::uint32_t{1} << (bits - 1)))
        ? static_cast<std::int64_t>(raw) - (std::int64_t{1} << bits)
        : static_cast<std::int64_t>(raw);
    return static_cast<double>(signed_value) * factor.scales[column];
}

void HMatrix::add_low_rank(Rectangle r, const std::vector<double>& u,
                           const std::vector<double>& singular_values,
                           const std::vector<double>& v,
                           const std::vector<int>& widths, std::size_t menu_size,
                           double truncation_error) {
    check_rectangle(r, false);
    const std::size_t m = r.row1 - r.row0, n = r.col1 - r.col0;
    const std::size_t rank = singular_values.size();
    require(rank <= std::min(m, n) && widths.size() == rank &&
            u.size() == checked_mul(m, rank) && v.size() == checked_mul(n, rank),
            "factor dimensions do not match block rank");
    require(std::isfinite(truncation_error) && truncation_error >= 0.0,
            "invalid truncation error");
    label_bits(menu_size);
    std::vector<double> left(u.size()), right(v.size());
    Block b;
    b.rectangle = r;
    b.widths = widths;
    b.menu_size = menu_size;
    b.truncation_error = truncation_error;
    for (std::size_t j = 0; j < rank; ++j) {
        require(std::isfinite(singular_values[j]) && singular_values[j] >= 0.0,
                "singular values must be finite and nonnegative");
        if (j) require(singular_values[j - 1] >= singular_values[j],
                       "singular values must be sorted descending");
        limit(widths[j]);
        double u2 = 0.0, v2 = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            const double value = u[j * m + i];
            require(std::isfinite(value), "nonfinite left factor");
            u2 += value * value;
            left[j * m + i] = std::sqrt(singular_values[j]) * value;
        }
        for (std::size_t i = 0; i < n; ++i) {
            const double value = v[j * n + i];
            require(std::isfinite(value), "nonfinite right factor");
            v2 += value * value;
            right[j * n + i] = std::sqrt(singular_values[j]) * value;
        }
        require(u2 <= 1.0 + 1e-8 && v2 <= 1.0 + 1e-8,
                "singular vectors must have norm at most one");
        b.quantization_bound += singular_values[j] * coefficient(widths[j]);
    }
    b.left = pack(left, m, widths);
    b.right = pack(right, n, widths);
    blocks_.push_back(std::move(b));
}

void HMatrix::finalize() {
    require(!finalized_, "matrix has already been finalized");
    require(!blocks_.empty(), "matrix has no blocks");
    std::vector<std::size_t> boundaries{0, order_};
    for (const auto& block : blocks_) {
        boundaries.push_back(block.rectangle.row0);
        boundaries.push_back(block.rectangle.row1);
    }
    std::sort(boundaries.begin(), boundaries.end());
    boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());
    for (std::size_t slab = 0; slab + 1 < boundaries.size(); ++slab) {
        std::vector<std::pair<std::size_t, std::size_t>> intervals;
        for (const auto& block : blocks_) {
            const auto& r = block.rectangle;
            if (r.row0 <= boundaries[slab] && boundaries[slab] < r.row1)
                intervals.emplace_back(r.col0, r.col1);
        }
        std::sort(intervals.begin(), intervals.end());
        std::size_t end = 0;
        for (const auto& interval : intervals) {
            require(interval.first == end, "block partition has a gap or overlap");
            end = interval.second;
        }
        require(end == order_, "block partition does not cover the matrix");
    }
    finalized_ = true;
}

std::vector<double> HMatrix::matmul(const std::vector<double>& x,
                                    std::size_t rhs) const {
    require(finalized_, "matrix must be finalized before use");
    require(rhs > 0 && rhs <= std::numeric_limits<std::size_t>::max() / order_
            && x.size() == order_ * rhs, "right-hand-side shape mismatch");
    for (double value : x) require(std::isfinite(value), "nonfinite right-hand side");
    std::vector<double> y(order_ * rhs, 0.0);
    for (const auto& block : blocks_) {
        const auto& r = block.rectangle;
        const std::size_t m = r.row1 - r.row0, n = r.col1 - r.col0;
        if (!block.dense.empty()) {
            for (std::size_t i = 0; i < m; ++i)
                for (std::size_t k = 0; k < n; ++k)
                    for (std::size_t p = 0; p < rhs; ++p)
                        y[permutation_[r.row0 + i] * rhs + p] +=
                            block.dense[i * n + k] * x[permutation_[r.col0 + k] * rhs + p];
        } else {
            for (std::size_t p = 0; p < rhs; ++p)
                for (std::size_t j = 0; j < block.widths.size(); ++j) {
                    double dot = 0.0;
                    for (std::size_t k = 0; k < n; ++k)
                        dot += decode(block.right, k, j, n, block.widths[j]) *
                               x[permutation_[r.col0 + k] * rhs + p];
                    for (std::size_t i = 0; i < m; ++i)
                        y[permutation_[r.row0 + i] * rhs + p] +=
                            decode(block.left, i, j, m, block.widths[j]) * dot;
                }
        }
    }
    return y;
}

std::vector<double> HMatrix::reconstruct() const {
    require(finalized_, "matrix must be finalized before use");
    require(order_ <= std::numeric_limits<std::size_t>::max() / order_,
            "dense output size overflow");
    std::vector<double> out(order_ * order_, 0.0);
    for (const auto& block : blocks_) {
        const auto& r = block.rectangle;
        const std::size_t m = r.row1 - r.row0, n = r.col1 - r.col0;
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t k = 0; k < n; ++k) {
                double value = block.dense.empty() ? 0.0 : block.dense[i * n + k];
                for (std::size_t j = 0; j < block.widths.size(); ++j)
                    value += decode(block.left, i, j, m, block.widths[j]) *
                             decode(block.right, k, j, n, block.widths[j]);
                out[permutation_[r.row0 + i] * order_ + permutation_[r.col0 + k]] = value;
            }
    }
    return out;
}

std::vector<BlockInfo> HMatrix::block_info() const {
    std::vector<BlockInfo> info;
    info.reserve(blocks_.size());
    for (const auto& block : blocks_) {
        const auto& r = block.rectangle;
        std::uint64_t logical = 0;
        std::uint64_t bytes = 0;
        if (!block.dense.empty()) {
            logical = checked_mul(block.dense.size(), 64);
            bytes = checked_mul(block.dense.size(), sizeof(double));
        } else {
            std::uint64_t sum_widths = 0;
            for (int width : block.widths) sum_widths = checked_add(sum_widths, width);
            logical = checked_mul(checked_add(r.row1-r.row0, r.col1-r.col0), sum_widths);
            logical = checked_add(logical, checked_mul(block.widths.size(), 128 + label_bits(block.menu_size)));
            bytes = checked_add(checked_mul(checked_add(block.left.words.size(), block.right.words.size()), 4),
                                checked_mul(checked_add(block.left.scales.size(), block.right.scales.size()), 8));
            bytes = checked_add(bytes, checked_mul(checked_add(block.left.offsets.size(), block.right.offsets.size()), sizeof(std::size_t)));
            bytes = checked_add(bytes, checked_mul(block.widths.size(), sizeof(int)));
        }
        info.push_back({r, !block.dense.empty(), block.widths.size(),
                        block.truncation_error, block.quantization_bound, logical, bytes});
    }
    return info;
}

std::uint64_t HMatrix::logical_bits() const {
    std::uint64_t bits = 0;
    for (const auto& block : block_info()) bits = checked_add(bits, block.logical_bits);
    return bits;
}

std::uint64_t HMatrix::allocated_bytes() const {
    std::uint64_t bytes = 0;
    for (const auto& block : block_info()) bytes = checked_add(bytes, block.allocated_bytes);
    return bytes;
}

double HMatrix::reconstruction_bound() const {
    double bound = 0.0;
    for (const auto& block : blocks_)
        bound = std::max(bound, block.truncation_error + block.quantization_bound);
    return bound;
}

}  // namespace qhmat
