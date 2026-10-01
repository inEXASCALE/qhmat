#include "qhmat/core.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace qhmat {
namespace {

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

template <class T> class DeviceBuffer {
public:
    explicit DeviceBuffer(const std::vector<T>& host) {
        if (host.empty()) return;
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&data_), host.size() * sizeof(T)),
                   "cudaMalloc");
        cuda_check(cudaMemcpy(data_, host.data(), host.size() * sizeof(T),
                              cudaMemcpyHostToDevice), "cudaMemcpy host-to-device");
    }
    explicit DeviceBuffer(std::size_t size) {
        if (size) cuda_check(cudaMalloc(reinterpret_cast<void**>(&data_), size * sizeof(T)),
                             "cudaMalloc");
    }
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return data_; }
private:
    T* data_ = nullptr;
};

struct DeviceBlock {
    std::size_t row0, row1, col0, col1;
    std::size_t rank, factor_offset, dense_offset;
    bool is_dense;
};

__device__ double decode(const std::uint32_t* words, const std::size_t* offsets,
                         const double* scales, const int* widths,
                         std::size_t component, std::size_t row) {
    const int bits = widths[component];
    const std::size_t position = row * std::size_t(bits);
    const std::size_t word = offsets[component] + position / 32;
    const unsigned shift = unsigned(position % 32);
    std::uint32_t raw = words[word] >> shift;
    if (shift + bits > 32) raw |= words[word + 1] << (32 - shift);
    const std::uint32_t mask = bits == 32 ? UINT32_MAX : (std::uint32_t{1} << bits) - 1;
    raw &= mask;
    const std::int64_t signed_value = (raw & (std::uint32_t{1} << (bits - 1)))
        ? std::int64_t(raw) - (std::int64_t{1} << bits) : std::int64_t(raw);
    return double(signed_value) * scales[component];
}

__global__ void apply_kernel(const DeviceBlock* blocks, std::size_t block_count,
                             const std::size_t* permutation,
                             const std::size_t* inverse_permutation,
                             const double* dense, const std::uint32_t* left_words,
                             const std::uint32_t* right_words,
                             const std::size_t* left_offsets,
                             const std::size_t* right_offsets,
                             const double* left_scales, const double* right_scales,
                             const int* widths, const double* x, double* y,
                             std::size_t order, std::size_t rhs) {
    const std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= order * rhs) return;
    const std::size_t original_row = index / rhs, column = index % rhs;
    const std::size_t clustered_row = inverse_permutation[original_row];
    double sum = 0.0;
    for (std::size_t b = 0; b < block_count; ++b) {
        const auto block = blocks[b];
        if (clustered_row < block.row0 || clustered_row >= block.row1) continue;
        const std::size_t local_row = clustered_row - block.row0;
        const std::size_t n = block.col1 - block.col0;
        if (block.is_dense) {
            for (std::size_t k = 0; k < n; ++k)
                sum += dense[block.dense_offset + local_row * n + k] *
                       x[permutation[block.col0 + k] * rhs + column];
        } else {
            for (std::size_t j = 0; j < block.rank; ++j) {
                const std::size_t component = block.factor_offset + j;
                double dot = 0.0;
                for (std::size_t k = 0; k < n; ++k)
                    dot += decode(right_words, right_offsets, right_scales, widths,
                                  component, k) *
                           x[permutation[block.col0 + k] * rhs + column];
                sum += decode(left_words, left_offsets, left_scales, widths,
                              component, local_row) * dot;
            }
        }
    }
    y[index] = sum;
}

}  // namespace

std::vector<double> HMatrix::matmul_cuda(const std::vector<double>& x,
                                         std::size_t rhs) const {
    if (!finalized_) throw std::invalid_argument("matrix must be finalized before use");
    if (!rhs || rhs > std::numeric_limits<std::size_t>::max() / order_ ||
        x.size() != order_ * rhs) throw std::invalid_argument("right-hand-side shape mismatch");
    if (order_ * rhs > std::size_t(std::numeric_limits<int>::max()) * 256)
        throw std::overflow_error("CUDA launch grid too large");
    for (double value : x)
        if (!std::isfinite(value)) throw std::invalid_argument("nonfinite right-hand side");
    int devices = 0;
    cuda_check(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    if (!devices) throw std::runtime_error("no CUDA device is available");

    std::vector<DeviceBlock> descriptors;
    std::vector<double> dense, left_scales, right_scales;
    std::vector<std::uint32_t> left_words, right_words;
    std::vector<std::size_t> left_offsets, right_offsets;
    std::vector<int> widths;
    descriptors.reserve(blocks_.size());
    for (const auto& block : blocks_) {
        const auto& r = block.rectangle;
        descriptors.push_back({r.row0, r.row1, r.col0, r.col1,
                               block.widths.size(), widths.size(), dense.size(),
                               !block.dense.empty()});
        dense.insert(dense.end(), block.dense.begin(), block.dense.end());
        const auto left_base = left_words.size(), right_base = right_words.size();
        for (std::size_t offset : block.left.offsets) left_offsets.push_back(left_base + offset);
        for (std::size_t offset : block.right.offsets) right_offsets.push_back(right_base + offset);
        left_words.insert(left_words.end(), block.left.words.begin(), block.left.words.end());
        right_words.insert(right_words.end(), block.right.words.begin(), block.right.words.end());
        left_scales.insert(left_scales.end(), block.left.scales.begin(), block.left.scales.end());
        right_scales.insert(right_scales.end(), block.right.scales.begin(), block.right.scales.end());
        widths.insert(widths.end(), block.widths.begin(), block.widths.end());
    }
    std::vector<std::size_t> inverse(order_);
    for (std::size_t i = 0; i < order_; ++i) inverse[permutation_[i]] = i;

    DeviceBuffer<DeviceBlock> d_blocks(descriptors);
    DeviceBuffer<std::size_t> d_permutation(permutation_), d_inverse(inverse);
    DeviceBuffer<double> d_dense(dense), d_left_scales(left_scales),
                         d_right_scales(right_scales), d_x(x);
    DeviceBuffer<std::uint32_t> d_left_words(left_words), d_right_words(right_words);
    DeviceBuffer<std::size_t> d_left_offsets(left_offsets), d_right_offsets(right_offsets);
    DeviceBuffer<int> d_widths(widths);
    DeviceBuffer<double> d_y(order_ * rhs);
    const std::size_t threads = 256;
    const auto grid = static_cast<unsigned>((order_ * rhs + threads - 1) / threads);
    apply_kernel<<<grid, static_cast<unsigned>(threads)>>>(
        d_blocks.get(), descriptors.size(), d_permutation.get(), d_inverse.get(),
        d_dense.get(), d_left_words.get(), d_right_words.get(),
        d_left_offsets.get(), d_right_offsets.get(), d_left_scales.get(),
        d_right_scales.get(), d_widths.get(), d_x.get(), d_y.get(), order_, rhs);
    cuda_check(cudaGetLastError(), "CUDA kernel launch");
    cuda_check(cudaDeviceSynchronize(), "CUDA kernel execution");
    std::vector<double> result(order_ * rhs);
    cuda_check(cudaMemcpy(result.data(), d_y.get(), result.size() * sizeof(double),
                          cudaMemcpyDeviceToHost), "cudaMemcpy device-to-host");
    return result;
}

}  // namespace qhmat
