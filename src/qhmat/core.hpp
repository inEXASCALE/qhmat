#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace qhmat {

struct Allocation {
    std::vector<int> bits;
    double bound = 0.0;
    std::uint64_t payload_bits = 0;
    bool feasible = false;
};

double coefficient(int bits);
Allocation allocate(const std::vector<double>& singular_values,
                    const std::vector<int>& bit_widths, double budget,
                    std::size_t rows, std::size_t cols);

struct Rectangle {
    std::size_t row0, row1, col0, col1;
    bool admissible;
    Rectangle() : row0(0), row1(0), col0(0), col1(0), admissible(false) {}
    Rectangle(std::size_t r0, std::size_t r1, std::size_t c0,
              std::size_t c1, bool flag = false)
        : row0(r0), row1(r1), col0(c0), col1(c1), admissible(flag) {}
};

struct Partition {
    std::vector<std::size_t> permutation;
    std::vector<Rectangle> blocks;
};

Partition spatial_partition(const std::vector<double>& points,
                            const std::vector<double>& support_lower,
                            const std::vector<double>& support_upper,
                            std::size_t count, std::size_t dimension,
                            std::size_t leaf_size, double eta);

struct BlockInfo {
    Rectangle rectangle;
    bool dense;
    std::size_t rank;
    double truncation_error;
    double quantization_bound;
    std::uint64_t logical_bits;
    std::uint64_t allocated_bytes;
};

class HMatrix {
public:
    HMatrix(std::size_t order, std::vector<std::size_t> permutation);
    void add_dense(Rectangle rectangle, const std::vector<double>& values);
    void add_low_rank(Rectangle rectangle, const std::vector<double>& u,
                      const std::vector<double>& singular_values,
                      const std::vector<double>& v,
                      const std::vector<int>& widths, std::size_t menu_size,
                      double truncation_error);
    void finalize();
    std::vector<double> matmul(const std::vector<double>& x,
                               std::size_t rhs) const;
    std::vector<double> matmul_cuda(const std::vector<double>& x,
                                    std::size_t rhs) const;
    std::vector<double> reconstruct() const;
    std::vector<BlockInfo> block_info() const;
    std::size_t order() const noexcept { return order_; }
    std::uint64_t logical_bits() const;
    std::uint64_t allocated_bytes() const;
    std::uint64_t resident_bytes() const;
    double reconstruction_bound() const;
    void save(const std::string& path) const;
    static HMatrix load(const std::string& path);

private:
    struct PackedFactor {
        std::vector<std::uint32_t> words;
        std::vector<std::size_t> offsets;
        std::vector<double> scales;
    };
    struct Block {
        Rectangle rectangle;
        std::vector<double> dense;
        std::vector<int> widths;
        PackedFactor left, right;
        std::size_t menu_size = 0;
        double truncation_error = 0.0;
        double quantization_bound = 0.0;
    };

    std::size_t order_;
    std::vector<std::size_t> permutation_;
    std::vector<Block> blocks_;
    bool finalized_ = false;

    void check_rectangle(Rectangle rectangle, bool diagonal) const;
    static PackedFactor pack(const std::vector<double>& matrix,
                             std::size_t rows, const std::vector<int>& widths);
    static double decode(const PackedFactor& factor, std::size_t row,
                         std::size_t column, std::size_t rows, int bits);
};

}  // namespace qhmat
