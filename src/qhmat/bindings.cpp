#include "qhmat/core.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <vector>

namespace py = pybind11;
using qhmat::HMatrix;
using qhmat::Rectangle;

namespace {
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

std::vector<double> vector1(const Array& array, const char* name) {
    if (array.ndim() != 1) throw py::value_error(std::string(name) + " must be one-dimensional");
    const auto* data = array.data();
    return {data, data + array.shape(0)};
}

std::vector<double> row_major2(const Array& array, std::size_t rows,
                               std::size_t cols, const char* name) {
    if (array.ndim() != 2 || std::size_t(array.shape(0)) != rows ||
        std::size_t(array.shape(1)) != cols)
        throw py::value_error(std::string(name) + " has the wrong shape");
    const auto* data = array.data();
    return {data, data + rows * cols};
}

std::vector<double> column_major2(const Array& array, std::size_t rows,
                                  std::size_t cols, const char* name) {
    if (array.ndim() != 2 || std::size_t(array.shape(0)) != rows ||
        std::size_t(array.shape(1)) != cols)
        throw py::value_error(std::string(name) + " has the wrong shape");
    auto input = array.unchecked<2>();
    std::vector<double> result(rows * cols);
    for (std::size_t j = 0; j < cols; ++j)
        for (std::size_t i = 0; i < rows; ++i)
            result[j * rows + i] = input(i, j);
    return result;
}

py::array_t<double> matrix_array(const std::vector<double>& values,
                                 std::size_t rows, std::size_t cols) {
    py::array_t<double> result({rows, cols});
    auto* data = result.mutable_data();
    std::copy(values.begin(), values.end(), data);
    return result;
}
}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ core for adaptive integer-quantized hierarchical matrices";

    py::class_<qhmat::Allocation>(m, "Allocation",
        "Result of discrete bit allocation for one factorized block.")
        .def_readonly("bits", &qhmat::Allocation::bits, "Width of each retained component.")
        .def_readonly("bound", &qhmat::Allocation::bound, "Singular-value error surrogate.")
        .def_readonly("payload_bits", &qhmat::Allocation::payload_bits,
                      "Packed factor payload in bits, excluding scales and labels.")
        .def_readonly("feasible", &qhmat::Allocation::feasible,
                      "Whether the surrogate fits within the supplied budget.");

    py::class_<Rectangle>(m, "Rectangle",
        "Block bounds [row0:row1, col0:col1] in clustered index coordinates.")
        .def(py::init<std::size_t, std::size_t, std::size_t, std::size_t, bool>(),
             py::arg("row0"), py::arg("row1"), py::arg("col0"),
             py::arg("col1"), py::arg("admissible") = false)
        .def_readonly("row0", &Rectangle::row0, "Inclusive row start.")
        .def_readonly("row1", &Rectangle::row1, "Exclusive row end.")
        .def_readonly("col0", &Rectangle::col0, "Inclusive column start.")
        .def_readonly("col1", &Rectangle::col1, "Exclusive column end.")
        .def_readonly("admissible", &Rectangle::admissible,
                      "Geometric admissibility flag, independent of storage format.");

    py::class_<qhmat::Partition>(m, "Partition",
        "Clustered permutation and disjoint block rectangles.")
        .def_readonly("permutation", &qhmat::Partition::permutation,
                      "Clustered position to source index mapping.")
        .def_readonly("blocks", &qhmat::Partition::blocks,
                      "Disjoint rectangles covering the full matrix.");

    py::class_<qhmat::BlockInfo>(m, "BlockInfo",
        "Read-only geometry, rank, error, and storage for a stored block.")
        .def_readonly("rectangle", &qhmat::BlockInfo::rectangle, "Block coordinates.")
        .def_readonly("dense", &qhmat::BlockInfo::dense, "Whether stored as dense FP64.")
        .def_readonly("rank", &qhmat::BlockInfo::rank, "Number of retained components.")
        .def_readonly("truncation_error", &qhmat::BlockInfo::truncation_error,
                      "Measured max-norm SVD truncation residual.")
        .def_readonly("quantization_bound", &qhmat::BlockInfo::quantization_bound,
                      "Singular-value quantization surrogate for this block.")
        .def_readonly("logical_bits", &qhmat::BlockInfo::logical_bits,
                      "Theoretical stored bits for this block.")
        .def_readonly("allocated_bytes", &qhmat::BlockInfo::allocated_bytes,
                      "Native vector payload bytes for this block.");

    m.def("coefficient", &qhmat::coefficient, py::arg("bits"),
          "Return c(b) = 1/q + 1/(4*q*q), where q = 2**(b-1) - 1.");
    m.def("allocate", [](const Array& singular_values, const std::vector<int>& widths,
                          double budget, std::size_t rows, std::size_t cols) {
        auto values = vector1(singular_values, "singular_values");
        py::gil_scoped_release release;
        return qhmat::allocate(values, widths, budget, rows, cols);
    }, py::arg("singular_values"), py::arg("bit_widths"), py::arg("budget"),
       py::arg("rows"), py::arg("cols"),
       "Choose per-component widths satisfying the singular-value surrogate budget.");

    m.def("spatial_partition", [](const Array& points, const Array& lower,
                                   const Array& upper, std::size_t leaf_size,
                                   double eta) {
        if (points.ndim() != 2) throw py::value_error("points must have shape (n, d)");
        const auto n = std::size_t(points.shape(0)), d = std::size_t(points.shape(1));
        auto p = row_major2(points, n, d, "points");
        auto lo = row_major2(lower, n, d, "support_lower");
        auto hi = row_major2(upper, n, d, "support_upper");
        py::gil_scoped_release release;
        return qhmat::spatial_partition(p, lo, hi, n, d, leaf_size, eta);
    }, py::arg("points"), py::arg("support_lower"), py::arg("support_upper"),
       py::arg("leaf_size"), py::arg("admissibility"),
       "Median-split points and form a disjoint block partition using box separation.");

    py::class_<HMatrix>(m, "NativeHMatrix",
        "Low-level C++ matrix builder; use HMatrix.from_dense for normal construction.")
        .def(py::init<std::size_t, std::vector<std::size_t>>(),
             py::arg("order"), py::arg("permutation"))
        .def("add_dense", [](HMatrix& matrix, Rectangle rect, const Array& data) {
            matrix.add_dense(rect, row_major2(data, rect.row1 - rect.row0,
                                              rect.col1 - rect.col0, "dense block"));
        }, "Add a dense FP64 block; its row and column clusters must be identical.")
        .def("add_low_rank", [](HMatrix& matrix, Rectangle rect, const Array& u,
                                const Array& sigma, const Array& v,
                                const std::vector<int>& bits, std::size_t menu_size,
                                double truncation_error) {
            const auto s = vector1(sigma, "singular_values");
            const auto m = rect.row1 - rect.row0, n = rect.col1 - rect.col0;
            const auto left = column_major2(u, m, s.size(), "left singular vectors");
            const auto right = column_major2(v, n, s.size(), "right singular vectors");
            py::gil_scoped_release release;
            matrix.add_low_rank(rect, left, s, right, bits, menu_size, truncation_error);
        }, py::arg("rectangle"), py::arg("u"), py::arg("singular_values"),
           py::arg("v"), py::arg("bits"), py::arg("menu_size"),
           py::arg("truncation_error"),
           "Add an off-diagonal SVD block; balanced quantized packing is internal.")
        .def("finalize", &HMatrix::finalize,
             "Validate that added blocks form a disjoint complete partition.")
        .def("save", &HMatrix::save, py::arg("path"),
             "Write a checksummed file; refuse to overwrite an existing file.")
        .def_static("load", &HMatrix::load, py::arg("path"),
                    "Load and validate a saved native matrix.")
        .def("matmul", [](const HMatrix& matrix, const Array& x,
                          bool use_cuda) {
            if (x.ndim() != 1 && x.ndim() != 2)
                throw py::value_error("right-hand side must be one- or two-dimensional");
            if (std::size_t(x.shape(0)) != matrix.order())
                throw py::value_error("right-hand side has the wrong row count");
            const auto rhs = x.ndim() == 1 ? std::size_t{1} : std::size_t(x.shape(1));
            std::vector<double> values;
            if (x.ndim() == 1) values = vector1(x, "right-hand side");
            else values = row_major2(x, matrix.order(), rhs, "right-hand side");
            std::vector<double> result;
            {
                py::gil_scoped_release release;
#ifdef QHMAT_HAS_CUDA
                result = use_cuda ? matrix.matmul_cuda(values, rhs) : matrix.matmul(values, rhs);
#else
                if (use_cuda) throw std::runtime_error("qhmat was built without CUDA support");
                result = matrix.matmul(values, rhs);
#endif
            }
            if (x.ndim() == 1) {
                py::array_t<double> out(matrix.order());
                std::copy(result.begin(), result.end(), out.mutable_data());
                return out;
            }
            return matrix_array(result, matrix.order(), rhs);
        }, py::arg("x"), py::arg("use_cuda") = false,
           "Apply to a vector or multiple right-hand sides.")
        .def("reconstruct", [](const HMatrix& matrix) {
            std::vector<double> result;
            {
                py::gil_scoped_release release;
                result = matrix.reconstruct();
            }
            return matrix_array(result, matrix.order(), matrix.order());
        }, "Reconstruct a dense FP64 matrix in source index order.")
        .def_property_readonly("order", &HMatrix::order, "Matrix order n.")
        .def_property_readonly("logical_bits", &HMatrix::logical_bits,
                               "Theoretical representation size in bits.")
        .def_property_readonly("allocated_bytes", &HMatrix::allocated_bytes,
                               "Native vector payload bytes.")
        .def_property_readonly("resident_bytes", &HMatrix::resident_bytes,
                               "Owned C++ object and vector capacity estimate in bytes.")
        .def_property_readonly("reconstruction_bound", &HMatrix::reconstruction_bound,
                               "Exact-arithmetic max-norm representation bound.")
        .def_property_readonly("blocks", &HMatrix::block_info,
                               "Metadata for the completed block partition.");

#ifdef QHMAT_HAS_CUDA
    m.attr("cuda_built") = true;
#else
    m.attr("cuda_built") = false;
#endif
}
