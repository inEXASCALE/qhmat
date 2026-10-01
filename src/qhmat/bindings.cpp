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

    py::class_<qhmat::Allocation>(m, "Allocation")
        .def_readonly("bits", &qhmat::Allocation::bits)
        .def_readonly("bound", &qhmat::Allocation::bound)
        .def_readonly("payload_bits", &qhmat::Allocation::payload_bits)
        .def_readonly("feasible", &qhmat::Allocation::feasible);

    py::class_<Rectangle>(m, "Rectangle")
        .def(py::init<std::size_t, std::size_t, std::size_t, std::size_t, bool>(),
             py::arg("row0"), py::arg("row1"), py::arg("col0"),
             py::arg("col1"), py::arg("admissible") = false)
        .def_readonly("row0", &Rectangle::row0)
        .def_readonly("row1", &Rectangle::row1)
        .def_readonly("col0", &Rectangle::col0)
        .def_readonly("col1", &Rectangle::col1)
        .def_readonly("admissible", &Rectangle::admissible);

    py::class_<qhmat::Partition>(m, "Partition")
        .def_readonly("permutation", &qhmat::Partition::permutation)
        .def_readonly("blocks", &qhmat::Partition::blocks);

    py::class_<qhmat::BlockInfo>(m, "BlockInfo")
        .def_readonly("rectangle", &qhmat::BlockInfo::rectangle)
        .def_readonly("dense", &qhmat::BlockInfo::dense)
        .def_readonly("rank", &qhmat::BlockInfo::rank)
        .def_readonly("truncation_error", &qhmat::BlockInfo::truncation_error)
        .def_readonly("quantization_bound", &qhmat::BlockInfo::quantization_bound)
        .def_readonly("logical_bits", &qhmat::BlockInfo::logical_bits)
        .def_readonly("allocated_bytes", &qhmat::BlockInfo::allocated_bytes);

    m.def("coefficient", &qhmat::coefficient, py::arg("bits"));
    m.def("allocate", [](const Array& singular_values, const std::vector<int>& widths,
                          double budget, std::size_t rows, std::size_t cols) {
        auto values = vector1(singular_values, "singular_values");
        py::gil_scoped_release release;
        return qhmat::allocate(values, widths, budget, rows, cols);
    }, py::arg("singular_values"), py::arg("bit_widths"), py::arg("budget"),
       py::arg("rows"), py::arg("cols"));

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
       py::arg("leaf_size"), py::arg("admissibility"));

    py::class_<HMatrix>(m, "NativeHMatrix")
        .def(py::init<std::size_t, std::vector<std::size_t>>(),
             py::arg("order"), py::arg("permutation"))
        .def("add_dense", [](HMatrix& matrix, Rectangle rect, const Array& data) {
            matrix.add_dense(rect, row_major2(data, rect.row1 - rect.row0,
                                              rect.col1 - rect.col0, "dense block"));
        })
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
           py::arg("truncation_error"))
        .def("finalize", &HMatrix::finalize)
        .def("save", &HMatrix::save, py::arg("path"))
        .def_static("load", &HMatrix::load, py::arg("path"))
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
        }, py::arg("x"), py::arg("use_cuda") = false)
        .def("reconstruct", [](const HMatrix& matrix) {
            std::vector<double> result;
            {
                py::gil_scoped_release release;
                result = matrix.reconstruct();
            }
            return matrix_array(result, matrix.order(), matrix.order());
        })
        .def_property_readonly("order", &HMatrix::order)
        .def_property_readonly("logical_bits", &HMatrix::logical_bits)
        .def_property_readonly("allocated_bytes", &HMatrix::allocated_bytes)
        .def_property_readonly("reconstruction_bound", &HMatrix::reconstruction_bound)
        .def_property_readonly("blocks", &HMatrix::block_info);

#ifdef QHMAT_HAS_CUDA
    m.attr("cuda_built") = true;
#else
    m.attr("cuda_built") = false;
#endif
}
