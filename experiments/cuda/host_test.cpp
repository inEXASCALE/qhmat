#include "data.hpp"
#include <iostream>
#include <sstream>
int main(int argc, char** argv) {
    try {
        if (argc == 2) {
            auto input = qmat::read_data(argv[1]);
            auto packed = qmat::pack(input.p, input.m, input.bits);
            qmat::require(packed.exact.size() == input.p.size(), "input integration mismatch");
            std::cout << "binary input: " << input.m << 'x' << input.n << " rank=" << input.rank << '\n';
        }
        for (const auto& mode : {"uniform4", "uniform8", "uniform12", "uniform16", "mixed"}) {
            auto a = qmat::reorder(qmat::fixture(), mode);
            auto p = qmat::pack(a.p, a.m, a.bits), q = qmat::pack(a.q, a.n, a.bits);
            for (int j = 0; j < a.rank; ++j) for (int i = 0; i < a.m; ++i) {
                double decoded = qmat::unpack(p, i, j, a.bits[j]) * p.scales[j];
                qmat::require(decoded == p.exact[j * a.m + i], "packed roundtrip mismatch");
                qmat::require(std::abs(decoded - a.p[j * a.m + i]) <= p.scales[j] / 2 + 1e-14,
                              "quantization error bound violated");
            }
            auto b = qmat::bounds(a, p, q);
            for (int col = 0; col < a.n; ++col) for (int row = 0; row < a.m; ++row) {
                long double original = 0, quantized = 0;
                for (int j = 0; j < a.rank; ++j) {
                    original += (long double)a.p[j * a.m + row] * a.q[j * a.n + col];
                    quantized += (long double)p.exact[j * a.m + row] * q.exact[j * a.n + col];
                }
                qmat::require(std::abs(original - quantized) <= b.quant + b.reference_slack,
                              "quantization certificate violated");
            }
        }
        std::istringstream truncated("bad");
        uint32_t value;
        bool rejected = false;
        try { qmat::read_values(truncated, &value, 1); } catch (const std::runtime_error&) { rejected = true; }
        qmat::require(rejected, "truncated input accepted");
        std::cout << "packing, zero columns, quantization bounds, and truncated input: passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
