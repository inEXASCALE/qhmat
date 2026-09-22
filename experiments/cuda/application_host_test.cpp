#include "application.hpp"
#include <iostream>
int main(int argc, char** argv) {
    try {
        qmat::require(argc == 2, "usage: qmat_apply_host INPUT.bin");
        auto a = qmat::read_application(argv[1]);
        for (bool quantized : {false, true}) {
            auto y = qmat::apply_host(a, a.rhs, quantized);
            auto& ref = a.references[quantized ? 3 : 2];
            double norm = 0, error = 0;
            for (size_t i=0; i<y.size(); ++i) {
                norm = std::max(norm, std::abs(ref[i])); error = std::max(error, std::abs(y[i]-ref[i]));
            }
            qmat::require(error <= 1e-11*std::max(norm, 1e-300), "host application disagrees with saved FP64 reference");
        }
        std::cout << "application input, packed decoding, permutation and references pass\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
