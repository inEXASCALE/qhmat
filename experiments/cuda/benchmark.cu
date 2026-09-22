#include "data.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>

void check(cudaError_t s) { qmat::require(s == cudaSuccess, cudaGetErrorString(s)); }
void check(cublasStatus_t s) { qmat::require(s == CUBLAS_STATUS_SUCCESS, "cuBLAS status " + std::to_string(s)); }
template<class T> struct Device {
    T* ptr = nullptr;
    size_t size;
    explicit Device(size_t n) : size(n) { check(cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T))); }
    explicit Device(const std::vector<T>& v) : Device(v.size()) {
        check(cudaMemcpy(ptr, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    ~Device() { cudaFree(ptr); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    std::vector<T> host() const {
        std::vector<T> v(size);
        check(cudaMemcpy(v.data(), ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
};
struct Blas {
    cublasHandle_t handle;
    Blas() { check(cublasCreate(&handle)); check(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH)); }
    ~Blas() { cublasDestroy(handle); }
};
struct Event {
    cudaEvent_t event;
    Event() { check(cudaEventCreate(&event)); }
    ~Event() { cudaEventDestroy(event); }
};
struct Timing { double median, minimum, maximum; };
Timing time_cuda(const std::function<void()>& fn, int warmup, int repeats) {
    for (int i = 0; i < warmup; ++i) fn();
    check(cudaDeviceSynchronize());
    Event start, stop;
    std::vector<float> times;
    for (int i = 0; i < repeats; ++i) {
        check(cudaEventRecord(start.event)); fn(); check(cudaEventRecord(stop.event));
        check(cudaEventSynchronize(stop.event));
        float ms; check(cudaEventElapsedTime(&ms, start.event, stop.event)); times.push_back(ms);
    }
    check(cudaGetLastError());
    std::sort(times.begin(), times.end());
    double median = (double(times[(repeats - 1) / 2]) + times[repeats / 2]) / 2;
    return {median, times.front(), times.back()};
}
__global__ void unpack_columns(const uint32_t* words, const uint64_t* offsets,
                               const double* scales, const uint32_t* bits,
                               int rows, size_t count, float* out) {
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    int j = index / rows, i = index % rows, b = bits[j];
    size_t bit = size_t(i) * b, word = offsets[j] + bit / 32;
    int shift = bit % 32;
    uint32_t raw = words[word] >> shift;
    if (shift + b > 32) raw |= words[word + 1] << (32 - shift);
    raw &= (1u << b) - 1;
    int value = (raw & (1u << (b - 1))) ? int(raw) - (1 << b) : int(raw);
    out[index] = float(double(value) * scales[j]);
}
__global__ void add_channel(float* out, const float* channel, size_t count) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) out[i] += channel[i];
}
__global__ void scale_int32(const int32_t* input, float* out, size_t count, double scale) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) out[i] = float(double(input[i]) * scale);
}
__global__ void to_half(const float* input, __half* out, size_t count) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) out[i] = __float2half_rn(input[i]);
}
int blocks(size_t count) { return int((count + 255) / 256); }
double max_error(const std::vector<float>& x, const std::vector<double>& y) {
    double err = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        qmat::require(std::isfinite(x[i]) && std::isfinite(y[i]), "nonfinite result");
        err = std::max(err, std::abs(double(x[i]) - y[i]));
    }
    return err;
}
std::vector<double> reference(cublasHandle_t h, const std::vector<double>& p,
                              const std::vector<double>& q, int m, int n, int rank) {
    Device<double> dp(p), dq(q), dc(size_t(m) * n);
    double alpha = 1, beta = 0;
    check(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_T, m, n, rank, &alpha,
                     dp.ptr, m, dq.ptr, n, &beta, dc.ptr, m));
    return dc.host();
}
void validate_small_cpu(const qmat::Data& a, const std::vector<double>& ref) {
    if (size_t(a.m) * a.n * a.rank > 1000000) return;
    for (int col = 0; col < a.n; ++col) for (int row = 0; row < a.m; ++row) {
        long double value = 0, weight = 0;
        for (int j = 0; j < a.rank; ++j) {
            long double term = (long double)a.p[j * a.m + row] * a.q[j * a.n + col];
            value += term; weight += std::abs(term);
        }
        qmat::require(std::abs(value - ref[col * a.m + row]) <=
                      16 * qmat::gamma(a.rank + 2, std::ldexp(1.0, -53)) * weight + 1e-30,
                      "GPU reference differs from independent CPU reference");
    }
}
void benchmark(qmat::Data input, const std::string& mode, int warmup, int repeats,
               std::ostream& csv, bool header) {
    auto t0 = std::chrono::steady_clock::now();
    auto a = qmat::reorder(std::move(input), mode);
    auto p = qmat::pack(a.p, a.m, a.bits), q = qmat::pack(a.q, a.n, a.bits);
    double setup_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    auto bound = qmat::bounds(a, p, q);
    cudaDeviceProp prop; check(cudaGetDeviceProperties(&prop, 0));
    int runtime, driver, blas_version;
    check(cudaRuntimeGetVersion(&runtime)); check(cudaDriverGetVersion(&driver));
    Blas blas; check(cublasGetVersion(blas.handle, &blas_version));
    const int m = a.m, n = a.n, rank = a.rank;
    const size_t count = size_t(m) * n;
    Device<float> dp(a.p), dq(a.q), output(count), channel(count), pd(a.p.size()), qd(a.q.size());
    Device<uint32_t> pw(p.words), qw(q.words), bits(a.bits);
    Device<uint64_t> po(p.offsets), qo(q.offsets);
    Device<double> ps(p.scales), qs(q.scales);
    auto original = reference(blas.handle, std::vector<double>(a.p.begin(), a.p.end()),
                              std::vector<double>(a.q.begin(), a.q.end()), m, n, rank);
    validate_small_cpu(a, original);
    auto quantized = reference(blas.handle, p.exact, q.exact, m, n, rank);
    double actual_quant = 0, original_norm = 0;
    for (size_t i = 0; i < count; ++i) {
        actual_quant = std::max(actual_quant, std::abs(original[i] - quantized[i]));
        original_norm = std::max(original_norm, std::abs(original[i]));
    }
    qmat::require(actual_quant <= bound.quant + bound.reference_slack, "quantization certificate failed");
    const float one = 1, zero = 0;
    auto gemm = [&](const float* left, const float* right, int r, float* dest, cublasComputeType_t compute) {
        check(cublasGemmEx(blas.handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, r, &one,
                          left, CUDA_R_32F, m, right, CUDA_R_32F, n, &zero, dest,
                          CUDA_R_32F, m, compute, CUBLAS_GEMM_DEFAULT));
    };
    auto baseline = [&] { gemm(dp.ptr, dq.ptr, rank, output.ptr, CUBLAS_COMPUTE_32F_PEDANTIC); };
    Timing baseline_time = time_cuda(baseline, warmup, repeats);
    auto dequant = [&] {
        unpack_columns<<<blocks(pd.size), 256>>>(pw.ptr, po.ptr, ps.ptr, bits.ptr, m, pd.size, pd.ptr);
        unpack_columns<<<blocks(qd.size), 256>>>(qw.ptr, qo.ptr, qs.ptr, bits.ptr, n, qd.size, qd.ptr);
    };
    auto grouped = [&] {
        for (int start = 0; start < rank;) {
            int end = start + 1;
            while (end < rank && a.bits[end] == a.bits[start]) ++end;
            gemm(pd.ptr + size_t(start) * m, qd.ptr + size_t(start) * n, end - start,
                 start == 0 ? output.ptr : channel.ptr, CUBLAS_COMPUTE_32F_PEDANTIC);
            if (start != 0) add_channel<<<blocks(count), 256>>>(output.ptr, channel.ptr, count);
            start = end;
        }
    };
    size_t packed_bytes = (p.words.size() + q.words.size()) * 4;
    size_t metadata_bytes = rank * (2 * sizeof(double) + 2 * sizeof(uint64_t) + sizeof(uint32_t));
    size_t workspace_bytes = (pd.size + qd.size + channel.size) * sizeof(float);
    if (header) csv << "device,cuda_runtime,cuda_driver,cublas_version,m,n,rank,mode,kernel,scope,median_ms,min_ms,max_ms,speedup_vs_fp32,effective_gflops,max_abs_error,relative_max_error,actual_quant_error,actual_arithmetic_error,q_cert,r_cert,reference_slack,validation,packed_bytes,metadata_bytes,workspace_bytes,cpu_prepare_ms,warmup,repeats\n";
    auto emit = [&](const std::string& name, const std::string& scope, Timing t,
                    const std::vector<double>& qref, double qcert, double rcert,
                    double qerror, const std::string& validation, size_t payload, size_t metadata, size_t workspace) {
        auto result = output.host();
        double error = max_error(result, original), arithmetic = max_error(result, qref);
        const bool decode_only = scope == "factor_unpack_only";
        if (validation == "certificate_pass") {
            qmat::require(arithmetic <= rcert + bound.reference_slack, name + ": arithmetic certificate failed");
            qmat::require(error <= qcert + rcert + bound.reference_slack, name + ": total certificate failed");
        }
        csv << std::setprecision(17) << '"' << prop.name << "\"," << runtime << ',' << driver << ',' << blas_version
            << ',' << m << ',' << n << ',' << rank << ',' << mode << ',' << name << ',' << scope << ','
            << t.median << ',' << t.minimum << ',' << t.maximum << ',';
        if (!decode_only) csv << baseline_time.median / t.median;
        csv << ',';
        if (!decode_only) csv << (2.0 * m * n * rank / (t.median * 1e6));
        csv << ',';
        if (decode_only) {
            for (int field = 0; field < 7; ++field) csv << ',';
            csv << "decode_pass,";
        } else {
            csv << error << ',' << (original_norm > 0 ? error / original_norm : error) << ','
                << qerror << ',' << arithmetic << ',' << qcert << ',' << rcert << ','
                << bound.reference_slack << ',' << validation << ',';
        }
        csv << payload << ',' << metadata << ',' << workspace << ',' << setup_ms << ','
            << warmup << ',' << repeats << '\n';
    };
    double fp_weight = 0;
    for (int j = 0; j < rank; ++j) fp_weight += p.max_abs[j] * q.max_abs[j];
    emit("fp32_cublas", "gemm_only", baseline_time, original, 0,
         qmat::gamma(rank, std::ldexp(1.0, -24)) * fp_weight, 0, "certificate_pass",
         (dp.size + dq.size) * 4, 0, 0);
    auto tf32 = [&] { gemm(dp.ptr, dq.ptr, rank, output.ptr, CUBLAS_COMPUTE_32F_FAST_TF32); };
    // Explicitly switch math mode only for the throughput references.
    check(cublasSetMathMode(blas.handle, CUBLAS_DEFAULT_MATH));
    auto tf_time = time_cuda(tf32, warmup, repeats);
    emit("tf32_cublas", "gemm_only", tf_time, original, 0, 0, 0, "error_measured",
         (dp.size + dq.size) * 4, 0, 0);
    {
        Device<__half> hp(dp.size), hq(dq.size);
        to_half<<<blocks(dp.size), 256>>>(dp.ptr, hp.ptr, dp.size);
        to_half<<<blocks(dq.size), 256>>>(dq.ptr, hq.ptr, dq.size);
        auto half_gemm = [&] {
            check(cublasGemmEx(blas.handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, rank, &one,
                              hp.ptr, CUDA_R_16F, m, hq.ptr, CUDA_R_16F, n, &zero,
                              output.ptr, CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        };
        auto half_time = time_cuda(half_gemm, warmup, repeats);
        emit("fp16_cublas", "gemm_only_preconverted", half_time, original, 0, 0, 0, "error_measured",
             (dp.size + dq.size) * 2, 0, 0);
    }
    check(cublasSetMathMode(blas.handle, CUBLAS_PEDANTIC_MATH));
    auto dequant_time = time_cuda(dequant, warmup, repeats);
    auto unpacked_p = pd.host(), unpacked_q = qd.host();
    for (size_t i = 0; i < p.exact.size(); ++i) qmat::require(unpacked_p[i] == float(p.exact[i]), "P CUDA unpack mismatch");
    for (size_t i = 0; i < q.exact.size(); ++i) qmat::require(unpacked_q[i] == float(q.exact[i]), "Q CUDA unpack mismatch");
    grouped(); check(cudaDeviceSynchronize());
    emit("unpack_only", "factor_unpack_only", dequant_time, quantized, bound.quant, bound.arithmetic,
         actual_quant, "certificate_pass", packed_bytes, metadata_bytes, (pd.size + qd.size) * 4);
    auto grouped_time = time_cuda(grouped, warmup, repeats);
    emit("columnwise_channels", "gemm_only_predequantized", grouped_time, quantized, bound.quant, bound.arithmetic,
         actual_quant, "certificate_pass", (pd.size + qd.size) * 4, rank * 4, channel.size * 4);
    auto packed_gemm = [&] { dequant(); grouped(); };
    auto packed_time = time_cuda(packed_gemm, warmup, repeats);
    emit("packed_columnwise_channels", "unpack_gemm_accumulate", packed_time, quantized,
         bound.quant, bound.arithmetic, actual_quant, "certificate_pass", packed_bytes, metadata_bytes, workspace_bytes);

    // cuBLAS INT8 requires aligned leading dimensions; bound INT32 accumulation before dispatch.
    if (mode == "uniform8" && m % 4 == 0 && n % 4 == 0 && int64_t(rank) * 127 * 127 <= INT32_MAX) {
        check(cublasSetMathMode(blas.handle, CUBLAS_DEFAULT_MATH));
        double pmax = *std::max_element(p.max_abs.begin(), p.max_abs.end());
        double qmax = *std::max_element(q.max_abs.begin(), q.max_abs.end());
        double sp = pmax / 127, sq = qmax / 127;
        std::vector<int8_t> pi(a.p.size()), qi(a.q.size());
        std::vector<double> pr(pi.size()), qr(qi.size());
        for (size_t i = 0; i < pi.size(); ++i) { pi[i] = sp == 0 ? 0 : std::clamp(int(std::nearbyint(a.p[i] / sp)), -127, 127); pr[i] = pi[i] * sp; }
        for (size_t i = 0; i < qi.size(); ++i) { qi[i] = sq == 0 ? 0 : std::clamp(int(std::nearbyint(a.q[i] / sq)), -127, 127); qr[i] = qi[i] * sq; }
        auto shared_ref = reference(blas.handle, pr, qr, m, n, rank);
        Device<int8_t> ip(pi), iq(qi); Device<int32_t> acc(count);
        int alpha = 1, beta = 0;
        auto int8 = [&] {
            check(cublasGemmEx(blas.handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, rank, &alpha,
                              ip.ptr, CUDA_R_8I, m, iq.ptr, CUDA_R_8I, n, &beta, acc.ptr,
                              CUDA_R_32I, m, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
            scale_int32<<<blocks(count), 256>>>(acc.ptr, output.ptr, count, sp * sq);
        };
        auto int_time = time_cuda(int8, warmup, repeats);
        double shared_q = 0, shared_norm = 0;
        for (size_t i = 0; i < count; ++i) {
            shared_q = std::max(shared_q, std::abs(original[i] - shared_ref[i]));
            shared_norm = std::max(shared_norm, std::abs(shared_ref[i]));
        }
        // Integer dot products are exact; only the scale epilogue rounds.
        double integer_r = qmat::gamma(3, std::ldexp(1.0, -24)) * shared_norm;
        double integer_q = rank * pmax * qmax * (1.0 / 127 + 1.0 / (4 * 127 * 127));
        emit("shared_scale_int8", "integer_gemm_scale", int_time, shared_ref, integer_q, integer_r,
             shared_q, "certificate_pass", pi.size() + qi.size(), 16, count * 4);
    } else if (mode == "uniform8") {
        std::cerr << "INT8 proxy skipped: leading-dimension alignment or INT32 overflow constraint\n";
    }
}
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") {
            bool header = true;
            for (const auto& mode : {"uniform4", "uniform8", "uniform12", "uniform16", "mixed"}) {
                benchmark(qmat::fixture(), mode, 1, 2, std::cout, header); header = false;
            }
            benchmark(qmat::fixture(32, 48, 16), "uniform8", 1, 2, std::cout, false);
            return 0;
        }
        qmat::require(argc == 6, "usage: qmat_cuda DATA MODE WARMUP REPEATS OUTPUT.csv | --self-test");
        int warmup = std::stoi(argv[3]), repeats = std::stoi(argv[4]);
        qmat::require(warmup >= 0 && repeats > 0, "invalid timing repetition count");
        std::ofstream csv(argv[5]); qmat::require(bool(csv), "cannot open output CSV");
        benchmark(qmat::read_data(argv[1]), argv[2], warmup, repeats, csv, true);
        qmat::require(bool(csv), "failed writing CSV");
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
