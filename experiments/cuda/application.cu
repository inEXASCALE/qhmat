// Complete Y=A*X application: fixed partition/ranks, no dense reconstruction.
#include "application.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>

void check(cudaError_t s) { qmat::require(s == cudaSuccess, cudaGetErrorString(s)); }
void check(cublasStatus_t s) { qmat::require(s == CUBLAS_STATUS_SUCCESS, "cuBLAS status " + std::to_string(s)); }
template<class T> struct Buffer {
    T* p=nullptr; size_t count;
    explicit Buffer(size_t n) : count(n) { check(cudaMalloc(reinterpret_cast<void**>(&p), std::max(size_t(1),n)*sizeof(T))); }
    explicit Buffer(const std::vector<T>& v) : Buffer(v.size()) {
        if (count) check(cudaMemcpy(p,v.data(),count*sizeof(T),cudaMemcpyHostToDevice));
    }
    ~Buffer() { cudaFree(p); }
    Buffer(const Buffer&)=delete; Buffer& operator=(const Buffer&)=delete;
};
struct Blas {
    cublasHandle_t h;
    Blas() { check(cublasCreate(&h)); check(cublasSetMathMode(h,CUBLAS_PEDANTIC_MATH)); }
    ~Blas() { cublasDestroy(h); }
};
struct Events {
    cudaEvent_t start,stop;
    Events() { check(cudaEventCreate(&start)); check(cudaEventCreate(&stop)); }
    ~Events() { cudaEventDestroy(start); cudaEventDestroy(stop); }
};
double median(std::vector<double> x) {
    std::sort(x.begin(),x.end()); return (x[(x.size()-1)/2]+x[x.size()/2])/2;
}
std::array<double,3> timed(const std::function<void()>& fn,int repeats) {
    Events e; std::vector<double> values;
    for (int i=0;i<repeats;++i) {
        check(cudaEventRecord(e.start)); fn(); check(cudaEventRecord(e.stop));
        check(cudaEventSynchronize(e.stop)); float ms;
        check(cudaEventElapsedTime(&ms,e.start,e.stop)); values.push_back(ms);
    }
    check(cudaGetLastError());
    return {median(values),*std::min_element(values.begin(),values.end()),*std::max_element(values.begin(),values.end())};
}
struct Job {
    int m,n,r,col0;
    const float *u,*v,*x;
    float *decoded_u,*decoded_v,*z,*y;
    const uint32_t *bits,*uw,*vw;
    const uint64_t *uo,*vo;
    const double *us,*vs;
};
__device__ float read_weight(const Job& j,bool left,int row,int col) {
    int b=j.bits[col];
    const uint32_t* words=left?j.uw:j.vw;
    const uint64_t* offsets=left?j.uo:j.vo;
    const double* scales=left?j.us:j.vs;
    size_t bit=size_t(row)*b, word=offsets[col]+bit/32;
    int shift=bit%32; uint32_t raw=words[word]>>shift;
    if (shift+b>32) raw |= words[word+1]<<(32-shift);
    raw &= b == 32 ? UINT32_MAX : (1u<<b)-1;
    int value=(raw&(1u<<(b-1)))?int(int64_t(raw)-(int64_t(1)<<b)):int(raw);
    return float(double(value)*scales[col]);
}
__global__ void gather_x(const float* x,float* xp,const uint32_t* perm,int n,int rhs) {
    size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (k<size_t(n)*rhs) xp[k]=x[perm[k%n]+(k/n)*n];
}
__global__ void unpack_jobs(const Job* jobs,int count) {
    int id=blockIdx.x;
    if (id>=count) return;
    Job j=jobs[id];
    for (int k=threadIdx.x;k<j.m*j.r;k+=blockDim.x) j.decoded_u[k]=read_weight(j,true,k%j.m,k/j.m);
    for (int k=threadIdx.x;k<j.n*j.r;k+=blockDim.x) j.decoded_v[k]=read_weight(j,false,k%j.n,k/j.n);
}
// Same tiled SIMT kernel for floating and packed factors; no integer Tensor Core claim.
// Packed values are decoded in shared-memory tiles, never into full global factors.
template<bool Packed, bool First> __global__ void factor_gemm(const Job* jobs,int rhs,int full_n) {
    Job j=jobs[blockIdx.z];
    const int rows=First?j.r:j.m, inner=First?j.n:j.r;
    int i=blockIdx.x*16+threadIdx.x, col=blockIdx.y*16+threadIdx.y;
    __shared__ float a[16][16], b[16][16];
    float sum=0;
    for (int start=0;start<inner;start+=16) {
        int k=start+threadIdx.y;
        float value=0;
        if (i<rows && k<inner) {
            if (Packed) value=First?read_weight(j,false,k,i):read_weight(j,true,i,k);
            else value=First?j.v[k+size_t(i)*j.n]:j.u[i+size_t(k)*j.m];
        }
        a[threadIdx.y][threadIdx.x]=value;
        k=start+threadIdx.x;
        b[threadIdx.y][threadIdx.x]=(col<rhs && k<inner)?
            (First?j.x[j.col0+k+size_t(col)*full_n]:j.z[k+size_t(col)*j.r]):0;
        __syncthreads();
        for (int t=0;t<16;++t) sum=fmaf(a[t][threadIdx.x],b[threadIdx.y][t],sum);
        __syncthreads();
    }
    if (i<rows && col<rhs) (First?j.z:j.y)[i+size_t(col)*rows]=sum;
}
struct RowTerm { const float* y; int offset,ld; };
__global__ void reduce_rows(const RowTerm* terms,const uint64_t* starts,const uint32_t* perm,
                            float* output,int n,int rhs) {
    size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (index>=size_t(n)*rhs) return;
    int row=index%n, col=index/n; float value=0;
    // Fixed leaf order and one writer per output entry: no atomics or output races.
    for (uint64_t k=starts[row];k<starts[row+1];++k) {
        RowTerm t=terms[k]; value+=t.y[t.offset+size_t(col)*t.ld];
    }
    output[perm[row]+size_t(col)*n]=value;
}
struct LeafDevice {
    Buffer<float> u,v,dense,du,dv,z,y;
    Buffer<uint32_t> bits,uw,vw;
    Buffer<uint64_t> uo,vo;
    Buffer<double> us,vs;
    LeafDevice(const qmat::AppLeaf& l,int rhs) : u(l.p),v(l.q),dense(l.dense),
        du(l.p.size()),dv(l.q.size()),z(size_t(l.rank)*rhs),y(size_t(l.m())*rhs),
        bits(l.bits),uw(l.pp.words),vw(l.pq.words),uo(l.pp.offsets),vo(l.pq.offsets),us(l.pp.scales),vs(l.pq.scales) {}
};
struct Group {
    int m,n,r,count;
    Buffer<Job> jobs;
    Buffer<const float*> u,v,du,dv,x,dense,z_read;
    Buffer<float*> z,y;
    Group(int m_,int n_,int r_,const std::vector<Job>& js,const std::vector<const float*>& ds) :
        m(m_),n(n_),r(r_),count(int(js.size())),jobs(js),u(js.size()),v(js.size()),du(js.size()),dv(js.size()),
        x(js.size()),dense(ds),z_read(js.size()),z(js.size()),y(js.size()) {
        std::vector<const float*> hu,hv,hdu,hdv,hx,hzr;
        std::vector<float*> hz,hy;
        for (auto& j:js) {
            hu.push_back(j.u); hv.push_back(j.v); hdu.push_back(j.decoded_u); hdv.push_back(j.decoded_v);
            hx.push_back(j.x+j.col0); hzr.push_back(j.z); hz.push_back(j.z); hy.push_back(j.y);
        }
        auto upload=[](auto& buffer,const auto& host) { check(cudaMemcpy(buffer.p,host.data(),host.size()*sizeof(host[0]),cudaMemcpyHostToDevice)); };
        upload(u,hu);upload(v,hv);upload(du,hdu);upload(dv,hdv);upload(x,hx);upload(z_read,hzr);upload(z,hz);upload(y,hy);
    }
};
struct Plan {
    const qmat::Application& a; int rhs;
    Blas blas;
    Buffer<float> dense,x,xp,output;
    Buffer<uint32_t> perm;
    std::vector<std::unique_ptr<LeafDevice>> leaves;
    std::vector<std::unique_ptr<Group>> groups;
    std::unique_ptr<Buffer<RowTerm>> terms;
    std::unique_ptr<Buffer<uint64_t>> starts;
    size_t fp_bytes=0,packed_bytes=0,factor_workspace=0,workspace=0,plan_bytes=0;
    Plan(const qmat::Application& input,int p) : a(input),rhs(p),dense(a.dense),
        x(std::vector<float>(a.x.begin(),a.x.begin()+size_t(a.n)*p)),xp(size_t(a.n)*p),output(size_t(a.n)*p),perm(a.permutation) {
        std::map<std::array<int,3>,std::vector<int>> buckets;
        for (size_t i=0;i<a.leaves.size();++i) {
            auto& l=a.leaves[i];
            leaves.emplace_back(new LeafDevice(l,p)); buckets[{l.m(),l.n(),l.rank}].push_back(int(i));
            fp_bytes+=(l.p.size()+l.q.size()+l.dense.size())*4;
            packed_bytes+=l.dense.size()*4+(l.pp.words.size()+l.pq.words.size()+l.bits.size())*4+
                          (l.pp.offsets.size()+l.pq.offsets.size()+l.pp.scales.size()+l.pq.scales.size())*8;
            factor_workspace+=(l.p.size()+l.q.size())*4;
            workspace+=size_t(l.rank+l.m())*p*4;
        }
        for (auto& bucket:buckets) {
            std::vector<Job> js; std::vector<const float*> ds;
            for (int id:bucket.second) {
                auto& l=a.leaves[id]; auto& d=*leaves[id];
                js.push_back(Job{l.m(),l.n(),l.rank,l.col0,d.u.p,d.v.p,xp.p,d.du.p,d.dv.p,d.z.p,d.y.p,
                                 d.bits.p,d.uw.p,d.vw.p,d.uo.p,d.vo.p,d.us.p,d.vs.p});
                ds.push_back(d.dense.p);
            }
            qmat::require(js.size()<=65535,"too many leaves in one CUDA group");
            groups.emplace_back(new Group(bucket.first[0],bucket.first[1],bucket.first[2],js,ds));
            plan_bytes+=js.size()*(sizeof(Job)+9*sizeof(void*));
        }
        std::vector<RowTerm> ht; std::vector<uint64_t> hs{0};
        for (int row=0;row<a.n;++row) {
            for (size_t k=0;k<a.leaves.size();++k) {
                auto& l=a.leaves[k];
                if (l.row0<=row && row<l.row1) ht.push_back(RowTerm{leaves[k]->y.p,row-l.row0,l.m()});
            }
            hs.push_back(ht.size());
        }
        terms.reset(new Buffer<RowTerm>(ht)); starts.reset(new Buffer<uint64_t>(hs));
        plan_bytes+=ht.size()*sizeof(RowTerm)+hs.size()*8+a.n*4;
        workspace+=size_t(a.n)*p*8; // Permuted RHS and final output; input X is separate.
        for (auto& g:groups) if (g->r) unpack_jobs<<<g->count,256>>>(g->jobs.p,g->count);
        check(cudaDeviceSynchronize());
    }
    void apply(int method) {
        float one=1,zero=0;
        if (method==0) {
            check(cublasSgemm(blas.h,CUBLAS_OP_N,CUBLAS_OP_N,a.n,rhs,a.n,&one,dense.p,a.n,x.p,a.n,&zero,output.p,a.n));
            return;
        }
        gather_x<<<(size_t(a.n)*rhs+255)/256,256>>>(x.p,xp.p,perm.p,a.n,rhs);
        for (auto& ptr:groups) {
            auto& g=*ptr;
            if (!g.r) {
                check(cublasSgemmBatched(blas.h,CUBLAS_OP_N,CUBLAS_OP_N,g.m,rhs,g.n,&one,g.dense.p,g.m,
                    g.x.p,a.n,&zero,g.y.p,g.m,g.count));
                continue;
            }
            if (method==2) unpack_jobs<<<g.count,256>>>(g.jobs.p,g.count);
            if (method<=3) {
                auto up=method==1?g.u.p:g.du.p; auto vp=method==1?g.v.p:g.dv.p;
                check(cublasSgemmBatched(blas.h,CUBLAS_OP_T,CUBLAS_OP_N,g.r,rhs,g.n,&one,vp,g.n,g.x.p,a.n,
                    &zero,g.z.p,g.r,g.count));
                check(cublasSgemmBatched(blas.h,CUBLAS_OP_N,CUBLAS_OP_N,g.m,rhs,g.r,&one,up,g.m,g.z_read.p,g.r,
                    &zero,g.y.p,g.m,g.count));
            } else {
                dim3 threads(16,16), first((g.r+15)/16,(rhs+15)/16,g.count), second((g.m+15)/16,(rhs+15)/16,g.count);
                if (method==4) {
                    factor_gemm<false,true><<<first,threads>>>(g.jobs.p,rhs,a.n);
                    factor_gemm<false,false><<<second,threads>>>(g.jobs.p,rhs,a.n);
                } else {
                    factor_gemm<true,true><<<first,threads>>>(g.jobs.p,rhs,a.n);
                    factor_gemm<true,false><<<second,threads>>>(g.jobs.p,rhs,a.n);
                }
            }
        }
        reduce_rows<<<(size_t(a.n)*rhs+255)/256,256>>>(terms->p,starts->p,perm.p,output.p,a.n,rhs);
    }
};
int main(int argc,char** argv) {
    try {
        qmat::require(argc==8,"usage: qmat_apply INPUT RHS WARMUP REPEATS TRIALS SEED OUTPUT.csv");
        auto a=qmat::read_application(argv[1]);
        int p=std::stoi(argv[2]),warmup=std::stoi(argv[3]),repeats=std::stoi(argv[4]),trials=std::stoi(argv[5]);
        qmat::require(p>0 && p<=a.rhs && warmup>=0 && repeats>0 && trials>0,"invalid application arguments");
        auto begin=std::chrono::steady_clock::now(); Plan plan(a,p);
        double plan_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        cudaDeviceProp prop; check(cudaGetDeviceProperties(&prop,0));
        int runtime,driver,version; check(cudaRuntimeGetVersion(&runtime));check(cudaDriverGetVersion(&driver));
        check(cublasGetVersion(plan.blas.h,&version));
        const std::array<std::string,6> names={"dense_fp32","h_fp32_batched","adaptive_unpacked_batched",
            "adaptive_predecoded_batched","h_fp32_fused","adaptive_packed_fused"};
        std::ofstream out(argv[7]); qmat::require(bool(out),"cannot open output"); out<<std::setprecision(17);
        out<<"method,trial,order_index,n,rhs,device,cuda_runtime,cuda_driver,cublas_version,median_ms,min_ms,max_ms,speedup_vs_dense,speedup_vs_h_fp32,max_abs_error,relative_max_error,arithmetic_error,arithmetic_allowance,representation_error,representation_bound,validation,factor_bytes,plan_bytes,workspace_bytes,plan_prepare_ms,groups,leaves,warmup,repeats\n";
        std::mt19937 rng{unsigned(std::stoul(argv[6]))};
        for (int trial=0;trial<trials;++trial) {
            std::array<int,6> order={0,1,2,3,4,5}; std::shuffle(order.begin(),order.end(),rng);
            std::array<std::array<double,3>,6> times;
            std::array<std::array<double,6>,6> errors;
            for (int method:order) {
                for (int w=0;w<warmup;++w) plan.apply(method);
                check(cudaDeviceSynchronize());
                times[method]=timed([&]{plan.apply(method);},repeats);
                std::vector<float> y(size_t(a.n)*p); check(cudaMemcpy(y.data(),plan.output.p,y.size()*4,cudaMemcpyDeviceToHost));
                int reference=method==0?1:((method==1 || method==4)?2:3);
                double err=0,norm=0,arithmetic=0,allowance=0,representation=0,model=0;
                for (size_t i=0;i<y.size();++i) {
                    qmat::require(std::isfinite(y[i]),"nonfinite GPU output");
                    double actual=std::abs(double(y[i])-a.references[reference][i]);
                    double limit=a.allowances[reference-1][i/a.n];
                    qmat::require(actual<=limit,"application arithmetic validation failed for "+names[method]);
                    arithmetic=std::max(arithmetic,actual); allowance=std::max(allowance,limit);
                    err=std::max(err,std::abs(double(y[i])-a.references[0][i])); norm=std::max(norm,std::abs(a.references[0][i]));
                    representation=std::max(representation,std::abs(a.references[reference][i]-a.references[0][i]));
                    model=std::max(model,a.model_bounds[reference-1][i/a.n]);
                }
                errors[method]={err,norm==0?err:err/norm,arithmetic,allowance,representation,model};
            }
            for (int position=0;position<6;++position) {
                int m=order[position]; auto t=times[m]; auto e=errors[m];
                size_t factor=m==0?size_t(a.n)*a.n*4:((m==1 || m==3 || m==4)?plan.fp_bytes:plan.packed_bytes);
                size_t workspace=m==0?size_t(a.n)*p*4:plan.workspace+(m==2?plan.factor_workspace:0);
                out<<names[m]<<','<<trial<<','<<position<<','<<a.n<<','<<p<<",\""<<prop.name<<"\","<<runtime<<','<<driver<<','<<version<<','
                   <<t[0]<<','<<t[1]<<','<<t[2]<<','<<times[0][0]/t[0]<<','<<times[1][0]/t[0];
                for (double x:e) out<<','<<x;
                out<<",arithmetic_pass,"<<factor<<','<<(m==0?0:plan.plan_bytes)<<','<<workspace<<','<<plan_ms<<','<<plan.groups.size()<<','
                   <<a.leaves.size()<<','<<warmup<<','<<repeats<<'\n';
            }
            out.flush(); qmat::require(bool(out),"application output write failed");
        }
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
