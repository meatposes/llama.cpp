// oneDNN F16 matmul benchmark to check the ~8x claim vs the fused int8 XMX kernel (8 ms/91 GFLOP).
// dst[N,M] f32 = W[N,K] f16 * X[K,M] f16.  Same dims/FLOP as the fused kernel: K=5120 N=17408 M=512.
#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <cstdio>
#include <vector>
#include <chrono>
using namespace dnnl;

int main(int argc,char**argv){
    int K=argc>1?atoi(argv[1]):5120, N=argc>2?atoi(argv[2]):17408, M=argc>3?atoi(argv[3]):512;
    sycl::queue q{sycl::gpu_selector_v};
    std::printf("device: %s  K=%d N=%d M=%d\n", q.get_device().get_info<sycl::info::device::name>().c_str(),K,N,M);

    auto eng = sycl_interop::make_engine(q.get_device(), q.get_context());
    auto strm = sycl_interop::make_stream(eng, q);

    // C[N,M] = A[N,K] * B[K,M], f16 inputs, f32 output. (orientation matches ggml dst[N,M]=W.X)
    memory::dims Ad={N,K}, Bd={K,M}, Cd={N,M};
    auto md_A = memory::desc(Ad, memory::data_type::f16, memory::format_tag::ab);
    auto md_B = memory::desc(Bd, memory::data_type::f16, memory::format_tag::ab);
    auto md_C = memory::desc(Cd, memory::data_type::f32, memory::format_tag::ab);

    auto A = memory(md_A, eng); auto B = memory(md_B, eng); auto C = memory(md_C, eng);
    // content irrelevant for a timing benchmark; DNNL runs full compute regardless

    primitive_attr attr;
    attr.set_fpmath_mode(fpmath_mode::f16);   // route through XMX f16, matches GGML_SYCL_F16
    auto pd = matmul::primitive_desc(eng, md_A, md_B, md_C, attr);
    auto prim = matmul(pd);
    std::unordered_map<int,memory> args={{DNNL_ARG_SRC,A},{DNNL_ARG_WEIGHTS,B},{DNNL_ARG_DST,C}};

    for(int i=0;i<3;++i) prim.execute(strm,args); strm.wait();  // warmup
    const int iters=30; auto t0=std::chrono::high_resolution_clock::now();
    for(int i=0;i<iters;++i) prim.execute(strm,args); strm.wait();
    auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count()/iters;
    double flop=2.0*(double)N*M*K;
    std::printf("  oneDNN f16 matmul: %.3f ms/iter  %.1f GFLOP  %.1f TOPS\n", ms, flop/1e9, flop/(ms/1e3)/1e12);
    std::printf("  fused int8 kernel was 8.0 ms -> oneDNN is %.1fx faster on the GEMM alone\n", 8.0/ms);
    return 0;
}
