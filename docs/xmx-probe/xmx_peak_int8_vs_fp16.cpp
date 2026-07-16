// Measure achievable XMX peak on B70: int8 vs fp16 joint_matrix_mad throughput.
// Each work-item hammers a long chain of mads on register-resident tiles (no memory traffic),
// so this reports near-peak DPAS rate and the int8/fp16 ratio.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <cstdio>
#include <chrono>
using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

constexpr int SG=16;
constexpr long REP=20000;

template<typename T, typename ACC, int M, int N, int K>
double bench(queue&q, const char* tag, int n_wg, int sg_per_wg){
    // dummy device buffers just to satisfy load/store
    T* a=malloc_device<T>(M*K,q); T* b=malloc_device<T>(K*N,q);
    ACC* c=malloc_device<ACC>(M*N*n_wg*sg_per_wg,q);
    q.memset(a,1,M*K*sizeof(T)); q.memset(b,1,K*N*sizeof(T)); q.wait();
    auto run=[&](){ return q.submit([&](handler&h){
        h.parallel_for(nd_range<1>((size_t)n_wg*sg_per_wg*SG, sg_per_wg*SG),[=](nd_item<1> it)[[sycl::reqd_sub_group_size(SG)]]{
            auto sg=it.get_sub_group();
            joint_matrix<sub_group,T,use::a,M,K,layout::row_major> tA;
            joint_matrix<sub_group,T,use::b,K,N,layout::ext_intel_packed> tB;
            joint_matrix<sub_group,ACC,use::accumulator,M,N> tC;
            joint_matrix_load(sg,tA,multi_ptr<T,access::address_space::global_space>(a),K);
            joint_matrix_load(sg,tB,multi_ptr<T,access::address_space::global_space>(b),N);
            joint_matrix_fill(sg,tC,0);
            for(long i=0;i<REP;++i){ joint_matrix_mad(sg,tC,tA,tB,tC); }
            joint_matrix_store(sg,tC,multi_ptr<ACC,access::address_space::global_space>(c+(size_t)(it.get_group(0)*sg_per_wg + it.get_local_id(0)/SG)*M*N),N,layout::row_major);
        });
    }); };
    run().wait();
    int iters=5; auto t0=std::chrono::high_resolution_clock::now();
    for(int i=0;i<iters;++i) run(); q.wait();
    auto t1=std::chrono::high_resolution_clock::now();
    double s=std::chrono::duration<double>(t1-t0).count()/iters;
    double macs=(double)REP*M*N*K*n_wg*sg_per_wg;
    double tops=2.0*macs/s/1e12;
    std::printf("  %-6s M%dN%dK%d wg=%d sg/wg=%d : %.1f TOPS\n",tag,M,N,K,n_wg,sg_per_wg,tops);
    free(a,q);free(b,q);free(c,q);
    return tops;
}

int main(){
    queue q; std::printf("device: %s\n", q.get_device().get_info<info::device::name>().c_str());
    // Occupancy: B70 has 160 Xe cores-ish; use many work-groups.
    int NWG=2048, SGPW=8;
    double i8 = bench<int8_t,int32_t,8,16,32>(q,"int8",NWG,SGPW);
    double f16= bench<sycl::half,float,8,16,16>(q,"fp16",NWG,SGPW);
    std::printf("  int8/fp16 throughput ratio = %.2fx\n", i8/f16);
    return 0;
}
