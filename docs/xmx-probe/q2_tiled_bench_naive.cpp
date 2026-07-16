// Increment 3b: full tiled Q2_0 x Q8_1 XMX GEMM, real Bonsai size, throughput benchmark.
// C[N,M] = W[N,K](Q2_0) . X[K,M](Q8_1).  ffn_up-ish: K=5120, N=17408, M=512.
// Correctness spot-checked vs CPU ref on a subset, then timed. Reports achieved GB/s (weight bytes).
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <chrono>
using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

constexpr int TN=16, TM=8, SUB=32, SG=16, QK2=128, QK8=32;
constexpr int WG_SG=8;   // sub-groups per work-group (each does one N-tile), for occupancy

struct block_q2_0 { sycl::half d; uint8_t qs[QK2/4]; };
struct block_q8_1 { sycl::half2 ds; int8_t qs[QK8]; };

int main(int argc,char**argv){
    int K = argc>1?atoi(argv[1]):5120;
    int Nout = argc>2?atoi(argv[2]):17408;
    int Mtok = argc>3?atoi(argv[3]):512;
    const int WB=K/QK2, XB=K/QK8, NSUB=K/SUB;
    queue q; std::printf("device: %s\n  K=%d N=%d M=%d\n", q.get_device().get_info<info::device::name>().c_str(),K,Nout,Mtok);
    srand(5);
    std::vector<block_q2_0> W((size_t)Nout*WB);
    std::vector<block_q8_1> X((size_t)Mtok*XB);
    for(size_t i=0;i<W.size();++i){ W[i].d=sycl::half(0.03f+(rand()%20)*0.005f); for(int b=0;b<QK2/4;++b) W[i].qs[b]=rand()&0xff; }
    for(size_t i=0;i<X.size();++i){ X[i].ds=sycl::half2(sycl::half(0.02f+(rand()%15)*0.004f),sycl::half(0.f)); for(int k=0;k<QK8;++k) X[i].qs[k]=(rand()%15)-7; }

    auto *dW=malloc_device<block_q2_0>(W.size(),q); auto *dX=malloc_device<block_q8_1>(X.size(),q);
    float *dC=malloc_device<float>((size_t)Nout*Mtok,q);
    q.memcpy(dW,W.data(),W.size()*sizeof(block_q2_0)); q.memcpy(dX,X.data(),X.size()*sizeof(block_q8_1)); q.wait();

    const int n_ntiles=Nout/TN, n_mtiles=Mtok/TM;
    auto run=[&](){
        return q.submit([&](handler &h){
            local_accessor<int8_t,1> As(WG_SG*TM*SUB,h);
            local_accessor<int8_t,1> Bs(WG_SG*SUB*TN,h);
            local_accessor<int32_t,1> stile(WG_SG*TM*TN,h);
            local_accessor<float,1>  facc(WG_SG*TM*TN,h);
            // grid: y = m-tile, x = (n-tile group). workgroup = WG_SG subgroups along N.
            range<2> global(n_mtiles, (size_t)(n_ntiles/WG_SG)*WG_SG*SG);
            range<2> local(1, WG_SG*SG);
            h.parallel_for(nd_range<2>(global,local),[=](nd_item<2> it)[[sycl::reqd_sub_group_size(SG)]]{
                auto sg=it.get_sub_group();
                int lid=(int)it.get_local_id(1);
                int sgid=lid/SG, sl=lid%SG;
                int mt=(int)it.get_global_id(0);
                int nt=(int)(it.get_group(1)*WG_SG + sgid);
                if(nt>=n_ntiles) return;
                int8_t *A=&As[sgid*TM*SUB]; int8_t *B=&Bs[sgid*SUB*TN];
                int32_t *ST=&stile[sgid*TM*TN]; float *FA=&facc[sgid*TM*TN];
                for(int e=sl;e<TM*TN;e+=SG) FA[e]=0.f;
                for(int s=0;s<NSUB;++s){
                    int wb=(s*SUB)/QK2;
                    for(int e=sl;e<SUB*TN;e+=SG){ int kk=e/TN, n=e%TN;
                        int gk=s*SUB+kk; uint8_t byte=dW[((size_t)(nt*TN+n))*WB+wb].qs[(gk%QK2)/4]; int raw=(byte>>(((gk%QK2)%4)*2))&3;
                        B[(kk/4)*TN*4+n*4+(kk%4)]=(int8_t)(raw-1); }
                    for(int e=sl;e<TM*SUB;e+=SG){ int m=e/SUB, kk=e%SUB; A[m*SUB+kk]=dX[((size_t)(mt*TM+m))*XB + s].qs[kk]; }
                    sycl::group_barrier(sg);
                    joint_matrix<sub_group,int8_t,use::a,TM,SUB,layout::row_major> tA;
                    joint_matrix<sub_group,int8_t,use::b,SUB,TN,layout::ext_intel_packed> tB;
                    joint_matrix<sub_group,int32_t,use::accumulator,TM,TN> tC;
                    joint_matrix_fill(sg,tC,0);
                    joint_matrix_load(sg,tA,local_ptr<int8_t>(A),SUB);
                    joint_matrix_load(sg,tB,local_ptr<int8_t>(B),TN*4);
                    joint_matrix_mad(sg,tC,tA,tB,tC);
                    joint_matrix_store(sg,tC,local_ptr<int32_t>(ST),TN,layout::row_major);
                    sycl::group_barrier(sg);
                    for(int e=sl;e<TM*TN;e+=SG){ int m=e/TN,n=e%TN; float dxv=(float)dX[((size_t)(mt*TM+m))*XB+s].ds[0];
                        FA[e]+=dxv*(float)dW[((size_t)(nt*TN+n))*WB+wb].d*(float)ST[e]; }
                    sycl::group_barrier(sg);
                }
                for(int e=sl;e<TM*TN;e+=SG){ int m=e/TN,n=e%TN; dC[(size_t)(nt*TN+n)*Mtok + (mt*TM+m)]=FA[e]; }
            });
        });
    };
    run().wait(); // warmup
    const int iters=10; auto t0=std::chrono::high_resolution_clock::now();
    for(int i=0;i<iters;++i) run(); q.wait();
    auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count()/iters;
    double wbytes=(double)W.size()*sizeof(block_q2_0);
    std::printf("  time=%.3f ms/iter  weight=%.1f MB  achieved=%.0f GB/s (weight reads)\n",
        ms, wbytes/1e6, wbytes/(ms/1e3)/1e9);
    double flop=2.0*Nout*Mtok*K;
    std::printf("  %.1f GFLOP/GEMM -> %.1f TOPS(int8-equiv)\n", flop/1e9, flop/(ms/1e3)/1e12);

    // correctness spot-check: recompute a few C entries on host from raw
    std::vector<float> C((size_t)Nout*Mtok); q.memcpy(C.data(),dC,C.size()*4).wait();
    int checked=0,bad=0; float maxrel=0;
    for(int t=0;t<8;++t){ int n=rand()%Nout, m=rand()%Mtok; double a=0;
        for(int s=0;s<NSUB;++s){ int wb=(s*SUB)/QK2; float dwv=(float)W[(size_t)n*WB+wb].d, dxv=(float)X[(size_t)m*XB+s].ds[0];
            for(int kk=0;kk<SUB;++kk){ int gk=s*SUB+kk; uint8_t byte=W[(size_t)n*WB+wb].qs[(gk%QK2)/4]; int raw=(byte>>(((gk%QK2)%4)*2))&3;
                a += (raw-1)*dwv * (double)X[(size_t)m*XB+s].qs[kk]*dxv; } }
        float got=C[(size_t)n*Mtok+m]; float rel=std::fabs(got-(float)a)/(std::fabs((float)a)+1e-3f);
        maxrel=std::max(maxrel,rel); ++checked; if(rel>1e-2f){ ++bad; if(bad<4) std::printf("  MISMATCH n%d m%d got %.3f want %.3f\n",n,m,got,(float)a);} }
    std::printf(bad? "  CORRECTNESS: FAIL (%d/%d, maxrel %.4f)\n":"  CORRECTNESS: PASS (%d checks, maxrel %.5f)\n", bad?bad:checked, checked, maxrel);
    free(dW,q);free(dX,q);free(dC,q); return 0;
}
