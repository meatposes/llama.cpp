// fp16-fused Q2_0 GEMM: dequant Q2_0->fp16 weights in-register (d_w baked in), fp16 activations,
// clean fp16 DPAS accumulate across all K. NO per-sub-block scaling - nothing between mads.
// C[N,M] f32 = W[N,K](Q2_0->fp16) . X[K,M](fp16).  ffn_up: K=5120 N=17408 M=512.  Bar: 1.11 ms.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <chrono>
using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

constexpr int TN=16, TM=8, TK=16, SG=16, QK2=128;   // fp16 DPAS tile M8 N16 K16, VNNI factor 2
constexpr int MSTRIP=4, WG_SG=8;

struct block_q2_0 { sycl::half d; uint8_t qs[QK2/4]; };

int main(int argc,char**argv){
    int K=argc>1?atoi(argv[1]):5120, Nout=argc>2?atoi(argv[2]):17408, Mtok=argc>3?atoi(argv[3]):512;
    const int WB=K/QK2, NKT=K/TK;   // K-tiles of 16
    queue q; std::printf("device: %s  K=%d N=%d M=%d\n",q.get_device().get_info<info::device::name>().c_str(),K,Nout,Mtok);
    srand(5);
    std::vector<block_q2_0> W((size_t)Nout*WB);
    std::vector<sycl::half> X((size_t)Mtok*K);   // fp16 activations [M,K]
    std::vector<uint8_t> wraw((size_t)Nout*K); std::vector<float> dw((size_t)Nout*WB);
    for(int n=0;n<Nout;++n) for(int wb=0;wb<WB;++wb){ dw[(size_t)n*WB+wb]=0.03f+(rand()%20)*0.005f; W[(size_t)n*WB+wb].d=sycl::half(dw[(size_t)n*WB+wb]);
        for(int b=0;b<QK2/4;++b){ uint8_t by=0; for(int j=0;j<4;++j){ uint8_t r=rand()%4; wraw[(size_t)n*K+wb*QK2+b*4+j]=r; by|=(r&3)<<(j*2);} W[(size_t)n*WB+wb].qs[b]=by; } }
    for(size_t i=0;i<X.size();++i) X[i]=sycl::half((float)((rand()%15)-7)*0.02f);

    auto *dW=malloc_device<block_q2_0>(W.size(),q); auto *dX=malloc_device<sycl::half>(X.size(),q);
    float *dC=malloc_device<float>((size_t)Nout*Mtok,q);
    q.memcpy(dW,W.data(),W.size()*sizeof(block_q2_0)); q.memcpy(dX,X.data(),X.size()*sizeof(sycl::half)); q.wait();

    const int n_ntiles=Nout/TN, n_strips=Mtok/(TM*MSTRIP);
    auto run=[&](){ return q.submit([&](handler&h){
        local_accessor<sycl::half,1> Bs(WG_SG*TK*TN,h);
        range<2> global(n_strips,(size_t)n_ntiles*SG);
        range<2> local(1,WG_SG*SG);
        h.parallel_for(nd_range<2>(global,local),[=](nd_item<2> it)[[sycl::reqd_sub_group_size(SG)]]{
            auto sg=it.get_sub_group(); int lid=(int)it.get_local_id(1); int sgid=lid/SG, sl=lid%SG;
            int strip=(int)it.get_global_id(0);
            int nt=(int)(it.get_group(1)*WG_SG+sgid); if(nt>=n_ntiles) return;
            sycl::half *B=&Bs[sgid*TK*TN];
            joint_matrix<sub_group,float,use::accumulator,TM,TN> facc[MSTRIP];
            for(int mi=0;mi<MSTRIP;++mi) joint_matrix_fill(sg,facc[mi],0.f);
            int m0=strip*TM*MSTRIP;
            for(int kt=0;kt<NKT;++kt){
                int wb=(kt*TK)/QK2;
                // expand weight K-tile -> fp16 (raw-1)*dw, VNNI factor 2 into SLM
                for(int e=sl;e<TK*TN;e+=SG){ int kk=e/TN,n=e%TN; int gk=kt*TK+kk;
                    uint8_t byte=dW[((size_t)(nt*TN+n))*WB+wb].qs[(gk%QK2)/4]; int raw=(byte>>(((gk%QK2)%4)*2))&3;
                    float dwv=(float)dW[((size_t)(nt*TN+n))*WB+wb].d;
                    B[(kk/2)*TN*2 + n*2 + (kk%2)] = sycl::half((raw-1)*dwv); }
                sycl::group_barrier(sg);
                joint_matrix<sub_group,sycl::half,use::b,TK,TN,layout::ext_intel_packed> tB;
                joint_matrix_load(sg,tB,local_ptr<sycl::half>(B),TN*2);
                for(int mi=0;mi<MSTRIP;++mi){
                    int mt=(m0/TM)+mi;
                    const sycl::half* aptr=dX + (size_t)(mt*TM)*K + kt*TK;   // [TM,TK] row-major, stride K
                    joint_matrix<sub_group,sycl::half,use::a,TM,TK,layout::row_major> tA;
                    joint_matrix_load(sg,tA,multi_ptr<const sycl::half,access::address_space::global_space>(aptr),K);
                    joint_matrix_mad(sg,facc[mi],tA,tB,facc[mi]);   // clean accumulate, no scaling
                }
                sycl::group_barrier(sg);
            }
            for(int mi=0;mi<MSTRIP;++mi){ int mt=(m0/TM)+mi;
                joint_matrix_store(sg,facc[mi],multi_ptr<float,access::address_space::global_space>(dC+(size_t)(nt*TN)*Mtok+(mt*TM)),Mtok,layout::col_major); }
        });
    }); };
    run().wait();
    int iters=20; auto t0=std::chrono::high_resolution_clock::now();
    for(int i=0;i<iters;++i) run(); q.wait();
    auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count()/iters;
    double flop=2.0*Nout*Mtok*K;
    std::printf("  fp16-fused: %.3f ms/iter  %.1f TOPS  (bar: 1.11 ms current path; oneDNN GEMM 0.732)\n", ms, flop/(ms/1e3)/1e12);

    std::vector<float> C((size_t)Nout*Mtok); q.memcpy(C.data(),dC,C.size()*4).wait();
    int bad=0; float maxrel=0;
    for(int t=0;t<8;++t){ int n=rand()%Nout,m=rand()%Mtok; double a=0;
        for(int k=0;k<K;++k){ int wb=k/QK2; uint8_t byte=W[(size_t)n*WB+wb].qs[(k%QK2)/4]; int raw=(byte>>(((k%QK2)%4)*2))&3;
            a += (raw-1)*(float)W[(size_t)n*WB+wb].d * (float)X[(size_t)m*K+k]; }
        float got=C[(size_t)n*Mtok+m], rel=std::fabs(got-(float)a)/(std::fabs((float)a)+1e-3f); maxrel=std::max(maxrel,rel);
        if(rel>2e-2f){++bad; if(bad<4) std::printf("  MISMATCH n%d m%d got %.3f want %.3f\n",n,m,got,(float)a);} }
    std::printf(bad?"  CORRECTNESS: FAIL (%d, maxrel %.4f)\n":"  CORRECTNESS: PASS (maxrel %.5f)\n", bad?bad:0, maxrel);
    free(dW,q);free(dX,q);free(dC,q); return 0;
}
