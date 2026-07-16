// Amortized tiled Q2_0 x Q8_1 XMX GEMM. Sub-group owns one N16 tile, processes MSTRIP m-tiles.
// Weight sub-block expanded to SLM VNNI ONCE per sub-block, reused across the m-strip.
// Activations loaded DIRECTLY from global into tA (no SLM). Only barrier: after weight expand.
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
constexpr int MSTRIP=4;      // m-tiles per sub-group (M covered = 8*8 = 64)
constexpr int WG_SG=8;       // sub-groups per work-group

struct block_q2_0 { sycl::half d; uint8_t qs[QK2/4]; };
struct block_q8_1 { sycl::half2 ds; int8_t qs[QK8]; };

int main(int argc,char**argv){
    int K=argc>1?atoi(argv[1]):5120, Nout=argc>2?atoi(argv[2]):17408, Mtok=argc>3?atoi(argv[3]):512;
    const int WB=K/QK2, XB=K/QK8, NSUB=K/SUB;
    queue q; std::printf("device: %s  K=%d N=%d M=%d MSTRIP=%d\n",q.get_device().get_info<info::device::name>().c_str(),K,Nout,Mtok,MSTRIP);
    srand(5);
    std::vector<block_q2_0> W((size_t)Nout*WB);
    std::vector<block_q8_1> X((size_t)Mtok*XB);
    for(size_t i=0;i<W.size();++i){ W[i].d=sycl::half(0.03f+(rand()%20)*0.005f); for(int b=0;b<QK2/4;++b) W[i].qs[b]=rand()&0xff; }
    for(size_t i=0;i<X.size();++i){ X[i].ds=sycl::half2(sycl::half(0.02f+(rand()%15)*0.004f),sycl::half(0.f)); for(int k=0;k<QK8;++k) X[i].qs[k]=(rand()%15)-7; }
    auto *dW=malloc_device<block_q2_0>(W.size(),q); auto *dX=malloc_device<block_q8_1>(X.size(),q);
    float *dC=malloc_device<float>((size_t)Nout*Mtok,q);
    q.memcpy(dW,W.data(),W.size()*sizeof(block_q2_0)); q.memcpy(dX,X.data(),X.size()*sizeof(block_q8_1)); q.wait();

    const int n_ntiles=Nout/TN, n_strips=Mtok/(TM*MSTRIP);
    // stride (in int8 elements) between consecutive tokens' sub-block-s activation data:
    const int Astride = XB*sizeof(block_q8_1);   // bytes; block_q8_1 laid out ds(4B)+qs(32B)=36B
    auto run=[&](){ return q.submit([&](handler&h){
        local_accessor<int8_t,1> Bs(WG_SG*SUB*TN,h);
        local_accessor<float,1>  FA(WG_SG*MSTRIP*TM*TN,h);
        local_accessor<float,1>  dwL(WG_SG*TN,h);
        local_accessor<float,1>  dxL(WG_SG*TM,h);
        range<2> global(n_strips,(size_t)n_ntiles*SG);
        range<2> local(1,WG_SG*SG);
        h.parallel_for(nd_range<2>(global,local),[=](nd_item<2> it)[[sycl::reqd_sub_group_size(SG)]]{
            auto sg=it.get_sub_group(); int lid=(int)it.get_local_id(1); int sgid=lid/SG, sl=lid%SG;
            int strip=(int)it.get_global_id(0);
            int nt=(int)(it.get_group(1)*WG_SG+sgid);
            if(nt>=n_ntiles) return;
            int8_t *B=&Bs[sgid*SUB*TN];
            float *dwl=&dwL[sgid*TN]; float *dxl=&dxL[sgid*TM];
            (void)FA;
            joint_matrix<sub_group,float,use::accumulator,TM,TN> facc[MSTRIP];
            for(int mi=0;mi<MSTRIP;++mi) joint_matrix_fill(sg,facc[mi],0.f);
            int m0=strip*TM*MSTRIP;
            int cur_wb=-1;
            for(int s=0;s<NSUB;++s){
                int wb=(s*SUB)/QK2;
                if(wb!=cur_wb){ cur_wb=wb; if(sl<TN) dwl[sl]=(float)dW[((size_t)(nt*TN+sl))*WB+wb].d; }
                for(int e=sl;e<SUB*TN;e+=SG){ int kk=e/TN,n=e%TN; int gk=s*SUB+kk;
                    uint8_t byte=dW[((size_t)(nt*TN+n))*WB+wb].qs[(gk%QK2)/4]; int raw=(byte>>(((gk%QK2)%4)*2))&3;
                    B[(kk/4)*TN*4+n*4+(kk%4)]=(int8_t)(raw-1); }
                sycl::group_barrier(sg);
                joint_matrix<sub_group,int8_t,use::b,SUB,TN,layout::ext_intel_packed> tB;
                joint_matrix_load(sg,tB,local_ptr<int8_t>(B),TN*4);
                for(int mi=0;mi<MSTRIP;++mi){
                    int mt=(m0/TM)+mi;
                    const int8_t* aptr=dX[((size_t)(mt*TM))*XB + s].qs; // token 0 of this m-tile, sub-block s
                    joint_matrix<sub_group,int8_t,use::a,TM,SUB,layout::row_major> tA;
                    joint_matrix<sub_group,int32_t,use::accumulator,TM,TN> tC;
                    joint_matrix_fill(sg,tC,0);
                    joint_matrix_load(sg,tA,multi_ptr<const int8_t,access::address_space::global_space>(aptr),Astride);
                    joint_matrix_mad(sg,tC,tA,tB,tC);
                    if(sl<TM) dxl[sl]=(float)dX[((size_t)(mt*TM+sl))*XB+s].ds[0];
                    auto wiC=sycl::ext::oneapi::detail::get_wi_data(sg,tC);
                    auto wiF=sycl::ext::oneapi::detail::get_wi_data(sg,facc[mi]);
                    for(int i=0;i<wiC.length();++i){ auto [row,col]=wiC[i].get_coord(); wiF[i]=wiF[i]+dxl[row]*dwl[col]*(float)(int32_t)wiC[i]; }
                }
                sycl::group_barrier(sg);
            }
            for(int mi=0;mi<MSTRIP;++mi){ int mt=(m0/TM)+mi;
                joint_matrix_store(sg,facc[mi],multi_ptr<float,access::address_space::global_space>(dC+(size_t)(nt*TN)*Mtok+(mt*TM)),Mtok,layout::col_major); }
        });
    }); };
    run().wait();
    int iters=10; auto t0=std::chrono::high_resolution_clock::now();
    for(int i=0;i<iters;++i) run(); q.wait();
    auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count()/iters;
    double flop=2.0*Nout*Mtok*K;
    std::printf("  time=%.3f ms/iter  %.1f TOPS(int8)\n", ms, flop/(ms/1e3)/1e12);

    std::vector<float> C((size_t)Nout*Mtok); q.memcpy(C.data(),dC,C.size()*4).wait();
    int bad=0; float maxrel=0;
    for(int t=0;t<8;++t){ int n=rand()%Nout,m=rand()%Mtok; double a=0;
        for(int s=0;s<NSUB;++s){ int wb=(s*SUB)/QK2; float dwv=(float)W[(size_t)n*WB+wb].d,dxv=(float)X[(size_t)m*XB+s].ds[0];
            for(int kk=0;kk<SUB;++kk){ int gk=s*SUB+kk; uint8_t byte=W[(size_t)n*WB+wb].qs[(gk%QK2)/4]; int raw=(byte>>(((gk%QK2)%4)*2))&3;
                a+=(raw-1)*dwv*(double)X[(size_t)m*XB+s].qs[kk]*dxv; } }
        float got=C[(size_t)n*Mtok+m], rel=std::fabs(got-(float)a)/(std::fabs((float)a)+1e-3f); maxrel=std::max(maxrel,rel);
        if(rel>1e-2f){++bad; if(bad<4) std::printf("  MISMATCH n%d m%d got %.3f want %.3f\n",n,m,got,(float)a);} }
    std::printf(bad?"  CORRECTNESS: FAIL (%d, maxrel %.4f)\n":"  CORRECTNESS: PASS (maxrel %.5f)\n", bad?bad:0, maxrel);
    free(dW,q);free(dX,q);free(dC,q); return 0;
}
