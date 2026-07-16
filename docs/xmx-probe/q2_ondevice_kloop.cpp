// Increment 3: on-device expand from real block_q2_0/block_q8_1, K=256 (2 blocks),
// per-block dw folded into the SLM accumulation. Validates the last novel mechanics.
// N=16 out, M=8 tok. PASS maxerr ~0.001 on B70.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cstdint>
using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

constexpr int N=16, M=8, K=256, SUB=32, NSUB=K/SUB, SG=16;  // K=256 = 2 Q2_0 blocks
constexpr int QK2=128, QK8=32;

// ggml-matching layouts
struct block_q2_0 { sycl::half d; uint8_t qs[QK2/4]; };          // 32 bytes qs
struct block_q8_1 { sycl::half2 ds; int8_t qs[QK8]; };            // d,sum ; 32 int8

int main(){
    queue q; std::printf("device: %s\n", q.get_device().get_info<info::device::name>().c_str());
    srand(3);
    // build N weight rows (1 block each), M activation rows (4 q8_1 blocks each)
    constexpr int WB=K/QK2; std::vector<block_q2_0> W(N*WB);   // W[n*WB+blk]
    std::vector<block_q8_1> X(M*NSUB);     // X[m*NSUB + s]
    std::vector<uint8_t> w_raw(N*K); std::vector<int8_t> x_raw(M*K);
    std::vector<float> dw(N*(K/QK2)), dx(M*NSUB);
    for(int n=0;n<N;++n) for(int wb=0;wb<WB;++wb){ dw[n*WB+wb]=0.05f+(rand()%20)*0.01f; W[n*WB+wb].d=sycl::half(dw[n*WB+wb]);
        for(int b=0;b<QK2/4;++b){ uint8_t byte=0; for(int j=0;j<4;++j){ uint8_t r=rand()%4; w_raw[n*K+wb*QK2+b*4+j]=r; byte|=(r&3)<<(j*2);} W[n*WB+wb].qs[b]=byte; } }
    for(int m=0;m<M;++m) for(int s=0;s<NSUB;++s){ dx[m*NSUB+s]=0.02f+(rand()%15)*0.01f; X[m*NSUB+s].ds=sycl::half2(sycl::half(dx[m*NSUB+s]),sycl::half(0.f));
        for(int k=0;k<QK8;++k){ int8_t v=(rand()%15)-7; x_raw[m*K+s*QK8+k]=v; X[m*NSUB+s].qs[k]=v; } }

    std::vector<float> Cref(N*M,0);
    for(int n=0;n<N;++n) for(int m=0;m<M;++m){ double a=0;
        for(int k=0;k<K;++k) a+=((int)w_raw[n*K+k]-1)*dw[n*WB+(k/QK2)]*(double)x_raw[m*K+k]*dx[m*NSUB+(k/SUB)];
        Cref[n*M+m]=(float)a; }

    auto *dW=malloc_device<block_q2_0>(N*WB,q); auto *dX=malloc_device<block_q8_1>(M*NSUB,q);
    float *dC=malloc_device<float>(N*M,q);
    q.memcpy(dW,W.data(),N*WB*sizeof(block_q2_0)); q.memcpy(dX,X.data(),M*NSUB*sizeof(block_q8_1)); q.wait();

    q.submit([&](handler &h){
        local_accessor<int8_t,1> As(M*SUB,h);      // A int8 [M,32] row-major
        local_accessor<int8_t,1> Bs(SUB*N,h);      // B int8 VNNI [32,N]
        local_accessor<int32_t,1> stile(M*N,h);
        local_accessor<float,1>  facc(M*N,h);
        h.parallel_for(nd_range<1>(SG,SG),[=](nd_item<1> it)[[sycl::reqd_sub_group_size(SG)]]{
            auto sg=it.get_sub_group(); int lid=it.get_local_id(0);
            for(int e=lid;e<M*N;e+=SG) facc[e]=0.f;
            it.barrier(access::fence_space::local_space);
            for(int s=0;s<NSUB;++s){
                // expand B: weight (raw-1) for this sub-block, VNNI pack into Bs
                for(int e=lid;e<SUB*N;e+=SG){ int kk=e/N, n=e%N;
                    int gk=s*SUB+kk; int wb=gk/QK2; int lk=gk%QK2; uint8_t byte=dW[n*WB+wb].qs[lk/4]; int raw=(byte>>((lk%4)*2))&3;
                    Bs[(kk/4)*N*4 + n*4 + (kk%4)] = (int8_t)(raw-1); }
                // load A: q8_1 int8 directly
                for(int e=lid;e<M*SUB;e+=SG){ int m=e/SUB, kk=e%SUB; As[m*SUB+kk]=dX[m*NSUB+s].qs[kk]; }
                it.barrier(access::fence_space::local_space);
                joint_matrix<sub_group,int8_t,use::a,M,SUB,layout::row_major> tA;
                joint_matrix<sub_group,int8_t,use::b,SUB,N,layout::ext_intel_packed> tB;
                joint_matrix<sub_group,int32_t,use::accumulator,M,N> tC;
                joint_matrix_fill(sg,tC,0);
                joint_matrix_load(sg,tA,local_ptr<int8_t>(As.get_multi_ptr<access::decorated::no>().get()),SUB);
                joint_matrix_load(sg,tB,local_ptr<int8_t>(Bs.get_multi_ptr<access::decorated::no>().get()),N*4);
                joint_matrix_mad(sg,tC,tA,tB,tC);
                joint_matrix_store(sg,tC,local_ptr<int32_t>(stile.get_multi_ptr<access::decorated::no>().get()),N,layout::row_major);
                it.barrier(access::fence_space::local_space);
                { int wb=(s*SUB)/QK2; for(int e=lid;e<M*N;e+=SG){ int m=e/N, n=e%N; float dxv=(float)dX[m*NSUB+s].ds[0]; facc[e]+=dxv*(float)dW[n*WB+wb].d*(float)stile[e]; } }
                it.barrier(access::fence_space::local_space);
            }
            for(int e=lid;e<M*N;e+=SG){ int m=e/N, n=e%N; dC[n*M+m]=facc[e]; }
        });
    }).wait();

    std::vector<float> C(N*M); q.memcpy(C.data(),dC,N*M*4).wait();
    int bad=0; float maxerr=0;
    for(int i=0;i<N*M;++i){ float e=std::fabs(C[i]-Cref[i]); maxerr=std::max(maxerr,e);
        if(e>2e-3f){ if(bad<5) std::printf("  [%d] got %.4f want %.4f\n",i,C[i],Cref[i]); ++bad; } }
    std::printf(bad? "FAIL: %d/%d bad (maxerr %.5f)\n":"PASS: all %d match (maxerr %.6f)\n", bad?bad:N*M, N*M, maxerr);
    if(!bad) std::printf("  -> on-device expand from real block_q2_0/block_q8_1 CORRECT\n");
    free(dW,q);free(dX,q);free(dC,q);
    return bad?1:0;
}
