// Increment 2b: fused single-kernel Q2_0 x Q8_1 -> float via XMX + SLM-staged scaling.
// Per sub-block: XMX int32 tile -> store to SLM -> scale by dx[m,s], accumulate into float SLM.
// Final: multiply by dw[n], write out. Validates the fused pattern the real kernel will use.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cmath>
using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

constexpr int N = 16, M = 8, K = 128, SUB = 32, NSUB = K / SUB, SG = 16;

int main() {
    queue q;
    std::printf("device: %s\n", q.get_device().get_info<info::device::name>().c_str());
    srand(3);
    std::vector<uint8_t> w_raw(N*K); std::vector<float> dw(N);
    for(int n=0;n<N;++n){ dw[n]=0.05f+(rand()%20)*0.01f; for(int k=0;k<K;++k) w_raw[n*K+k]=rand()%4; }
    std::vector<int8_t> x_raw(M*K); std::vector<float> dx(M*NSUB);
    for(int m=0;m<M;++m){ for(int s=0;s<NSUB;++s) dx[m*NSUB+s]=0.02f+(rand()%15)*0.01f; for(int k=0;k<K;++k) x_raw[m*K+k]=(rand()%15)-7; }
    std::vector<float> Cref(N*M,0);
    for(int n=0;n<N;++n) for(int m=0;m<M;++m){ double a=0;
        for(int k=0;k<K;++k) a+=((int)w_raw[n*K+k]-1)*dw[n]*(double)x_raw[m*K+k]*dx[m*NSUB+(k/SUB)];
        Cref[n*M+m]=(float)a; }

    std::vector<int8_t> A(NSUB*M*SUB), Bv(NSUB*SUB*N);
    for(int s=0;s<NSUB;++s){
        for(int m=0;m<M;++m) for(int kk=0;kk<SUB;++kk) A[s*M*SUB+m*SUB+kk]=x_raw[m*K+s*SUB+kk];
        for(int kk=0;kk<SUB;++kk) for(int n=0;n<N;++n)
            Bv[s*SUB*N+(kk/4)*N*4+n*4+(kk%4)]=(int8_t)((int)w_raw[n*K+s*SUB+kk]-1);
    }
    int8_t *dA=malloc_device<int8_t>(A.size(),q), *dB=malloc_device<int8_t>(Bv.size(),q);
    float *ddw=malloc_device<float>(N,q), *ddx=malloc_device<float>(M*NSUB,q), *dC=malloc_device<float>(N*M,q);
    q.memcpy(dA,A.data(),A.size()); q.memcpy(dB,Bv.data(),Bv.size());
    q.memcpy(ddw,dw.data(),N*4); q.memcpy(ddx,dx.data(),M*NSUB*4); q.wait();

    q.submit([&](handler &h){
        local_accessor<int32_t,1> stile(M*N,h);   // int32 XMX tile
        local_accessor<float,1>   facc(M*N,h);     // float accumulator [m*N+n]
        h.parallel_for(nd_range<1>(SG,SG),[=](nd_item<1> it)[[sycl::reqd_sub_group_size(SG)]]{
            auto sg=it.get_sub_group();
            int lid=it.get_local_id(0);
            for(int e=lid;e<M*N;e+=SG) facc[e]=0.f;
            it.barrier(access::fence_space::local_space);
            for(int s=0;s<NSUB;++s){
                joint_matrix<sub_group,int8_t,use::a,M,SUB,layout::row_major> tA;
                joint_matrix<sub_group,int8_t,use::b,SUB,N,layout::ext_intel_packed> tB;
                joint_matrix<sub_group,int32_t,use::accumulator,M,N> tC;
                joint_matrix_fill(sg,tC,0);
                joint_matrix_load(sg,tA,multi_ptr<int8_t,access::address_space::global_space>(dA+s*M*SUB),SUB);
                joint_matrix_load(sg,tB,multi_ptr<int8_t,access::address_space::global_space>(dB+s*SUB*N),N*4);
                joint_matrix_mad(sg,tC,tA,tB,tC);
                joint_matrix_store(sg,tC,local_ptr<int32_t>(stile.get_pointer()),N,layout::row_major);
                it.barrier(access::fence_space::local_space);
                for(int e=lid;e<M*N;e+=SG){ int m=e/N; facc[e]+=ddx[m*NSUB+s]*(float)stile[e]; }
                it.barrier(access::fence_space::local_space);
            }
            // final dw[n] multiply, write C[n,m] (transpose to match ref layout [n*M+m])
            for(int e=lid;e<M*N;e+=SG){ int m=e/N, n=e%N; dC[n*M+m]=facc[e]*ddw[n]; }
        });
    }).wait();

    std::vector<float> C(N*M); q.memcpy(C.data(),dC,N*M*4).wait();
    int bad=0; float maxerr=0;
    for(int i=0;i<N*M;++i){ float e=std::fabs(C[i]-Cref[i]); maxerr=std::max(maxerr,e);
        if(e>1e-3f){ if(bad<5) std::printf("  [%d] got %.4f want %.4f\n",i,C[i],Cref[i]); ++bad; } }
    std::printf(bad? "FAIL: %d/%d bad (maxerr %.5f)\n":"PASS: all %d match (maxerr %.6f)\n", bad?bad:N*M, N*M, bad?0.f:maxerr);
    if(!bad) std::printf("  -> fused XMX + SLM scaling CORRECT\n");
    free(dA,q);free(dB,q);free(ddw,q);free(ddx,q);free(dC,q);
    return bad?1:0;
}
