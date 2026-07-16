// Increment 2a: validate Q2_0->int8(raw-1) expansion + VNNI pack + XMX int32 accumulation.
// Kernel stores per-subblock int32 tiles; host applies d_x*d_w scaling and sums; compare to ref.
// N=16 out, M=8 tok, K=128 (1 Q2_0 block deep = 4 Q8_1 sub-blocks).
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
    std::vector<uint8_t> w_raw(N * K);   // Q2_0 raw 0..3
    std::vector<float>   dw(N);
    for (int n=0;n<N;++n){ dw[n]=0.05f+(rand()%20)*0.01f; for(int k=0;k<K;++k) w_raw[n*K+k]=rand()%4; }
    std::vector<int8_t> x_raw(M * K);    // Q8_1 raw int8
    std::vector<float>  dx(M * NSUB);
    for(int m=0;m<M;++m){ for(int s=0;s<NSUB;++s) dx[m*NSUB+s]=0.02f+(rand()%15)*0.01f; for(int k=0;k<K;++k) x_raw[m*K+k]=(rand()%15)-7; }

    // Reference: C[n,m] = sum_k (raw_w-1)*dw[n] * x_raw*dx[m,k/32]
    std::vector<float> Cref(N*M,0);
    for(int n=0;n<N;++n) for(int m=0;m<M;++m){ double a=0;
        for(int k=0;k<K;++k) a += ((int)w_raw[n*K+k]-1)*dw[n] * (double)x_raw[m*K+k]*dx[m*NSUB+(k/SUB)];
        Cref[n*M+m]=(float)a; }

    // Host prep int8 tiles per sub-block: A=X[M,32] row-major, B=W(raw-1)[32,N] VNNI.
    std::vector<int8_t> A(NSUB*M*SUB), Bv(NSUB*SUB*N);
    for(int s=0;s<NSUB;++s){
        for(int m=0;m<M;++m) for(int kk=0;kk<SUB;++kk) A[s*M*SUB+m*SUB+kk]=x_raw[m*K+s*SUB+kk];
        for(int kk=0;kk<SUB;++kk) for(int n=0;n<N;++n)
            Bv[s*SUB*N + (kk/4)*N*4 + n*4 + (kk%4)] = (int8_t)((int)w_raw[n*K+s*SUB+kk]-1);
    }
    int8_t *dA=malloc_device<int8_t>(A.size(),q), *dB=malloc_device<int8_t>(Bv.size(),q);
    int32_t *dS=malloc_device<int32_t>(NSUB*M*N,q);   // per-subblock int32 tiles [s][m][n]
    q.memcpy(dA,A.data(),A.size()); q.memcpy(dB,Bv.data(),Bv.size()); q.wait();

    q.submit([&](handler &h){ h.parallel_for(nd_range<1>(SG,SG),[=](nd_item<1> it)[[sycl::reqd_sub_group_size(SG)]]{
        auto sg=it.get_sub_group();
        for(int s=0;s<NSUB;++s){
            joint_matrix<sub_group,int8_t,use::a,M,SUB,layout::row_major> tA;
            joint_matrix<sub_group,int8_t,use::b,SUB,N,layout::ext_intel_packed> tB;
            joint_matrix<sub_group,int32_t,use::accumulator,M,N> tC;
            joint_matrix_fill(sg,tC,0);
            joint_matrix_load(sg,tA,multi_ptr<int8_t,access::address_space::global_space>(dA+s*M*SUB),SUB);
            joint_matrix_load(sg,tB,multi_ptr<int8_t,access::address_space::global_space>(dB+s*SUB*N),N*4);
            joint_matrix_mad(sg,tC,tA,tB,tC);
            joint_matrix_store(sg,tC,multi_ptr<int32_t,access::address_space::global_space>(dS+s*M*N),N,layout::row_major);
        }
    });}).wait();

    std::vector<int32_t> S(NSUB*M*N);
    q.memcpy(S.data(),dS,S.size()*4).wait();
    // Host scale+sum: C[n,m] = sum_s dx[m,s]*dw[n]*S[s][m][n]  (S stored row-major [m,n])
    int bad=0; float maxerr=0;
    for(int n=0;n<N;++n) for(int m=0;m<M;++m){ double a=0;
        for(int s=0;s<NSUB;++s) a += (double)dx[m*NSUB+s]*dw[n]*S[s*M*N+m*N+n];
        float e=std::fabs((float)a-Cref[n*M+m]); maxerr=std::max(maxerr,e);
        if(e>1e-3f){ if(bad<5) std::printf("  [n%d m%d] got %.4f want %.4f\n",n,m,(float)a,Cref[n*M+m]); ++bad; } }
    std::printf(bad? "FAIL: %d/%d bad (maxerr %.5f)\n" : "PASS: all %d match (maxerr %.6f)\n", bad?bad:N*M, N*M>0?N*M:0, bad?0.0f:maxerr);
    if(!bad) std::printf("  -> Q2_0 expand + VNNI + XMX int32 accumulation CORRECT\n");
    free(dA,q);free(dB,q);free(dS,q);
    return bad?1:0;
}
