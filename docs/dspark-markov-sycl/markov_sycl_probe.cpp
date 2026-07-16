// Standalone SYCL dspark Markov resample: correction[v]=dot(w1[prev],w2[v]); out=argmax(base+corr),
// prev chained on device across n_use positions. Validate vs CPU ref, time vs 113 ms host baseline.
#include <sycl/sycl.hpp>
#include <cstdio>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cfloat>
using namespace sycl;

constexpr int WG=256, NPART=512;   // NPART work-groups (partials), WG threads each

int main(){
    const long V=248320, R=256; const int n_use=4;
    queue q{property::queue::in_order()}; std::printf("device: %s  V=%ld R=%ld n_use=%d\n",q.get_device().get_info<info::device::name>().c_str(),V,R,n_use);
    srand(7);
    std::vector<float> w1((size_t)V*R), w2((size_t)V*R), base((size_t)n_use*V);
    for(auto&x:w1) x=(rand()%200-100)*0.01f;
    for(auto&x:w2) x=(rand()%200-100)*0.01f;
    for(auto&x:base) x=(rand()%1000-500)*0.01f;
    int32_t anchor=12345;

    // CPU reference (sequential chaining)
    std::vector<int32_t> ref(n_use); int32_t prev=anchor;
    for(int k=0;k<n_use;++k){ float bv=-FLT_MAX; int32_t bi=0;
        const float* w1p=w1.data()+(size_t)prev*R;
        for(long v=0;v<V;++v){ const float* w2r=w2.data()+(size_t)v*R; float c=0; for(int r=0;r<R;++r) c+=w1p[r]*w2r[r];
            float lg=base[(size_t)k*V+v]+c; if(lg>bv){bv=lg;bi=(int32_t)v;} }
        ref[k]=bi; prev=bi; }

    float *dW1=malloc_device<float>(w1.size(),q), *dW2=malloc_device<float>(w2.size(),q), *dB=malloc_device<float>(base.size(),q);
    float *dPV=malloc_device<float>(NPART,q); int32_t *dPI=malloc_device<int32_t>(NPART,q);
    int32_t *dPrev=malloc_device<int32_t>(1,q), *dOut=malloc_device<int32_t>(n_use,q);
    q.memcpy(dW1,w1.data(),w1.size()*4); q.memcpy(dW2,w2.data(),w2.size()*4); q.memcpy(dB,base.data(),base.size()*4);
    q.memcpy(dPrev,&anchor,4); q.wait();

    auto resample=[&](){
      for(int k=0;k<n_use;++k){
        // stage 1: partial gemv+argmax over vocab stripes
        q.submit([&](handler&h){
          local_accessor<float,1> w1s(R,h); local_accessor<float,1> sv(WG,h); local_accessor<int32_t,1> si(WG,h);
          h.parallel_for(nd_range<1>((size_t)NPART*WG,WG),[=](nd_item<1> it){
            int tid=(int)it.get_local_id(0); long wg=it.get_group(0);
            int32_t prev=dPrev[0]; const float* w1p=dW1+(size_t)prev*R;
            for(int r=tid;r<R;r+=WG) w1s[r]=w1p[r];
            it.barrier(access::fence_space::local_space);
            float bv=-FLT_MAX; int32_t bi=0;
            for(long v=wg*WG+tid; v<V; v+=(long)NPART*WG){ const float* w2r=dW2+(size_t)v*R; float c=0;
              for(int r=0;r<R;++r) c+=w1s[r]*w2r[r];
              float lg=dB[(size_t)k*V+v]+c; if(lg>bv||(lg==bv&&(int32_t)v<bi)){bv=lg;bi=(int32_t)v;} }
            sv[tid]=bv; si[tid]=bi; it.barrier(access::fence_space::local_space);
            for(int s=WG/2;s>0;s>>=1){ if(tid<s){ if(sv[tid+s]>sv[tid]||(sv[tid+s]==sv[tid]&&si[tid+s]<si[tid])){sv[tid]=sv[tid+s];si[tid]=si[tid+s];} } it.barrier(access::fence_space::local_space); }
            if(tid==0){ dPV[wg]=sv[0]; dPI[wg]=si[0]; }
          });
        });
        // stage 2: final reduce partials -> out[k], prev
        q.submit([&](handler&h){
          local_accessor<float,1> sv(WG,h); local_accessor<int32_t,1> si(WG,h);
          h.parallel_for(nd_range<1>(WG,WG),[=](nd_item<1> it){
            int tid=(int)it.get_local_id(0); float bv=-FLT_MAX; int32_t bi=0;
            for(int i=tid;i<NPART;i+=WG){ if(dPV[i]>bv||(dPV[i]==bv&&dPI[i]<bi)){bv=dPV[i];bi=dPI[i];} }
            sv[tid]=bv; si[tid]=bi; it.barrier(access::fence_space::local_space);
            for(int s=WG/2;s>0;s>>=1){ if(tid<s){ if(sv[tid+s]>sv[tid]||(sv[tid+s]==sv[tid]&&si[tid+s]<si[tid])){sv[tid]=sv[tid+s];si[tid]=si[tid+s];} } it.barrier(access::fence_space::local_space); }
            if(tid==0){ dOut[k]=si[0]; dPrev[0]=si[0]; }
          });
        });
      }
    };
    resample(); q.wait();
    std::vector<int32_t> out(n_use); q.memcpy(out.data(),dOut,n_use*4).wait();
    int bad=0; for(int k=0;k<n_use;++k) if(out[k]!=ref[k]){ ++bad; std::printf("  k%d got %d want %d\n",k,out[k],ref[k]); }
    std::printf(bad?"  CORRECTNESS: FAIL (%d)\n":"  CORRECTNESS: PASS (all %d chained argmax match)\n", bad?bad:n_use);

    // reset prev, time
    q.memcpy(dPrev,&anchor,4).wait();
    int iters=50; auto t0=std::chrono::high_resolution_clock::now();
    for(int i=0;i<iters;++i){ q.memcpy(dPrev,&anchor,4); resample(); } q.wait();
    auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count()/iters;
    std::printf("  SYCL resample: %.3f ms/round (n_use=4)   [host was 113 ms; target decode ~24 ms]\n", ms);
    free(dW1,q);free(dW2,q);free(dB,q);free(dPV,q);free(dPI,q);free(dPrev,q);free(dOut,q);
    return bad?1:0;
}
