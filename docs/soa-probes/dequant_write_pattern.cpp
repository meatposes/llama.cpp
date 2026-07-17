// Compare SoA Q2_0 dequant kernels: my 4-scalar-write vs generic-template-style 2-elem half2 write.
// Same SoA layout for all; the profile showed the SoA kernel (0.494s) is ~85% slower than the AoS
// generic template (0.268s). Hypothesis: the half2 vectorized write is the speedup.
#include <sycl/sycl.hpp>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <chrono>
using namespace sycl;
constexpr int QK=128, QSB=QK/4;
struct block_q2_0 { sycl::half d; uint8_t qs[QSB]; };

int main(int argc,char**argv){
    long N=argc>1?atol(argv[1]):17408, K=argc>2?atol(argv[2]):5120;
    queue q{property::queue::in_order()};
    long NBrow=K/QK, nblk=N*NBrow, nel=N*K;
    std::printf("device: %s  N=%ld K=%ld nel=%ld\n",q.get_device().get_info<info::device::name>().c_str(),N,K,nel);
    srand(11);
    std::vector<block_q2_0> aos(nblk);
    for(auto&x:aos){ x.d=sycl::half(0.03f+(rand()%20)*0.005f); for(int b=0;b<QSB;++b) x.qs[b]=rand()&0xff; }
    // current SoA: [all qs][all d]
    long nbytes=nblk*32;
    std::vector<uint8_t> soa((size_t)nblk*34);
    for(long gb=0;gb<nblk;++gb){ for(int b=0;b<QSB;++b) soa[(size_t)gb*32+b]=aos[gb].qs[b]; *(sycl::half*)(soa.data()+(size_t)nbytes+gb*2)=aos[gb].d; }
    uint8_t* dB=malloc_device<uint8_t>(soa.size(),q); q.memcpy(dB,soa.data(),soa.size()).wait();
    sycl::half* dY=malloc_device<sycl::half>(nel,q);
    std::vector<sycl::half> Y(nel);
    auto refv=[&](long gb,int e){ int raw=(aos[gb].qs[e/4]>>((e%4)*2))&3; return (raw-1)*(float)aos[gb].d; };
    auto check=[&](const char*tag){ q.memcpy(Y.data(),dY,nel*2).wait(); int bad=0; for(int t=0;t<300&&!bad;++t){long gb=rand()%nblk;int e=rand()%QK; if(std::fabs((float)Y[(size_t)gb*QK+e]-refv(gb,e))>2e-3f)bad++;} return bad?"FAIL":"ok"; };
    auto timeit=[&](auto fn){ fn().wait(); int it=40; auto a=std::chrono::high_resolution_clock::now(); for(int i=0;i<it;++i)fn(); q.wait(); auto b=std::chrono::high_resolution_clock::now(); return std::chrono::duration<double,std::milli>(b-a).count()/it; };
    const long NB=nblk, NBY=nbytes;

    // K1: current - 1 qs byte (4 elems) per work-item, 4 scalar half writes
    auto k1=[&](){ return q.submit([&](handler&h){ int BS=256; long ng=(NBY+BS-1)/BS;
        h.parallel_for(nd_range<1>((size_t)ng*BS,BS),[=](nd_item<1> it){ long bi=it.get_global_id(0); if(bi>=NBY)return;
            long gb=bi/32,wb=bi%32; uint8_t byte=dB[bi]; float d=(float)*(const sycl::half*)(dB+NBY+gb*2);
            sycl::half* y=dY+(size_t)gb*QK+wb*4;
            y[0]=sycl::half(((byte&3)-1)*d);y[1]=sycl::half((((byte>>2)&3)-1)*d);y[2]=sycl::half((((byte>>4)&3)-1)*d);y[3]=sycl::half((((byte>>6)&3)-1)*d);
        }); }); };
    // K2: 1 qs byte per work-item, but write as two half2 (vectorized). elems 0..3 -> two half2.
    auto k2=[&](){ return q.submit([&](handler&h){ int BS=256; long ng=(NBY+BS-1)/BS;
        h.parallel_for(nd_range<1>((size_t)ng*BS,BS),[=](nd_item<1> it){ long bi=it.get_global_id(0); if(bi>=NBY)return;
            long gb=bi/32,wb=bi%32; uint8_t byte=dB[bi]; sycl::half d=*(const sycl::half*)(dB+NBY+gb*2);
            sycl::half2 lo{ sycl::half(((byte&3)-1))*d, sycl::half(((byte>>2)&3)-1)*d };
            sycl::half2 hi{ sycl::half(((byte>>4)&3)-1)*d, sycl::half(((byte>>6)&3)-1)*d };
            sycl::half2* y=(sycl::half2*)(dY+(size_t)gb*QK+wb*4); y[0]=lo; y[1]=hi;
        }); }); };
    // K3: 2 qs bytes (8 elems) per work-item, four half2 writes (fewer work-items, more per-item ILP)
    auto k3=[&](){ return q.submit([&](handler&h){ int BS=256; long n2=(NBY/2); long ng=(n2+BS-1)/BS;
        h.parallel_for(nd_range<1>((size_t)ng*BS,BS),[=](nd_item<1> it){ long j=it.get_global_id(0); if(j>=n2)return;
            long bi=j*2; long gb=bi/32,wb=bi%32; sycl::half d=*(const sycl::half*)(dB+NBY+gb*2);
            const uint8_t* p=dB+bi; sycl::half2* y=(sycl::half2*)(dY+(size_t)gb*QK+wb*4);
            #pragma unroll
            for(int t=0;t<2;++t){ uint8_t byte=p[t]; y[t*2]={sycl::half(((byte&3)-1))*d,sycl::half(((byte>>2)&3)-1)*d}; y[t*2+1]={sycl::half(((byte>>4)&3)-1)*d,sycl::half(((byte>>6)&3)-1)*d}; }
        }); }); };

    double t1=timeit(k1); const char*c1=check("k1");
    double t2=timeit(k2); const char*c2=check("k2");
    double t3=timeit(k3); const char*c3=check("k3");
    double outMB=(double)nel*2/1e6;
    std::printf("  K1 (4 scalar writes)      : %.3f ms  %.0f GB/s out  %s\n",t1,outMB/(t1/1e3)/1e3,c1);
    std::printf("  K2 (2x half2 write)       : %.3f ms  %.0f GB/s out  %s\n",t2,outMB/(t2/1e3)/1e3,c2);
    std::printf("  K3 (2 bytes, 4x half2)    : %.3f ms  %.0f GB/s out  %s\n",t3,outMB/(t3/1e3)/1e3,c3);
    free(dB,q);free(dY,q); return 0;
}
