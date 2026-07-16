// Isolate Q2_0 -> F16 dequant time for one ffn_up weight [N=17408, K=5120].
// Replicates the two real kernels from ggml-sycl/dequantize.hpp:
//   AoS: dequantize_block pattern (2 elems/work-item, reads block_q2_0)
//   SoA: the deployed reorder kernel (1 qs byte / 4 elems per work-item) - what the server uses.
#include <sycl/sycl.hpp>
#include <cstdio>
#include <vector>
#include <chrono>
#include <cstdint>
using namespace sycl;
constexpr int QK2=128;
struct block_q2_0 { sycl::half d; uint8_t qs[QK2/4]; };

int main(int argc,char**argv){
    long N=argc>1?atol(argv[1]):17408, K=argc>2?atol(argv[2]):5120;
    long nel=N*K; long nblk=nel/QK2;
    queue q; std::printf("device: %s  weight [N=%ld K=%ld] = %ld elems, %ld blocks\n",
        q.get_device().get_info<info::device::name>().c_str(),N,K,nel,nblk);

    // AoS buffer: nblk block_q2_0. SoA buffer: [qs region nblk*32][d region nblk*half].
    size_t aos_bytes=(size_t)nblk*sizeof(block_q2_0);
    size_t soa_bytes=(size_t)nblk*(QK2/4) + (size_t)nblk*sizeof(sycl::half);
    uint8_t *dAoS=malloc_device<uint8_t>(aos_bytes,q);
    uint8_t *dSoA=malloc_device<uint8_t>(soa_bytes,q);
    sycl::half *dY=malloc_device<sycl::half>(nel,q);
    q.memset(dAoS,0x11,aos_bytes); q.memset(dSoA,0x11,soa_bytes); q.wait();
    double outMB=(double)nel*sizeof(sycl::half)/1e6, inMB=(double)aos_bytes/1e6;

    // AoS kernel: generic dequantize_block, 2 elems/work-item, 256 wi/group
    auto runAoS=[&](){ return q.submit([&](handler&h){
        int BS=256; long ng=(nel/2 + BS-1)/BS;
        h.parallel_for(nd_range<1>((size_t)ng*BS,BS),[=](nd_item<1> it){
            long i=2*(long)it.get_global_id(0); if(i>=nel) return;
            const block_q2_0* x=(const block_q2_0*)dAoS; long ib=i/QK2; int iqs=(i%QK2);
            sycl::half d=x[ib].d; int byte=iqs/4;
            int r0=(x[ib].qs[byte]>>((iqs%4)*2))&3, r1=(x[ib].qs[byte]>>(((iqs+1)%4)*2))&3;
            dY[i]=sycl::half((float)(r0-1)*(float)d); dY[i+1]=sycl::half((float)(r1-1)*(float)d);
        });
    }); };
    // SoA kernel: deployed reorder dequant, 1 qs byte (4 elems) per work-item, 256 wi/group
    auto runSoA=[&](){ return q.submit([&](handler&h){
        int BS=256; long nbytes=nel/4; long ng=(nbytes+BS-1)/BS;
        h.parallel_for(nd_range<1>((size_t)ng*BS,BS),[=](nd_item<1> it){
            long bi=(long)it.get_global_id(0); if(bi>=nbytes) return;
            uint8_t b=dSoA[bi]; const sycl::half* s=(const sycl::half*)(dSoA+nbytes)+bi/(QK2/4);
            float d=(float)*s; sycl::half* y=dY+bi*4;
            y[0]=sycl::half(((b&3)-1)*d); y[1]=sycl::half((((b>>2)&3)-1)*d);
            y[2]=sycl::half((((b>>4)&3)-1)*d); y[3]=sycl::half((((b>>6)&3)-1)*d);
        });
    }); };

    auto time=[&](auto fn,const char*tag){ fn().wait(); int it=30;
        auto t0=std::chrono::high_resolution_clock::now();
        for(int i=0;i<it;++i) fn(); q.wait();
        auto t1=std::chrono::high_resolution_clock::now();
        double ms=std::chrono::duration<double,std::milli>(t1-t0).count()/it;
        double gb=(inMB+outMB)/1e3;
        std::printf("  %-4s dequant: %.3f ms/iter  (in %.1f MB + out %.1f MB, %.0f GB/s)\n",tag,ms,inMB,outMB,gb/(ms/1e3));
    };
    time(runAoS,"AoS"); time(runSoA,"SoA");
    std::printf("  [oneDNN f16 GEMM for this shape was 0.732 ms; add dequant for the current-path total]\n");
    free(dAoS,q);free(dSoA,q);free(dY,q); return 0;
}
