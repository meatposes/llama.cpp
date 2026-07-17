// SoA dequant layout sweep. For a real ffn_up-size Q2_0 weight, build AoS + superblock-SoA(S) for
// several S, and measure BOTH: (a) dequant Q2_0->F16 throughput, (b) MMVQ-style weight-read
// throughput (one sub-group per output row reads qs+d across K blocks and dots). Correctness vs a
// CPU reference for each. Find S that speeds dequant without regressing the MMVQ read.
//
// Layouts (per the real backend, QK2_0=128, 32 qs bytes + 1 half scale per block):
//   AoS:  [d(2B) qs(32B)] per block, contiguous.  (== S=1 interleave, MMVQ strided)
//   SoA(S): repeating [ S blocks' qs (S*32B) ][ S blocks' d (S*2B) ].  S=nblocks == current SoA.
#include <sycl/sycl.hpp>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cstdint>
using namespace sycl;
constexpr int QK=128, QSB=QK/4;   // 32 qs bytes/block
struct block_q2_0 { sycl::half d; uint8_t qs[QSB]; };

int main(int argc,char**argv){
    long N = argc>1?atol(argv[1]):17408;   // out rows
    long K = argc>2?atol(argv[2]):5120;     // in cols
    queue q{property::queue::in_order()};
    long NBrow = K/QK;                 // blocks per row (40)
    long nblk = N*NBrow;               // total blocks
    std::printf("device: %s  N=%ld K=%ld  blocks/row=%ld total=%ld\n",
        q.get_device().get_info<info::device::name>().c_str(),N,K,NBrow,nblk);
    srand(11);
    // source AoS blocks (row-major: row r blocks at r*NBrow .. )
    std::vector<block_q2_0> aos(nblk);
    for(size_t i=0;i<aos.size();++i){ aos[i].d=sycl::half(0.03f+(rand()%20)*0.005f); for(int b=0;b<QSB;++b) aos[i].qs[b]=rand()&0xff; }

    // ---- build a layout buffer for superblock size S (S in blocks). qs region + interleaved d per superblock.
    // layout byte size = nblk*(32+2). For block gb: superblock sb=gb/S, within=gb%S.
    //   qs at: sb*(S*34) + within*32
    //   d  at: sb*(S*34) + S*32 + within*2
    auto build=[&](long S)->std::vector<uint8_t>{
        std::vector<uint8_t> buf((size_t)nblk*34);
        for(long gb=0;gb<nblk;++gb){ long sb=gb/S, within=gb%S; size_t base=(size_t)sb*(S*34);
            uint8_t* qd=buf.data()+base+within*32; for(int b=0;b<QSB;++b) qd[b]=aos[gb].qs[b];
            *(sycl::half*)(buf.data()+base+(size_t)S*32+within*2)=aos[gb].d; }
        return buf;
    };
    // CPU reference dequant (row-major F16 out [N,K])
    auto refval=[&](long gb,int elem)->float{ int raw=(aos[gb].qs[elem/4]>>((elem%4)*2))&3; return (raw-1)*(float)aos[gb].d; };

    sycl::half* dY=malloc_device<sycl::half>((size_t)N*K,q);
    std::vector<sycl::half> Y((size_t)N*K);

    auto test_layout=[&](long S){
        auto buf=build(S);
        uint8_t* dBuf=malloc_device<uint8_t>(buf.size(),q);
        q.memcpy(dBuf,buf.data(),buf.size()).wait();
        const long SS=S;
        // (a) DEQUANT: 1 qs byte (4 elems) per work-item, coalesced writes. Layout-aware addressing.
        auto dq=[&](){ return q.submit([&](handler&h){ int BS=256; long nbytes=(long)nblk*32; long ng=(nbytes+BS-1)/BS;
            h.parallel_for(nd_range<1>((size_t)ng*BS,BS),[=](nd_item<1> it){
                long bi=(long)it.get_global_id(0); if(bi>=nbytes) return;   // bi = global qs-byte index
                long gb=bi/32, wb=bi%32;                                    // block, which qs byte
                long sb=gb/SS, within=gb%SS; size_t base=(size_t)sb*(SS*34);
                uint8_t byte=dBuf[base+within*32+wb];
                float d=(float)*(const sycl::half*)(dBuf+base+(size_t)SS*32+within*2);
                sycl::half* y=dY+(size_t)gb*QK + wb*4;
                y[0]=sycl::half(((byte&3)-1)*d); y[1]=sycl::half((((byte>>2)&3)-1)*d);
                y[2]=sycl::half((((byte>>4)&3)-1)*d); y[3]=sycl::half((((byte>>6)&3)-1)*d);
            }); }); };
        // (b) MMVQ-READ proxy: one sub-group (16) per output row, reads all its blocks' qs+d, dots with a dummy activation, writes a scalar. Measures weight-read efficiency in MMVQ's per-row pattern.
        float* dOut=malloc_device<float>(N,q);
        auto mv=[&](){ return q.submit([&](handler&h){ int SGN=16;
            h.parallel_for(nd_range<1>((size_t)N*SGN,SGN),[=](nd_item<1> it)[[sycl::reqd_sub_group_size(16)]]{
                long row=it.get_global_id(0)/SGN; int lane=it.get_local_id(0)%SGN;
                float acc=0;
                for(long lb=lane; lb<NBrow; lb+=SGN){ long gb=row*NBrow+lb;    // this lane's blocks in the row
                    long sb=gb/SS, within=gb%SS; size_t base=(size_t)sb*(SS*34);
                    const uint8_t* qs=dBuf+base+within*32;
                    float d=(float)*(const sycl::half*)(dBuf+base+(size_t)SS*32+within*2);
                    int s=0;
                    #pragma unroll
                    for(int b=0;b<32;++b){ uint8_t v=qs[b]; s+=(v&3)+((v>>2)&3)+((v>>4)&3)+((v>>6)&3); }
                    acc+=d*s;
                }
                if(lane==0) dOut[row]=acc;
            }); }); };
        // correctness (dequant)
        dq().wait(); q.memcpy(Y.data(),dY,Y.size()*2).wait();
        int bad=0; for(int t=0;t<200 && !bad;++t){ long gb=rand()%nblk; int e=rand()%QK; float got=(float)Y[(size_t)gb*QK+e], want=refval(gb,e); if(std::fabs(got-want)>2e-3f) bad++; }
        // time dequant
        int it=30; dq().wait(); auto t0=std::chrono::high_resolution_clock::now(); for(int i=0;i<it;++i) dq(); q.wait(); auto t1=std::chrono::high_resolution_clock::now();
        double dq_ms=std::chrono::duration<double,std::milli>(t1-t0).count()/it;
        // time mmvq-read
        mv().wait(); auto t2=std::chrono::high_resolution_clock::now(); for(int i=0;i<it;++i) mv(); q.wait(); auto t3=std::chrono::high_resolution_clock::now();
        double mv_ms=std::chrono::duration<double,std::milli>(t3-t2).count()/it;
        double wbytes=(double)nblk*32;
        std::printf("  S=%-6ld dequant=%.3f ms (%.0f GB/s)  mmvq-read=%.3f ms (%.0f GB/s)  %s\n",
            S, dq_ms, wbytes/(dq_ms/1e3)/1e9, mv_ms, wbytes/(mv_ms/1e3)/1e9, bad?"CORRECT-FAIL":"ok");
        free(dBuf,q); free(dOut,q);
    };
    std::printf("=== layout sweep (S = superblock size in blocks; S=1 ~AoS, S=%ld = current SoA) ===\n", nblk);
    for(long S : {1L, 2L, 4L, 8L, 16L, 40L, 80L, nblk}) test_layout(S);
    free(dY,q);
    return 0;
}
