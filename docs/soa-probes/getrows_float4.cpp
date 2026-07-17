// get_rows gather-copy: scalar (1 elem/item, current) vs float4 (4 elem/item). rows x ne00 F32.
#include <sycl/sycl.hpp>
#include <cstdio>
#include <vector>
#include <chrono>
using namespace sycl;
int main(){
    const long rows=4096, ne00=5120;           // gather 4096 rows of 5120 f32
    queue q{property::queue::in_order()};
    long NR=131072;                             // source table rows (like a small vocab/state)
    float* src=malloc_device<float>((size_t)NR*ne00,q);
    int32_t* idx=malloc_device<int32_t>(rows,q);
    float* dst=malloc_device<float>((size_t)rows*ne00,q);
    { std::vector<int32_t> h(rows); for(auto&x:h)x=rand()%NR; q.memcpy(idx,h.data(),rows*4);} q.memset(src,1,(size_t)NR*ne00*4).wait();
    auto sc=[&](){ return q.submit([&](handler&h){ int BS=256; long ng=(ne00+BS-1)/BS;
        h.parallel_for(nd_range<3>(range<3>(rows,1,ng*BS),range<3>(1,1,BS)),[=](nd_item<3> it){
            long i00=it.get_group(2)*BS+it.get_local_id(2); if(i00>=ne00)return; long r=it.get_group(0);
            dst[(size_t)r*ne00+i00]=src[(size_t)idx[r]*ne00+i00];
        }); }); };
    auto v4=[&](){ return q.submit([&](handler&h){ int BS=256; long n4=ne00/4; long ng=(n4+BS-1)/BS;
        h.parallel_for(nd_range<3>(range<3>(rows,1,ng*BS),range<3>(1,1,BS)),[=](nd_item<3> it){
            long j=it.get_group(2)*BS+it.get_local_id(2); if(j>=n4)return; long r=it.get_group(0);
            ((sycl::float4*)(dst+(size_t)r*ne00))[j]=((const sycl::float4*)(src+(size_t)idx[r]*ne00))[j];
        }); }); };
    auto t=[&](auto fn){ fn().wait(); int it=50; auto a=std::chrono::high_resolution_clock::now(); for(int i=0;i<it;++i)fn(); q.wait(); auto b=std::chrono::high_resolution_clock::now(); return std::chrono::duration<double,std::milli>(b-a).count()/it; };
    double mb=(double)rows*ne00*4/1e6;
    double ts=t(sc), tv=t(v4);
    printf("device: %s\n  scalar: %.3f ms (%.0f GB/s)\n  float4: %.3f ms (%.0f GB/s)  %.2fx\n",
        q.get_device().get_info<info::device::name>().c_str(), ts, 2*mb/(ts/1e3)/1e3, tv, 2*mb/(tv/1e3)/1e3, ts/tv);
    return 0;
}
