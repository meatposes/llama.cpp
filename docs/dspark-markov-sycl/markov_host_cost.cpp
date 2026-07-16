// Measure the dspark Markov-head host GEMV cost: [n_vocab x rank] . [rank], per drafted token.
#include <cstdio>
#include <vector>
#include <chrono>
#include <cstdlib>
int main(){
    const long V=248320, R=256;
    std::vector<float> w((size_t)V*R), emb(R), out(V);
    for(auto&x:w) x=(rand()%100)*0.01f; for(auto&x:emb) x=0.5f;
    double mb=(double)V*R*4/1e6;
    // naive host loop (the #else path in speculative.cpp)
    auto t0=std::chrono::high_resolution_clock::now();
    for(int it=0;it<4;++it){ // block_size=4 tokens
      for(long v=0;v<V;++v){ const float*row=w.data()+(size_t)v*R; float b=0; for(long r=0;r<R;++r) b+=emb[r]*row[r]; out[v]=b; }
    }
    auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    printf("naive host GEMV: %.1f ms for 4 tokens (block_size=4), %.1f MB weight, %.1f GB/s\n",
           ms, mb, 4*mb/1e3/(ms/1e3));
    printf("  -> ~%.1f ms per draft round on CPU. Target decode is ~24 ms/token (42 t/s).\n", ms);
    return (int)out[0]&0;
}
