// Feasibility probe: int8 joint_matrix (XMX DPAS) GEMM on Battlemage.
// C[M x N] (int32) = A[M x K] (int8) * B[K x N] (int8). One sub-group, one tile.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <cstdio>
#include <vector>
#include <cstdlib>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

constexpr int M = 8;
constexpr int N = 16;
constexpr int K = 32;
constexpr int SG = 16; // sub-group size on Intel

int main() {
    queue q;
    std::printf("device: %s\n", q.get_device().get_info<info::device::name>().c_str());

    std::vector<int8_t>  A(M * K), B(K * N);
    std::vector<int32_t> C(M * N, 0), Cref(M * N, 0);
    srand(1);
    for (auto &x : A) x = (rand() % 5) - 2; // -2..2
    for (auto &x : B) x = (rand() % 5) - 2;
    // CPU reference (row-major A, row-major B)
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            int acc = 0;
            for (int k = 0; k < K; ++k) acc += (int)A[m*K+k] * (int)B[k*N+n];
            Cref[m*N+n] = acc;
        }

    int8_t  *dA = malloc_device<int8_t>(M*K, q);
    int8_t  *dB = malloc_device<int8_t>(K*N, q);
    int32_t *dC = malloc_device<int32_t>(M*N, q);
    std::vector<int8_t> Bvnni(K * N);
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
            Bvnni[(k/4)*N*4 + n*4 + (k%4)] = B[k*N + n];
    q.memcpy(dA, A.data(), M*K).wait();
    q.memcpy(dB, Bvnni.data(), K*N).wait();

    q.submit([&](handler &h) {
        h.parallel_for(nd_range<1>(range<1>(SG), range<1>(SG)),
            [=](nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                auto sg = it.get_sub_group();
                joint_matrix<sub_group, int8_t,  use::a, M, K, layout::row_major> tA;
                joint_matrix<sub_group, int8_t,  use::b, K, N, layout::ext_intel_packed> tB;
                joint_matrix<sub_group, int32_t, use::accumulator, M, N> tC;
                joint_matrix_fill(sg, tC, 0);
                joint_matrix_load(sg, tA, multi_ptr<int8_t, access::address_space::global_space>(dA), K);
                joint_matrix_load(sg, tB, multi_ptr<int8_t, access::address_space::global_space>(dB), N*4);
                joint_matrix_mad(sg, tC, tA, tB, tC);
                joint_matrix_store(sg, tC, multi_ptr<int32_t, access::address_space::global_space>(dC), N, layout::row_major);
            });
    }).wait();

    q.memcpy(C.data(), dC, M*N*sizeof(int32_t)).wait();
    int bad = 0;
    for (int i = 0; i < M*N; ++i) if (C[i] != Cref[i]) { if (bad < 5) std::printf("  mismatch [%d]: got %d want %d\n", i, C[i], Cref[i]); ++bad; }
    std::printf(bad ? "FAIL: %d/%d mismatched\n" : "PASS: all %d match\n", bad ? bad : M*N, M*N);
    free(dA,q); free(dB,q); free(dC,q);
    return bad ? 1 : 0;
}
