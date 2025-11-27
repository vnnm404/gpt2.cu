
#include <cstdio>
#include <iostream>
#include <cuda.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#define BLOCK 64


__global__ void add(float* A, float* B, float* C, int N, int bidx){
    // asm volatile(
    //     ".reg .b32 	%r<31>;"
    //     "mov.u32 %r10, 0x0;");
//     asm volatile(
//         "	.reg .f32 	%f<25>;\n"

//     "ld.global.f32 %%f1, [%0];\n"
//     "ld.global.f32 %%f2, [%1];\n"
//     "add.f32 %%f3, %%f1, %%f2;\n"
//     "st.global.f32 [%2], %%f3;\n"
//     :
//     : "l"(A), "l"(B), "l"(C)
// );
    // for(int i = 0; i < 2; i++){
    //     printf("%f\n", A[i]);
    // }

    asm volatile( 
        "{\n"
"	.reg .pred 	%p<7>;\n"
"	.reg .b32 	%r<31>;\n"
"	.reg .f32 	%f<25>;\n"
".reg .b64 %rd12;\n"
".reg .b64 %rd13;\n"
".reg .b64 %rd14;\n"
"mov.u64 	%rd12, %0;\n"
"mov.u64 	%rd13, %1;\n"
"mov.u64 	%rd14, %2;\n"
".reg .b64 %rd<11>;\n"

"	// begin inline asm\n"
"	mov.u32 %%r1, %%ctaid.x;\n"
"	// end inline asm\n"
"	shl.b32 	%%r26, %%r1, 10;\n"
"	mov.u32 	%%r27, %%tid.x;\n"
"	shl.b32 	%%r28, %%r27, 2;\n"
"	and.b32  	%%r29, %%r28, 508;\n"
"	or.b32  	%%r30, %%r26, %%r29;\n"
"	mul.wide.s32 	%%rd10, %%r30, 4;\n"
"	add.s64 	%%rd5, %%rd14, %%rd10;\n"
"	add.s64 	%%rd6, %%rd5, 2048;\n"
"	add.s64 	%%rd1, %%rd12, %%rd10;\n"
"	add.s64 	%%rd2, %%rd1, 2048;\n"
"	mov.pred 	%%p1, -1;\n"
"	// begin inline asm\n"
"	mov.u32 %%r2, 0x0;\n"
"	mov.u32 %%r3, 0x0;\n"
"	mov.u32 %%r4, 0x0;\n"
"	mov.u32 %%r5, 0x0;\n"
"	@%%p1 ld.global.v4.b32 { %%r2, %%r3, %%r4, %%r5 }, [ %%rd1 + 0 ];\n"
"	// end inline asm\n"
"	mov.b32 	%%f1, %%r2;\n"
"	mov.b32 	%%f2, %%r3;\n"
"	mov.b32 	%%f3, %%r4;\n"
"	mov.b32 	%%f4, %%r5;\n"
"	// begin inline asm\n"
"	mov.u32 %%r6, 0x0;\n"
"	mov.u32 %%r7, 0x0;\n"
"	mov.u32 %%r8, 0x0;\n"
"	mov.u32 %%r9, 0x0;\n"
"	@%%p1 ld.global.v4.b32 { %%r6, %%r7, %%r8, %%r9 }, [ %%rd2 + 0 ];\n"
"	// end inline asm\n"
"	mov.b32 	%%f5, %%r6;\n"
"	mov.b32 	%%f6, %%r7;\n"
"	mov.b32 	%%f7, %%r8;\n"
"	mov.b32 	%%f8, %%r9;\n"
"	add.s64 	%%rd3, %%rd13, %%rd10;\n"
"	add.s64 	%%rd4, %%rd3, 2048;\n"
"	// begin inline asm\n"
"	mov.u32 %%r10, 0x0;\n"
"	mov.u32 %%r11, 0x0;\n"
"	mov.u32 %%r12, 0x0;\n"
"	mov.u32 %%r13, 0x0;\n"
"	@%%p1 ld.global.v4.b32 { %%r10, %%r11, %%r12, %%r13 }, [ %%rd3 + 0 ];\n"
"	// end inline asm\n"
"	mov.b32 	%%f9, %%r10;\n"
"	mov.b32 	%%f10, %%r11;\n"
"	mov.b32 	%%f11, %%r12;\n"
"	mov.b32 	%%f12, %%r13;\n"
"	// begin inline asm\n"
"	mov.u32 %%r14, 0x0;\n"
"	mov.u32 %%r15, 0x0;\n"
"	mov.u32 %%r16, 0x0;\n"
"	mov.u32 %%r17, 0x0;\n"
"	@%%p1 ld.global.v4.b32 { %%r14, %%r15, %%r16, %%r17 }, [ %%rd4 + 0 ];\n"
"	// end inline asm\n"
"	mov.b32 	%%f13, %%r14;\n"
"	mov.b32 	%%f14, %%r15;\n"
"	mov.b32 	%%f15, %%r16;\n"
"	mov.b32 	%%f16, %%r17;\n"
"	add.f32 	%%f17, %%f1, %%f9;\n"
"	add.f32 	%%f18, %%f2, %%f10;\n"
"	add.f32 	%%f19, %%f3, %%f11;\n"
"	add.f32 	%%f20, %%f4, %%f12;\n"
"	add.f32 	%%f21, %%f5, %%f13;\n"
"	add.f32 	%%f22, %%f6, %%f14;\n"
"	add.f32 	%%f23, %%f7, %%f15;\n"
"	add.f32 	%%f24, %%f8, %%f16;\n"
"	mov.b32 	%%r18, %%f17;\n"
"	mov.b32 	%%r19, %%f18;\n"
"	mov.b32 	%%r20, %%f19;\n"
"	mov.b32 	%%r21, %%f20;\n"
"	// begin inline asm\n"
"	@%%p1 st.global.v4.b32 [ %%rd5 + 0 ], { %%r18, %%r19, %%r20, %%r21 };\n"
"	// end inline asm\n"
"	mov.b32 	%%r22, %%f21;\n"
"	mov.b32 	%%r23, %%f22;\n"
"	mov.b32 	%%r24, %%f23;\n"
"	mov.b32 	%%r25, %%f24;\n"
"	// begin inline asm\n"
"	@%%p1 st.global.v4.b32 [ %%rd6 + 0 ], { %%r22, %%r23, %%r24, %%r25 };\n"
"	// end inline asm\n"
"                                        // -- End function\n"
"}\n"

    : :"l"(A), "l"(B), "l"(C));
}

__global__ void add_norm(float* A, float* B, float* C, int N, int bidx){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

__global__ void matmul_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    int lda, int lda1,
    int ldb, int ldb1,
    int ldc, int ldc1,
    int BLOCK_M, int BLOCK_N, int BLOCK_K
){
    asm volatile(
      "{\n"
        ".shared .b32 global_smem[1024];  \n"
".reg .u32 %%r1167;\n"
"mov.u32 %%r1167, %8;\n"
".reg .u32 %%r1168;\n"
"mov.u32 %%r1168, %7;\n"
".reg .u32 %%r1169;\n"
"mov.u32 %%r1169, %6;\n"
".reg .u32 %%r1170;\n"
"mov.u32 %%r1170, %5;\n"
".reg .u64 %%rd117;\n"
"mov.u64 %%rd117, %2;\n"
".reg .u64 %%rd118;\n"
"mov.u64 %%rd118, %1;\n"
".reg .u64 %%rd119;\n"
"mov.u64 %%rd119, %0;\n"
"	.reg .pred 	%%p<80>;\n"
"	.reg .b32 	%%r<1166>;\n"
"	.reg .f32 	%%f<610>;\n"
"	.reg .b64 	%%rd<116>;\n"
"$L__func_begin0:\n"
"\n"
"// %%bb.0:\n"
"$L__tmp0:\n"
"	// begin inline asm\n"
"	mov.u32 %%r212, %%ctaid.x;\n"
"	// end inline asm\n"
"	// begin inline asm\n"
"	mov.u32 %%r213, %%ctaid.y;\n"
"	// end inline asm\n"
"	shl.b32 	%%r1, %%r212, 6;\n"
"	mov.u32 	%%r2, %%tid.x;\n"
"	bfe.u32 	%%r3, %%r2, 4, 3;\n"
"	shl.b32 	%%r4, %%r2, 2;\n"
"	and.b32  	%%r5, %%r4, 60;\n"
"	or.b32  	%%r6, %%r1, %%r3;\n"
"	or.b32  	%%r7, %%r6, 8;\n"
"	or.b32  	%%r8, %%r6, 16;\n"
"	or.b32  	%%r9, %%r6, 24;\n"
"	or.b32  	%%r10, %%r6, 32;\n"
"	or.b32  	%%r11, %%r6, 40;\n"
"	or.b32  	%%r12, %%r6, 48;\n"
"	or.b32  	%%r13, %%r6, 56;\n"
"	shl.b32 	%%r14, %%r213, 6;\n"
"	or.b32  	%%r15, %%r14, %%r5;\n"
"	mul.lo.s32 	%%r289, %%r6, %%r1169;\n"
"	mul.lo.s32 	%%r290, %%r7, %%r1169;\n"
"	mul.lo.s32 	%%r291, %%r8, %%r1169;\n"
"	mul.lo.s32 	%%r292, %%r9, %%r1169;\n"
"	mul.lo.s32 	%%r293, %%r10, %%r1169;\n"
"	mul.lo.s32 	%%r294, %%r11, %%r1169;\n"
"	mul.lo.s32 	%%r295, %%r12, %%r1169;\n"
"	mul.lo.s32 	%%r296, %%r13, %%r1169;\n"
"	setp.lt.s32 	%%p33, %%r1170, 1;\n"
"	setp.gt.s32 	%%p34, %%r1170, 0;\n"
"	add.s32 	%%r297, %%r289, %%r5;\n"
"	add.s32 	%%r298, %%r290, %%r5;\n"
"	add.s32 	%%r299, %%r291, %%r5;\n"
"	add.s32 	%%r300, %%r292, %%r5;\n"
"	add.s32 	%%r301, %%r293, %%r5;\n"
"	add.s32 	%%r302, %%r294, %%r5;\n"
"	add.s32 	%%r303, %%r295, %%r5;\n"
"	add.s32 	%%r304, %%r296, %%r5;\n"
"	mul.wide.s32 	%%rd36, %%r297, 4;\n"
"	add.s64 	%%rd4, %%rd119, %%rd36;\n"
"	mul.wide.s32 	%%rd37, %%r298, 4;\n"
"	add.s64 	%%rd5, %%rd119, %%rd37;\n"
"	mul.wide.s32 	%%rd38, %%r299, 4;\n"
"	add.s64 	%%rd6, %%rd119, %%rd38;\n"
"	mul.wide.s32 	%%rd39, %%r300, 4;\n"
"	add.s64 	%%rd7, %%rd119, %%rd39;\n"
"	mul.wide.s32 	%%rd40, %%r301, 4;\n"
"	add.s64 	%%rd8, %%rd119, %%rd40;\n"
"	mul.wide.s32 	%%rd41, %%r302, 4;\n"
"	add.s64 	%%rd9, %%rd119, %%rd41;\n"
"	mul.wide.s32 	%%rd42, %%r303, 4;\n"
"	add.s64 	%%rd10, %%rd119, %%rd42;\n"
"	mul.wide.s32 	%%rd43, %%r304, 4;\n"
"	add.s64 	%%rd11, %%rd119, %%rd43;\n"
"	shr.u32 	%%r305, %%r2, 2;\n"
"	and.b32  	%%r16, %%r305, 8;\n"
"	and.b32  	%%r306, %%r305, 20;\n"
"	xor.b32  	%%r307, %%r306, %%r5;\n"
"	xor.b32  	%%r308, %%r307, %%r16;\n"
"	shl.b32 	%%r309, %%r3, 6;\n"
"	or.b32  	%%r18, %%r308, %%r309;\n"
"	shl.b32 	%%r310, %%r18, 2;\n"
"	mov.u32 	%%r311, global_smem;\n"
"	add.s32 	%%r214, %%r311, %%r310;\n"
"	add.s32 	%%r216, %%r214, 2048;\n"
"	add.s32 	%%r218, %%r214, 4096;\n"
"	add.s32 	%%r220, %%r214, 6144;\n"
"	add.s32 	%%r222, %%r214, 8192;\n"
"	add.s32 	%%r224, %%r214, 10240;\n"
"	add.s32 	%%r226, %%r214, 12288;\n"
"	add.s32 	%%r228, %%r214, 14336;\n"
"	selp.b32 	%%r215, 16, 0, %%p34;\n"
"	mov.pred 	%%p56, -1;\n"
"	// begin inline asm\n"
"	@%%p56 cp.async.cg.shared.global [ %%r214 + 0 ], [ %%rd4 + 0 ], 0x10, %%r215;\n"
"	// end inline asm\n"
"	// begin inline asm\n"
"	@%%p56 cp.async.cg.shared.global [ %%r216 + 0 ], [ %%rd5 + 0 ], 0x10, %%r215;\n"
"	// end inline asm\n"
"	// begin inline asm\n"
"	@%%p56 cp.async.cg.shared.global [ %%r218 + 0 ], [ %%rd6 + 0 ], 0x10, %%r215;\n"
"	// end inline asm\n"
"	// begin inline asm\n"
"	@%%p56 cp.async.cg.shared.global [ %%r220 + 0 ], [ %%rd7 + 0 ], 0x10, %%r215;\n"
"	// end inline asm\n"
"	// begin inline asm\n"
"	@%%p56 cp.async.cg.shared.global [ %%r222 + 0 ], [ %%rd8 + 0 ], 0x10, %%r215;\n"
"	// end inline asm\n"
"	// begin inline asm\n"
"	@%%p56 cp.async.cg.shared.global [ %%r224 + 0 ], [ %%rd9 + 0 ], 0x10, %%r215;\n"
"	// end inline asm\n"
"	// begin inline asm\n"
"	@%%p56 cp.async.cg.shared.global [ %%r226 + 0 ], [ %%rd10 + 0 ], 0x10, %%r215;\n"
"	// end inline asm\n"
"	// begin inline asm\n"
"	@%%p56 cp.async.cg.shared.global [ %%r228 + 0 ], [ %%rd11 + 0 ], 0x10, %%r215;\n"
"	// end inline asm\n"
"	// begin inline asm\n"
"	cp.async.commit_group ;\n"
"	// end inline asm\n"
"	mul.lo.s32 	%%r312, %%r1168, %%r3;\n"
"	shl.b32 	%%r313, %%r1168, 3;\n"
"	add.s32 	%%r314, %%r312, %%r313;\n"
"	add.s32 	%%r315, %%r314, %%r313;\n"
"	add.s32 	%%r316, %%r315, %%r313;\n"
"	add.s32 	%%r317, %%r316, %%r313;\n"
"	add.s32 	%%r318, %%r317, %%r313;\n"
"	add.s32 	%%r319, %%r318, %%r313;\n"
"	add.s32 	%%r320, %%r319, %%r313;\n"
"	add.s32 	%%r321, %%r15, %%r312;\n"
"	add.s32 	%%r322, %%r15, %%r314;\n"
"	add.s32 	%%r323, %%r15, %%r315;\n"
"	add.s32 	%%r324, %%r15, %%r316;\n"
"	add.s32 	%%r325, %%r15, %%r317;\n"
"	add.s32 	%%r326, %%r15, %%r318;\n"
"	add.s32 	%%r327, %%r15, %%r319;\n"
"	add.s32 	%%r328, %%r15, %%r320;\n"
"	mul.wide.s32 	%%rd44, %%r321, 4;\n"
"	add.s64 	%%rd12, %%rd118, %%rd44;\n"
"	mul.wide.s32 	%%rd45, %%r322, 4;\n"
"	add.s64 	%%rd13, %%rd118, %%rd45;\n"
"	mul.wide.s32 	%%rd46, %%r323, 4;\n"
"	add.s64 	%%rd14, %%rd118, %%rd46;\n"
"	mul.wide.s32 	%%rd47, %%r324, 4;\n"
"	add.s64 	%%rd15, %%rd118, %%rd47;\n"
"	mul.wide.s32 	%%rd48, %%r325, 4;\n"
"	add.s64 	%%rd16, %%rd118, %%rd48;\n"
"	mul.wide.s32 	%%rd49, %%r326, 4;\n"
"	add.s64 	%%rd17, %%rd118, %%rd49;\n"
"	mul.wide.s32 	%%rd50, %%r327, 4;\n"
"	add.s64 	%%rd18, %%rd118, %%rd50;\n"
"	mul.wide.s32 	%%rd51, %%r328, 4;\n"
"	add.s64 	%%rd19, %%rd118, %%rd51;\n"
"	shr.u32 	%%r329, %%r2, 1;\n"
"	and.b32  	%%r330, %%r329, 24;\n"
"	xor.b32  	%%r331, %%r330, %%r5;\n"
"	or.b32  	%%r20, %%r331, %%r309;\n"
"	shl.b32 	%%r332, %%r20, 2;\n"
"	add.s32 	%%r333, %%r311, %%r332;\n"
"	add.s32 	%%r230, %%r333, 32768;\n"
"	add.s32 	%%r232, %%r333, 34816;\n"
"	add.s32 	%%r234, %%r333, 36864;\n"
"	add.s32 	%%r236, %%r333, 38912;\n"
"	add.s32 	%%r238, %%r333, 40960;\n"
"	add.s32 	%%r240, %%r333, 43008;\n"
"	add.s32 	%%r242, %%r333, 45056;\n"
"	add.s32 	%%r244, %%r333, 47104;\n"
"	// begin inline asm\n"
"	@%%p56 cp.async.cg.shared.global [ %%r230 + 0 ], [ %%rd12 + 0 ], 0x10, %%r215;\n"
"	// end inline asm\n"
"	// begin inline asm\n"
"	@%%p56 cp.async.cg.shared.global [ %%r232 + 0 ], [ %%rd13 + 0 ], 0x10, %%r215;\n"
"	// end inline asm\n"
"	// begin inline asm\n"
"	@%%p56 cp.async.cg.shared.global [ %%r234 + 0 ], [ %%rd14 + 0 ], 0x10, %%r215;\n"
"	// end inline asm\n"
"	// begin inline asm\n"
"	@%%p56 cp.async.cg.shared.global [ %%r236 + 0 ], [ %%rd15 + 0 ], 0x10, %%r215;\n"
"	// end inline asm\n"
"	// begin inline asm\n"
"	@%%p56 cp.async.cg.shared.global [ %%r238 + 0 ], [ %%rd16 + 0 ], 0x10, %%r215;\n"
"	// end inline asm\n"
"	// begin inline asm\n"
"	@%%p56 cp.async.cg.shared.global [ %%r240 + 0 ], [ %%rd17 + 0 ], 0x10, %%r215;\n"
"	// end inline asm\n"
"	// begin inline asm\n"
"	@%%p56 cp.async.cg.shared.global [ %%r242 + 0 ], [ %%rd18 + 0 ], 0x10, %%r215;\n"
"	// end inline asm\n"
"	// begin inline asm\n"
"	@%%p56 cp.async.cg.shared.global [ %%r244 + 0 ], [ %%rd19 + 0 ], 0x10, %%r215;\n"
"	// end inline asm\n"
"	// begin inline asm\n"
"	cp.async.commit_group ;\n"
"	// end inline asm\n"
"	setp.gt.s32 	%%p35, %%r1170, 64;\n"
"	or.b32  	%%r334, %%r5, 64;\n"
"	or.b32  	%%r335, %%r3, 64;\n"
"	add.s32 	%%r336, %%r289, %%r334;\n"
"	add.s32 	%%r337, %%r290, %%r334;\n"
"	add.s32 	%%r338, %%r291, %%r334;\n"
"	add.s32 	%%r339, %%r292, %%r334;\n"
"	add.s32 	%%r340, %%r293, %%r334;\n"
"	add.s32 	%%r341, %%r294, %%r334;\n"
"	add.s32 	%%r342, %%r295, %%r334;\n"
"	add.s32 	%%r343, %%r296, %%r334;\n"
"	mul.wide.s32 	%%rd52, %%r336, 4;\n"
"	add.s64 	%%rd20, %%rd119, %%rd52;\n"
"	mul.wide.s32 	%%rd53, %%r337, 4;\n"
"	add.s64 	%%rd21, %%rd119, %%rd53;\n"
"	mul.wide.s32 	%%rd54, %%r338, 4;\n"
"	add.s64 	%%rd22, %%rd119, %%rd54;\n"
"	mul.wide.s32 	%%rd55, %%r339, 4;\n"
"	add.s64 	%%rd23, %%rd119, %%rd55;\n"
"	mul.wide.s32 	%%rd56, %%r340, 4;\n"
"	add.s64 	%%rd24, %%rd119, %%rd56;\n"
"	mul.wide.s32 	%%rd57, %%r341, 4;\n"
"	add.s64 	%%rd25, %%rd119, %%rd57;\n"
"	mul.wide.s32 	%%rd58, %%r342, 4;\n"
"	add.s64 	%%rd26, %%rd119, %%rd58;\n"
"	mul.wide.s32 	%%rd59, %%r343, 4;\n"
"	add.s64 	%%rd27, %%rd119, %%rd59;\n"
"	bar.sync 	0;\n"
"	add.s32 	%%r246, %%r214, 16384;\n"
"	add.s32 	%%r248, %%r214, 18432;\n"
"	add.s32 	%%r250, %%r214, 20480;\n"
"	add.s32 	%%r252, %%r214, 22528;\n"
"	add.s32 	%%r254, %%r214, 24576;\n"
"	add.s32 	%%r256, %%r214, 26624;\n"
"	add.s32 	%%r258, %%r214, 28672;\n"
"	add.s32 	%%r260, %%r214, 30720;\n"
"	selp.b32 	%%r247, 16, 0, %%p35;\n"
"	// begin inline asm\n"
"	@%%p56 cp.async.cg.shared.global [ %%r246 + 0 ], [ %%rd20 + 0 ], 0x10, %%r247;\n"
"	// end inline asm\n"
"	// begin inline asm\n"
"	@%%p56 cp.async.cg.shared.global [ %%r248 + 0 ], [ %%rd21 + 0 ], 0x10, %%r247;\n"
"	// end inline asm\n"
"	// begin inline asm\n"
"	@%%p56 cp.async.cg.shared.global [ %%r250 + 0 ], [ %%rd22 + 0 ], 0x10, %%r247;\n"
"	// end inline asm\n"
"	// begin inline asm\n"
"	@%%p56 cp.async.cg.shared.global [ %%r252 + 0 ], [ %%rd23 + 0 ], 0x10, %%r247;\n"
"	// end inline asm\n"
"	// begin inline asm\n"
"	@%%p56 cp.async.cg.shared.global [ %%r254 + 0 ], [ %%rd24 + 0 ], 0x10, %%r247;\n"
"	// end inline asm\n"
"	// begin inline asm\n"
"	@%%p56 cp.async.cg.shared.global [ %%r256 + 0 ], [ %%rd25 + 0 ], 0x10, %%r247;\n"
"	// end inline asm\n"
"	// begin inline asm\n"
"	@%%p56 cp.async.cg.shared.global [ %%r258 + 0 ], [ %%rd26 + 0 ], 0x10, %%r247;\n"
"	// end inline asm\n"
"	// begin inline asm\n"
"	@%%p56 cp.async.cg.shared.global [ %%r260 + 0 ], [ %%rd27 + 0 ], 0x10, %%r247;\n"
"	// end inline asm\n"
"	// begin inline asm\n"
"	cp.async.commit_group ;\n"
"	// end inline asm\n"
"	mul.lo.s32 	%%r344, %%r1168, %%r335;\n"
"	add.s32 	%%r345, %%r344, %%r313;\n"
"	add.s32 	%%r346, %%r345, %%r313;\n"
"	add.s32 	%%r347, %%r346, %%r313;\n"
"	add.s32 	%%r348, %%r347, %%r313;\n"
"	add.s32 	%%r349, %%r348, %%r313;\n"
"	add.s32 	%%r350, %%r349, %%r313;\n"
"	add.s32 	%%r351, %%r350, %%r313;\n"
"	add.s32 	%%r352, %%r15, %%r344;\n"
"	add.s32 	%%r353, %%r15, %%r345;\n"
"	add.s32 	%%r354, %%r15, %%r346;\n"
"	add.s32 	%%r355, %%r15, %%r347;\n"
"	add.s32 	%%r356, %%r15, %%r348;\n"
"	add.s32 	%%r357, %%r15, %%r349;\n"
"	add.s32 	%%r358, %%r15, %%r350;\n"
"	add.s32 	%%r359, %%r15, %%r351;\n"
"	mul.wide.s32 	%%rd60, %%r352, 4;\n"
"	add.s64 	%%rd28, %%rd118, %%rd60;\n"
"	mul.wide.s32 	%%rd61, %%r353, 4;\n"
"	add.s64 	%%rd29, %%rd118, %%rd61;\n"
"	mul.wide.s32 	%%rd62, %%r354, 4;\n"
"	add.s64 	%%rd30, %%rd118, %%rd62;\n"
"	mul.wide.s32 	%%rd63, %%r355, 4;\n"
"	add.s64 	%%rd31, %%rd118, %%rd63;\n"
"	mul.wide.s32 	%%rd64, %%r356, 4;\n"
"	add.s64 	%%rd32, %%rd118, %%rd64;\n"
"	mul.wide.s32 	%%rd65, %%r357, 4;\n"
"	add.s64 	%%rd33, %%rd118, %%rd65;\n"
"	mul.wide.s32 	%%rd66, %%r358, 4;\n"
"	add.s64 	%%rd34, %%rd118, %%rd66;\n"
"	mul.wide.s32 	%%rd67, %%r359, 4;\n"
"	add.s64 	%%rd35, %%rd118, %%rd67;\n"
"	add.s32 	%%r262, %%r333, 49152;\n"
"	add.s32 	%%r264, %%r333, 51200;\n"
"	add.s32 	%%r266, %%r333, 53248;\n"
"	add.s32 	%%r268, %%r333, 55296;\n"
"	add.s32 	%%r270, %%r333, 57344;\n"
"	add.s32 	%%r272, %%r333, 59392;\n"
"	add.s32 	%%r274, %%r333, 61440;\n"
"	add.s32 	%%r276, %%r333, 63488;\n"
"	// begin inline asm\n"
"	@%%p56 cp.async.cg.shared.global [ %%r262 + 0 ], [ %%rd28 + 0 ], 0x10, %%r247;\n"
"	// end inline asm\n"
"	// begin inline asm\n"
"	@%%p56 cp.async.cg.shared.global [ %%r264 + 0 ], [ %%rd29 + 0 ], 0x10, %%r247;\n"
"	// end inline asm\n"
"	// begin inline asm\n"
"	@%%p56 cp.async.cg.shared.global [ %%r266 + 0 ], [ %%rd30 + 0 ], 0x10, %%r247;\n"
"	// end inline asm\n"
"	// begin inline asm\n"
"	@%%p56 cp.async.cg.shared.global [ %%r268 + 0 ], [ %%rd31 + 0 ], 0x10, %%r247;\n"
"	// end inline asm\n"
"	// begin inline asm\n"
"	@%%p56 cp.async.cg.shared.global [ %%r270 + 0 ], [ %%rd32 + 0 ], 0x10, %%r247;\n"
"	// end inline asm\n"
"	// begin inline asm\n"
"	@%%p56 cp.async.cg.shared.global [ %%r272 + 0 ], [ %%rd33 + 0 ], 0x10, %%r247;\n"
"	// end inline asm\n"
"	// begin inline asm\n"
"	@%%p56 cp.async.cg.shared.global [ %%r274 + 0 ], [ %%rd34 + 0 ], 0x10, %%r247;\n"
"	// end inline asm\n"
"	// begin inline asm\n"
"	@%%p56 cp.async.cg.shared.global [ %%r276 + 0 ], [ %%rd35 + 0 ], 0x10, %%r247;\n"
"	// end inline asm\n"
"	// begin inline asm\n"
"	cp.async.commit_group ;\n"
"	// end inline asm\n"
"	// begin inline asm\n"
"	cp.async.wait_group 0x2;\n"
"	// end inline asm\n"
"	bar.sync 	0;\n"
"	and.b32  	%%r21, %%r2, 7;\n"
"	bfe.u32 	%%r22, %%r2, 4, 1;\n"
"	and.b32  	%%r360, %%r305, 16;\n"
"	and.b32  	%%r361, %%r2, 15;\n"
"	or.b32  	%%r362, %%r361, %%r360;\n"
"	xor.b32  	%%r363, %%r22, %%r21;\n"
"	shl.b32 	%%r364, %%r363, 2;\n"
"	shl.b32 	%%r23, %%r362, 6;\n"
"	or.b32  	%%r24, %%r23, %%r364;\n"
"	shl.b32 	%%r365, %%r24, 2;\n"
"	add.s32 	%%r282, %%r311, %%r365;\n"
"	// begin inline asm\n"
"}\n"
"\n"
    : : "l"(A), "l"(B), "l"(C), "r"(M), "r"(N), "r"(K), "r"(lda), "r"(lda1), "r"(ldb), "r"(ldb1), "r"(ldc), "r"(ldc1), "r"(BLOCK_M), "r"(BLOCK_N), "r"(BLOCK_K));
}

int main() {
    int M = 128, K = 128, N = 128;

    // Host matrices
    float *h_A = new float[M*K];
    float *h_B = new float[K*N];
    float *h_C = new float[M*N];

    // Initialize host matrices
    for (int i = 0; i < M*K; i++) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K*N; i++) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // Device matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M*K*sizeof(float));
    cudaMalloc(&d_B, K*N*sizeof(float));
    cudaMalloc(&d_C, M*N*sizeof(float));

    cudaMemcpy(d_A, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K*N*sizeof(float), cudaMemcpyHostToDevice);

    // Grid and block sizes
    dim3 grid((M + BLOCK - 1) / BLOCK, (N + BLOCK - 1) / BLOCK);
    dim3 block(1, 1); // adjust if kernel uses multiple threads

    // Launch kernel
    matmul_kernel<<<grid, block>>>(
        d_A, d_B, d_C,
        M, N, K,
        K, 1,  // A.stride(0), A.stride(1)
        N, 1,  // B.stride(0), B.stride(1)
        N, 1,  // C.stride(0), C.stride(1)
        BLOCK, BLOCK, BLOCK
    );
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    // Optional: verify
    float max_error = 0.0f;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += h_A[i*K + k] * h_B[k*N + j];
            max_error = fmax(max_error, fabs(sum - h_C[i*N + j]));
            std::cout<<sum<<" "<<h_C[i*N+j]<<std::endl;
        }
    std::cout << "Max error: " << max_error << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}


// int main() {
//     int N = 1024;
//     size_t bytes = N * sizeof(float);

//     float *hA = new float[N];
//     float *hB = new float[N];
//     float *hC = new float[N];

//     // Initialize host arrays
//     for (int i = 0; i < N; i++) {
//         hA[i] = i;
//         hB[i] = 2 * i;
//     }

//     // Device pointers
//     float *A, *B, *C;
//     cudaMalloc(&A, bytes);
//     cudaMalloc(&B, bytes);
//     cudaMalloc(&C, bytes);

//     // Copy A and B to device
//     cudaMemcpy(A, hA, bytes, cudaMemcpyHostToDevice);
//     cudaMemcpy(B, hB, bytes, cudaMemcpyHostToDevice);

//     // Launch kernel
//     add<<<1, 128>>>(A, B, C, N, 1);
//     gpuErrchk(cudaGetLastError());
//     cudaDeviceSynchronize();

//     // Copy back results
//     cudaMemcpy(hC, C, bytes, cudaMemcpyDeviceToHost);

//     // Print first 10 values
//     printf("First 10 results:\n");
//     for (int i = 0; i < 10; i++) {
//         printf("C[%d] = %f\n", i, hC[i]);
//     }

//     // Cleanup
//     delete[] hA;
//     delete[] hB;
//     delete[] hC;

//     cudaFree(A);
//     cudaFree(B);
//     cudaFree(C);

//     return 0;
// }