"""
Microkernel filenames lists for avx512skx.

Auto-generated file. Do not edit!
  Generator: tools/update-microkernels.py
"""

ALL_AVX512SKX_MICROKERNEL_SRCS = [
    "src/f16-f32-vcvt/gen/f16-f32-vcvt-avx512skx-u16.c",
    "src/f16-f32-vcvt/gen/f16-f32-vcvt-avx512skx-u32.c",
    "src/f16-f32acc-rdsum/gen/f16-f32acc-rdsum-7p7x-avx512skx-c16.c",
    "src/f16-f32acc-rdsum/gen/f16-f32acc-rdsum-7p7x-avx512skx-c32.c",
    "src/f16-f32acc-rdsum/gen/f16-f32acc-rdsum-7p7x-avx512skx-c64.c",
    "src/f16-f32acc-rdsum/gen/f16-f32acc-rdsum-7p7x-avx512skx-c128.c",
    "src/f16-f32acc-rsum/gen/f16-f32acc-rsum-avx512skx-u16.c",
    "src/f16-f32acc-rsum/gen/f16-f32acc-rsum-avx512skx-u32-acc2.c",
    "src/f16-f32acc-rsum/gen/f16-f32acc-rsum-avx512skx-u48-acc3.c",
    "src/f16-f32acc-rsum/gen/f16-f32acc-rsum-avx512skx-u64-acc2.c",
    "src/f16-f32acc-rsum/gen/f16-f32acc-rsum-avx512skx-u64-acc4.c",
    "src/f16-f32acc-rsum/gen/f16-f32acc-rsum-avx512skx-u128-acc4.c",
    "src/f16-rminmax/gen/f16-rmax-avx512skx-u16.c",
    "src/f16-rminmax/gen/f16-rmax-avx512skx-u32-acc2.c",
    "src/f16-rminmax/gen/f16-rmax-avx512skx-u48-acc3.c",
    "src/f16-rminmax/gen/f16-rmax-avx512skx-u64-acc2.c",
    "src/f16-rminmax/gen/f16-rmax-avx512skx-u64-acc4.c",
    "src/f16-rminmax/gen/f16-rmin-avx512skx-u16.c",
    "src/f16-rminmax/gen/f16-rmin-avx512skx-u32-acc2.c",
    "src/f16-rminmax/gen/f16-rmin-avx512skx-u48-acc3.c",
    "src/f16-rminmax/gen/f16-rmin-avx512skx-u64-acc2.c",
    "src/f16-rminmax/gen/f16-rmin-avx512skx-u64-acc4.c",
    "src/f16-rminmax/gen/f16-rminmax-avx512skx-u16.c",
    "src/f16-rminmax/gen/f16-rminmax-avx512skx-u32-acc2.c",
    "src/f16-rminmax/gen/f16-rminmax-avx512skx-u48-acc3.c",
    "src/f16-rminmax/gen/f16-rminmax-avx512skx-u64-acc2.c",
    "src/f16-rminmax/gen/f16-rminmax-avx512skx-u64-acc4.c",
    "src/f16-vsqrt/gen/f16-vsqrt-avx512skx-sqrt-u16.c",
    "src/f16-vsqrt/gen/f16-vsqrt-avx512skx-sqrt-u32.c",
    "src/f16-vsqrt/gen/f16-vsqrt-avx512skx-sqrt-u64.c",
    "src/f32-f16-vcvt/gen/f32-f16-vcvt-avx512skx-u16.c",
    "src/f32-f16-vcvt/gen/f32-f16-vcvt-avx512skx-u32.c",
    "src/f32-qc8w-gemm/gen/f32-qc4w-gemm-1x32-minmax-avx512skx-broadcast.c",
    "src/f32-qc8w-gemm/gen/f32-qc4w-gemm-2x32-minmax-avx512skx-broadcast.c",
    "src/f32-qc8w-gemm/gen/f32-qc4w-gemm-3x32-minmax-avx512skx-broadcast.c",
    "src/f32-qc8w-gemm/gen/f32-qc4w-gemm-4x32-minmax-avx512skx-broadcast.c",
    "src/f32-qc8w-gemm/gen/f32-qc4w-gemm-5x32-minmax-avx512skx-broadcast.c",
    "src/f32-qc8w-gemm/gen/f32-qc4w-gemm-6x32-minmax-avx512skx-broadcast.c",
    "src/f32-qc8w-gemm/gen/f32-qc4w-gemm-7x32-minmax-avx512skx-broadcast.c",
    "src/f32-qc8w-gemm/gen/f32-qc4w-gemm-8x32-minmax-avx512skx-broadcast.c",
    "src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x16-minmax-avx512skx-broadcast.c",
    "src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x32-minmax-avx512skx-broadcast.c",
    "src/f32-qc8w-gemm/gen/f32-qc8w-gemm-2x16-minmax-avx512skx-broadcast.c",
    "src/f32-qc8w-gemm/gen/f32-qc8w-gemm-2x32-minmax-avx512skx-broadcast.c",
    "src/f32-qc8w-gemm/gen/f32-qc8w-gemm-3x16-minmax-avx512skx-broadcast.c",
    "src/f32-qc8w-gemm/gen/f32-qc8w-gemm-3x32-minmax-avx512skx-broadcast.c",
    "src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x16-minmax-avx512skx-broadcast.c",
    "src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x32-minmax-avx512skx-broadcast.c",
    "src/f32-qc8w-gemm/gen/f32-qc8w-gemm-5x16-minmax-avx512skx-broadcast.c",
    "src/f32-qc8w-gemm/gen/f32-qc8w-gemm-5x32-minmax-avx512skx-broadcast.c",
    "src/f32-qc8w-gemm/gen/f32-qc8w-gemm-6x16-minmax-avx512skx-broadcast.c",
    "src/f32-qc8w-gemm/gen/f32-qc8w-gemm-6x32-minmax-avx512skx-broadcast.c",
    "src/f32-qc8w-gemm/gen/f32-qc8w-gemm-7x16-minmax-avx512skx-broadcast.c",
    "src/f32-qc8w-gemm/gen/f32-qc8w-gemm-7x32-minmax-avx512skx-broadcast.c",
    "src/f32-qc8w-gemm/gen/f32-qc8w-gemm-8x16-minmax-avx512skx-broadcast.c",
    "src/f32-qc8w-gemm/gen/f32-qc8w-gemm-8x32-minmax-avx512skx-broadcast.c",
    "src/f32-qs8-vcvt/gen/f32-qs8-vcvt-avx512skx-u32.c",
    "src/f32-qs8-vcvt/gen/f32-qs8-vcvt-avx512skx-u64.c",
    "src/f32-qs8-vcvt/gen/f32-qs8-vcvt-avx512skx-u96.c",
    "src/f32-qs8-vcvt/gen/f32-qs8-vcvt-avx512skx-u128.c",
    "src/f32-qu8-vcvt/gen/f32-qu8-vcvt-avx512skx-u32.c",
    "src/f32-qu8-vcvt/gen/f32-qu8-vcvt-avx512skx-u64.c",
    "src/f32-qu8-vcvt/gen/f32-qu8-vcvt-avx512skx-u96.c",
    "src/f32-qu8-vcvt/gen/f32-qu8-vcvt-avx512skx-u128.c",
    "src/math/gen/f32-tanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-div.c",
    "src/math/gen/f32-tanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj.c",
    "src/math/gen/f32-tanh-avx512skx-expm1minus-rr1-lut8-p4h3ps-gather-div.c",
    "src/math/gen/f32-tanh-avx512skx-expm1minus-rr1-lut8-p4h3ps-gather-nr1.c",
    "src/math/gen/f32-tanh-avx512skx-expm1minus-rr1-lut8-p4h3ps-gather-nr1adj.c",
    "src/math/gen/f32-tanh-avx512skx-expm1minus-rr1-lut8-p4h3ps-perm-div.c",
    "src/math/gen/f32-tanh-avx512skx-expm1minus-rr1-lut8-p4h3ps-perm-nr1.c",
    "src/math/gen/f32-tanh-avx512skx-expm1minus-rr1-lut8-p4h3ps-perm-nr1adj.c",
    "src/math/gen/f32-tanh-avx512skx-expm1minus-rr1-p6h5ts-div.c",
    "src/math/gen/f32-tanh-avx512skx-expm1minus-rr1-p6h5ts-nr1.c",
    "src/math/gen/f32-tanh-avx512skx-expm1minus-rr1-p6h5ts-nr1adj.c",
    "src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-1x8c8-minmax-avx512skx.c",
    "src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-5x8c8-minmax-avx512skx.c",
    "src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-7x8c8-minmax-avx512skx.c",
    "src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-8x8c8-minmax-avx512skx.c",
    "src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x8c8-minmax-avx512skx.c",
    "src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-5x8c8-minmax-avx512skx.c",
    "src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-7x8c8-minmax-avx512skx.c",
    "src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-8x8c8-minmax-avx512skx.c",
    "src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-1x8c8-minmax-avx512skx.c",
    "src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-5x8c8-minmax-avx512skx.c",
    "src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-7x8c8-minmax-avx512skx.c",
    "src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-8x8c8-minmax-avx512skx.c",
    "src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-1x8c8-minmax-avx512skx.c",
    "src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-2x8c8-minmax-avx512skx.c",
    "src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-3x8c8-minmax-avx512skx.c",
    "src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-4x8c8-minmax-avx512skx.c",
    "src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-5x8c8-minmax-avx512skx.c",
    "src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-6x8c8-minmax-avx512skx.c",
    "src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-7x8c8-minmax-avx512skx.c",
    "src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-8x8c8-minmax-avx512skx.c",
    "src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-1x8c8-minmax-avx512skx.c",
    "src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-5x8c8-minmax-avx512skx.c",
    "src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-7x8c8-minmax-avx512skx.c",
    "src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-8x8c8-minmax-avx512skx.c",
    "src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x8c8-minmax-avx512skx.c",
    "src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16c8-minmax-avx512skx-prfm.c",
    "src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16c8-minmax-avx512skx.c",
    "src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x8c8-minmax-avx512skx.c",
    "src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x16c8-minmax-avx512skx-prfm.c",
    "src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x16c8-minmax-avx512skx.c",
    "src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x8c8-minmax-avx512skx.c",
    "src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x16c8-minmax-avx512skx-prfm.c",
    "src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x16c8-minmax-avx512skx.c",
    "src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x8c8-minmax-avx512skx.c",
    "src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x16c8-minmax-avx512skx-prfm.c",
    "src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x16c8-minmax-avx512skx.c",
    "src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c8-minmax-avx512skx.c",
    "src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16c8-minmax-avx512skx-prfm.c",
    "src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16c8-minmax-avx512skx.c",
    "src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x8c8-minmax-avx512skx.c",
    "src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x16c8-minmax-avx512skx-prfm.c",
    "src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x16c8-minmax-avx512skx.c",
    "src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x8c8-minmax-avx512skx.c",
    "src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x16c8-minmax-avx512skx-prfm.c",
    "src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x16c8-minmax-avx512skx.c",
    "src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x8c8-minmax-avx512skx.c",
    "src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x16c8-minmax-avx512skx-prfm.c",
    "src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x16c8-minmax-avx512skx.c",
    "src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x8c8-minmax-avx512skx.c",
    "src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x16c8-minmax-avx512skx-prfm.c",
    "src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x16c8-minmax-avx512skx.c",
    "src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x8c8-minmax-avx512skx.c",
    "src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-3x8c8-minmax-avx512skx.c",
    "src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x8c8-minmax-avx512skx.c",
    "src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-5x8c8-minmax-avx512skx.c",
    "src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-5x16c8-minmax-avx512skx-prfm.c",
    "src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-5x16c8-minmax-avx512skx.c",
    "src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-6x8c8-minmax-avx512skx.c",
    "src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-7x8c8-minmax-avx512skx.c",
    "src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-7x16c8-minmax-avx512skx-prfm.c",
    "src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-7x16c8-minmax-avx512skx.c",
    "src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-8x8c8-minmax-avx512skx.c",
    "src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-8x16c8-minmax-avx512skx-prfm.c",
    "src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-8x16c8-minmax-avx512skx.c",
    "src/qs8-dwconv/gen/qs8-dwconv-5f5m5l16c16s1r-minmax-fp32-avx512skx-mul32.c",
    "src/qs8-dwconv/gen/qs8-dwconv-5f5m5l32c16s1r-minmax-fp32-avx512skx-mul32.c",
    "src/qs8-dwconv/gen/qs8-dwconv-6f6m7l16c16s1r-minmax-fp32-avx512skx-mul32.c",
    "src/qs8-dwconv/gen/qs8-dwconv-6f6m7l32c16s1r-minmax-fp32-avx512skx-mul32.c",
    "src/qs8-dwconv/gen/qs8-dwconv-8f8m9l16c16s1r-minmax-fp32-avx512skx-mul32.c",
    "src/qs8-dwconv/gen/qs8-dwconv-8f8m9l32c16s1r-minmax-fp32-avx512skx-mul32.c",
    "src/qs8-dwconv/gen/qs8-dwconv-9p16c-minmax-fp32-avx512skx-mul32.c",
    "src/qs8-dwconv/gen/qs8-dwconv-9p32c-minmax-fp32-avx512skx-mul32.c",
    "src/qs8-dwconv/gen/qs8-dwconv-25p16c-minmax-fp32-avx512skx-mul32.c",
    "src/qs8-dwconv/gen/qs8-dwconv-25p32c-minmax-fp32-avx512skx-mul32.c",
    "src/qs8-f32-vcvt/gen/qs8-f32-vcvt-avx512skx-u16.c",
    "src/qs8-f32-vcvt/gen/qs8-f32-vcvt-avx512skx-u32.c",
    "src/qs8-f32-vcvt/gen/qs8-f32-vcvt-avx512skx-u48.c",
    "src/qs8-f32-vcvt/gen/qs8-f32-vcvt-avx512skx-u64.c",
    "src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-3p32c-minmax-fp32-avx512skx-mul32.c",
    "src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l16c16s1r-minmax-fp32-avx512skx-mul32.c",
    "src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l32c16s1r-minmax-fp32-avx512skx-mul32.c",
    "src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l16c16s1r-minmax-fp32-avx512skx-mul32.c",
    "src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l32c16s1r-minmax-fp32-avx512skx-mul32.c",
    "src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l16c16s1r-minmax-fp32-avx512skx-mul32.c",
    "src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l32c16s1r-minmax-fp32-avx512skx-mul32.c",
    "src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p16c-minmax-fp32-avx512skx-mul32.c",
    "src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p32c-minmax-fp32-avx512skx-mul32.c",
    "src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p16c-minmax-fp32-avx512skx-mul32.c",
    "src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p32c-minmax-fp32-avx512skx-mul32.c",
    "src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-avx512skx.c",
    "src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c8-minmax-fp32-avx512skx-prfm.c",
    "src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c8-minmax-fp32-avx512skx.c",
    "src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-5x16c8-minmax-fp32-avx512skx-prfm.c",
    "src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-5x16c8-minmax-fp32-avx512skx.c",
    "src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x16c8-minmax-fp32-avx512skx-prfm.c",
    "src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x16c8-minmax-fp32-avx512skx.c",
    "src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x16c8-minmax-fp32-avx512skx-prfm.c",
    "src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x16c8-minmax-fp32-avx512skx.c",
    "src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c8-minmax-fp32-avx512skx.c",
    "src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x16c8-minmax-fp32-avx512skx-prfm.c",
    "src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x16c8-minmax-fp32-avx512skx.c",
    "src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-5x8c8-minmax-fp32-avx512skx.c",
    "src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-5x16c8-minmax-fp32-avx512skx-prfm.c",
    "src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-5x16c8-minmax-fp32-avx512skx.c",
    "src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-6x8c8-minmax-fp32-avx512skx.c",
    "src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-7x8c8-minmax-fp32-avx512skx.c",
    "src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-7x16c8-minmax-fp32-avx512skx-prfm.c",
    "src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-7x16c8-minmax-fp32-avx512skx.c",
    "src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-8x8c8-minmax-fp32-avx512skx.c",
    "src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-8x16c8-minmax-fp32-avx512skx-prfm.c",
    "src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-8x16c8-minmax-fp32-avx512skx.c",
    "src/qs8-rdsum/gen/qs8-rdsum-7p7x-minmax-fp32-avx512skx-c64.c",
    "src/qs8-rdsum/gen/qs8-rdsum-7p7x-minmax-fp32-avx512skx-c128.c",
    "src/qs8-rsum/gen/qs8-rsum-minmax-fp32-avx512skx-u64.c",
    "src/qs8-rsum/gen/qs8-rsum-minmax-fp32-avx512skx-u128-acc2.c",
    "src/qs8-rsum/gen/qs8-rsum-minmax-fp32-avx512skx-u128.c",
    "src/qs8-rsum/gen/qs8-rsum-minmax-fp32-avx512skx-u256-acc2.c",
    "src/qs8-rsum/gen/qs8-rsum-minmax-fp32-avx512skx-u256-acc4.c",
    "src/qs8-rsum/gen/qs8-rsum-minmax-fp32-avx512skx-u256.c",
    "src/qs8-vadd/gen/qs8-vadd-minmax-avx512skx-mul32-ld128-u16.c",
    "src/qs8-vadd/gen/qs8-vadd-minmax-avx512skx-mul32-ld128-u32.c",
    "src/qs8-vaddc/gen/qs8-vaddc-minmax-avx512skx-mul32-ld128-u16.c",
    "src/qs8-vaddc/gen/qs8-vaddc-minmax-avx512skx-mul32-ld128-u32.c",
    "src/qu8-dwconv/gen/qu8-dwconv-5f5m5l16c16s1r-minmax-fp32-avx512skx-mul32.c",
    "src/qu8-dwconv/gen/qu8-dwconv-5f5m5l32c16s1r-minmax-fp32-avx512skx-mul32.c",
    "src/qu8-dwconv/gen/qu8-dwconv-6f6m7l16c16s1r-minmax-fp32-avx512skx-mul32.c",
    "src/qu8-dwconv/gen/qu8-dwconv-6f6m7l32c16s1r-minmax-fp32-avx512skx-mul32.c",
    "src/qu8-dwconv/gen/qu8-dwconv-8f8m9l16c16s1r-minmax-fp32-avx512skx-mul32.c",
    "src/qu8-dwconv/gen/qu8-dwconv-8f8m9l32c16s1r-minmax-fp32-avx512skx-mul32.c",
    "src/qu8-dwconv/gen/qu8-dwconv-9p16c-minmax-fp32-avx512skx-mul32.c",
    "src/qu8-dwconv/gen/qu8-dwconv-9p32c-minmax-fp32-avx512skx-mul32.c",
    "src/qu8-dwconv/gen/qu8-dwconv-25p16c-minmax-fp32-avx512skx-mul32.c",
    "src/qu8-dwconv/gen/qu8-dwconv-25p32c-minmax-fp32-avx512skx-mul32.c",
    "src/qu8-f32-vcvt/gen/qu8-f32-vcvt-avx512skx-u16.c",
    "src/qu8-f32-vcvt/gen/qu8-f32-vcvt-avx512skx-u32.c",
    "src/qu8-f32-vcvt/gen/qu8-f32-vcvt-avx512skx-u48.c",
    "src/qu8-f32-vcvt/gen/qu8-f32-vcvt-avx512skx-u64.c",
    "src/qu8-gemm/gen/qu8-gemm-1x8c8-minmax-fp32-avx512skx.c",
    "src/qu8-gemm/gen/qu8-gemm-1x16c8-minmax-fp32-avx512skx-prfm.c",
    "src/qu8-gemm/gen/qu8-gemm-1x16c8-minmax-fp32-avx512skx.c",
    "src/qu8-gemm/gen/qu8-gemm-5x16c8-minmax-fp32-avx512skx-prfm.c",
    "src/qu8-gemm/gen/qu8-gemm-5x16c8-minmax-fp32-avx512skx.c",
    "src/qu8-gemm/gen/qu8-gemm-7x16c8-minmax-fp32-avx512skx-prfm.c",
    "src/qu8-gemm/gen/qu8-gemm-7x16c8-minmax-fp32-avx512skx.c",
    "src/qu8-gemm/gen/qu8-gemm-8x16c8-minmax-fp32-avx512skx-prfm.c",
    "src/qu8-gemm/gen/qu8-gemm-8x16c8-minmax-fp32-avx512skx.c",
    "src/qu8-igemm/gen/qu8-igemm-1x8c8-minmax-fp32-avx512skx.c",
    "src/qu8-igemm/gen/qu8-igemm-1x16c8-minmax-fp32-avx512skx-prfm.c",
    "src/qu8-igemm/gen/qu8-igemm-1x16c8-minmax-fp32-avx512skx.c",
    "src/qu8-igemm/gen/qu8-igemm-5x8c8-minmax-fp32-avx512skx.c",
    "src/qu8-igemm/gen/qu8-igemm-5x16c8-minmax-fp32-avx512skx-prfm.c",
    "src/qu8-igemm/gen/qu8-igemm-5x16c8-minmax-fp32-avx512skx.c",
    "src/qu8-igemm/gen/qu8-igemm-6x8c8-minmax-fp32-avx512skx.c",
    "src/qu8-igemm/gen/qu8-igemm-7x8c8-minmax-fp32-avx512skx.c",
    "src/qu8-igemm/gen/qu8-igemm-7x16c8-minmax-fp32-avx512skx-prfm.c",
    "src/qu8-igemm/gen/qu8-igemm-7x16c8-minmax-fp32-avx512skx.c",
    "src/qu8-igemm/gen/qu8-igemm-8x8c8-minmax-fp32-avx512skx.c",
    "src/qu8-igemm/gen/qu8-igemm-8x16c8-minmax-fp32-avx512skx-prfm.c",
    "src/qu8-igemm/gen/qu8-igemm-8x16c8-minmax-fp32-avx512skx.c",
    "src/qu8-vadd/gen/qu8-vadd-minmax-avx512skx-mul32-ld128-u16.c",
    "src/qu8-vadd/gen/qu8-vadd-minmax-avx512skx-mul32-ld128-u32.c",
    "src/qu8-vaddc/gen/qu8-vaddc-minmax-avx512skx-mul32-ld128-u16.c",
    "src/qu8-vaddc/gen/qu8-vaddc-minmax-avx512skx-mul32-ld128-u32.c",
    "src/x8-lut/gen/x8-lut-avx512skx-vpshufb-u64.c",
    "src/x8-lut/gen/x8-lut-avx512skx-vpshufb-u128.c",
    "src/x8-lut/gen/x8-lut-avx512skx-vpshufb-u192.c",
    "src/x8-lut/gen/x8-lut-avx512skx-vpshufb-u256.c",
]
