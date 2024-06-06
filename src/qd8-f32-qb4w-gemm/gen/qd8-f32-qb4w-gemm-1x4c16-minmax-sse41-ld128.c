// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/MRx4c8-sse.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <smmintrin.h>

#include <xnnpack/gemm.h>
#include <xnnpack/math.h>
#include <xnnpack/unaligned.h>
#include <stdio.h>

void print128_num_i16(__m128i var)
{
    uint16_t val[8];
    memcpy(val, &var, sizeof(val));
    printf("Numerical: %i %i %i %i %i %i %i %i \n",
           val[0], val[1], val[2], val[3], val[4], val[5],
           val[6], val[7]);
}


void xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x4c16__sse41_ld128(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qb4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  size_t bl = params->sse.blocksize;
  assert(bl <= round_up_po2(kc, 2));
  assert(bl != 0);
  size_t n_blocks = kc / bl;
  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  float* c0 = c;

  const __m128i vmask = _mm_load_si128((const __m128i*) params->sse.mask);  // 0xF0
  do {
    const __m128 vksum = _mm_load_ps((const float*) w);
    __m128i vinput_zero_point0 = _mm_cvtsi32_si128(*((const int*) &quantization_params[0].zero_point));
    vinput_zero_point0 = _mm_shuffle_epi32(vinput_zero_point0, _MM_SHUFFLE(0, 0, 0, 0));

    __m128 vinput_zero_point0_float = _mm_cvtepi32_ps(vinput_zero_point0);
    __m128 vout0x0123 = _mm_mul_ps(vksum, vinput_zero_point0_float);
    w = (const int32_t*) w + 4;

    for (size_t nb=0; nb<n_blocks; ++nb){
      __m128i vacc0x0 = _mm_setzero_si128();
      __m128i vacc0x1 = _mm_setzero_si128();
      __m128i vacc0x2 = _mm_setzero_si128();
      __m128i vacc0x3 = _mm_setzero_si128();
      size_t k = bl;

    while (k >= 32 * sizeof(int8_t)) {
      // a[0....15]
      // a[0....7] = a0c0
      // a[8....15] = a0c1
      const __m128i va0c0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0c0 = _mm_cvtepi8_epi16(va0c0); // 8 16 bit ints bottom
      a0 += 8;
      const __m128i va0c1 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0c1 = _mm_cvtepi8_epi16(va0c1); // 8 16 bit ints bottom
      a0 += 8;

      // c01 here means the lower nibbles of the two planes while c23 means the
      // higher nibbles of the first two planes. m128i load of weights gives me
      // the first two planes. the lower/upper nibbles of these two planes is mul_added
      // with the 16 elements from the activation. c01 is used for the first 16
      // elements in the activation while c23 are used for the next 16.
      // b = 0
      const __m128i vb0c0123 = _mm_load_si128((const __m128i*) w);
      const __m128i vbs0c01 = _mm_slli_epi32(vb0c0123, 4);
      const __m128i vb0c01 = _mm_and_si128(vbs0c01, vmask);
      const __m128i vsb0c01 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb0c01);
      const __m128i vxb0c0 = _mm_unpacklo_epi8(vb0c01, vsb0c01);
      const __m128i vxb0c1 = _mm_unpackhi_epi8(vb0c01, vsb0c01);

      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0c0, vxb0c0));
      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0c1, vxb0c1));

      // b = 1
      const __m128i vb1c0123 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
      const __m128i vbs1c01 = _mm_slli_epi32(vb1c0123, 4);
      const __m128i vb1c01 = _mm_and_si128(vbs1c01, vmask);
      const __m128i vsb1c01 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb1c01);
      const __m128i vxb1c0 = _mm_unpacklo_epi8(vb1c01, vsb1c01);
      const __m128i vxb1c1 = _mm_unpackhi_epi8(vb1c01, vsb1c01);

      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0c0, vxb1c0));
      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0c1, vxb1c1));

      // b = 2
      const __m128i vb2c0123 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 32));
      const __m128i vbs2c01 = _mm_slli_epi32(vb2c0123, 4);
      const __m128i vb2c01 = _mm_and_si128(vbs2c01, vmask);
      const __m128i vsb2c01 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb2c01);
      const __m128i vxb2c0 = _mm_unpacklo_epi8(vb2c01, vsb2c01);
      const __m128i vxb2c1 = _mm_unpackhi_epi8(vb2c01, vsb2c01);

      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0c0, vxb2c0));
      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0c1, vxb2c1));

      // b = 3
      const __m128i vb3c0123 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 48));
      const __m128i vbs3c01 = _mm_slli_epi32(vb3c0123, 4);
      const __m128i vb3c01 = _mm_and_si128(vbs3c01, vmask);
      const __m128i vsb3c01 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb3c01);
      const __m128i vxb3c0 = _mm_unpacklo_epi8(vb3c01, vsb3c01);
      const __m128i vxb3c1 = _mm_unpackhi_epi8(vb3c01, vsb3c01);

      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0c0, vxb3c0));
      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0c1, vxb3c1));

      // a[16....31]
      // a[16....23] = a0c2
      // a[24....31] = a0c3
      const __m128i va0c2 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0c2 = _mm_cvtepi8_epi16(va0c2); // 8 16 bit ints bottom
      a0 += 8;
      const __m128i va0c3 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0c3 = _mm_cvtepi8_epi16(va0c3); // 8 16 bit ints bottom
      a0 += 8;

      // b = 0
      const __m128i vb0c23 = _mm_and_si128(vb0c0123, vmask);
      const __m128i vsb0c23 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb0c23);
      const __m128i vxb0c2 = _mm_unpacklo_epi8(vb0c23, vsb0c23);
      const __m128i vxb0c3 = _mm_unpackhi_epi8(vb0c23, vsb0c23);

      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0c2, vxb0c2));
      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0c3, vxb0c3));

      // b = 1
      const __m128i vb1c23 = _mm_and_si128(vb1c0123, vmask);
      const __m128i vsb1c23 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb1c23);
      const __m128i vxb1c2 = _mm_unpacklo_epi8(vb1c23, vsb1c23);
      const __m128i vxb1c3 = _mm_unpackhi_epi8(vb1c23, vsb1c23);

      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0c2, vxb1c2));
      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0c3, vxb1c3));

      // b = 2
      const __m128i vb2c23 = _mm_and_si128(vb2c0123, vmask);
      const __m128i vsb2c23 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb2c23);
      const __m128i vxb2c2 = _mm_unpacklo_epi8(vb2c23, vsb2c23);
      const __m128i vxb2c3 = _mm_unpackhi_epi8(vb2c23, vsb2c23);

      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0c2, vxb2c2));
      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0c3, vxb2c3));

      // b = 3
      const __m128i vb3c23 = _mm_and_si128(vb3c0123, vmask);
      const __m128i vsb3c23 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb3c23);
      const __m128i vxb3c2 = _mm_unpacklo_epi8(vb3c23, vsb3c23);
      const __m128i vxb3c3 = _mm_unpackhi_epi8(vb3c23, vsb3c23);

      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0c2, vxb3c2));
      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0c3, vxb3c3));

      w = (const int8_t*) w + 64;
      k -= 32 * sizeof(int8_t);
    }

    // accumulate float
    const __m128 vfilter_output_scale0123 = _mm_load_ps((const float*) w);
    w = (const float*) w + 4;

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);

    vout0x0123 = _mm_add_ps(vout0x0123, _mm_mul_ps(_mm_cvtepi32_ps(vacc0x0123), vfilter_output_scale0123));
    }
    __m128 one_sixteenth = _mm_set_ps1(1.0f/16);
    vout0x0123 = _mm_mul_ps(vout0x0123, one_sixteenth);

    const __m128 vinput_scale0 = _mm_load1_ps(&quantization_params[0].inv_scale);

    vout0x0123 = _mm_mul_ps(vout0x0123, vinput_scale0);


    const __m128 vbias0123 = _mm_load_ps((const float*) w);
    w = (const float*) w + 4;
    vout0x0123 = _mm_add_ps(vout0x0123, vbias0123);

    const __m128 vmin = _mm_load_ps(params->sse.min);
    vout0x0123 = _mm_max_ps(vout0x0123, vmin);

    const __m128 vmax = _mm_load_ps(params->sse.max);
    vout0x0123 = _mm_min_ps(vout0x0123, vmax);

    if XNN_LIKELY(nc >= 4) {
      _mm_storeu_ps(c0, vout0x0123);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vout0x0123);
        vout0x0123 = _mm_unpackhi_ps(vout0x0123, vout0x0123);
        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vout0x0123);
      }
      nc = 0;
    }
  } while (nc != 0);
}
