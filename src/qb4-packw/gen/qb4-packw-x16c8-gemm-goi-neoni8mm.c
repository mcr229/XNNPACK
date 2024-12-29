// Auto-generated file. Do not edit!
//   Template: src/qb4-packw/kr-neoni8mm.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <arm_neon.h>

#include "xnnpack/packw.h"

// convert a vector from packed nibbles to planar, and accumulate sum
static XNN_INTRINSIC
int8x16_t xnn_packed2planar(
    int32x4_t *vacc,
    const uint8x16_t v,
    const uint8x16_t vmask,
    const uint8x16_t veor_mask,
    const int32x4_t neg_zp,
    const uint8x16_t vones_zero_mask)
{
    const uint8x16_t vl = vshrq_n_u8(v, 4);    // isolate lower int 4
    const uint8x16_t vh = vandq_u8(v, vmask);  // isolate upper int 4
    *vacc = vreinterpretq_s32_u32(vmmlaq_u32(vreinterpretq_u32_s32(*vacc), vones_zero_mask, vl));
    *vacc = vreinterpretq_s32_u32(vmmlaq_u32(vreinterpretq_u32_s32(*vacc), vones_zero_mask, vh));
    *vacc = vaddq_s32(*vacc, neg_zp);
    const uint8x16_t v01 = vzip1q_u8(vh, vl);
    const uint8x16_t v23 = vzip2q_u8(vh, vl);
    const uint8x16_t v02 = vreinterpretq_u8_u64(vuzp1q_u64(vreinterpretq_u64_u8(v01), vreinterpretq_u64_u8(v23)));
    const uint8x16_t v13 = vreinterpretq_u8_u64(vuzp2q_u64(vreinterpretq_u64_u8(v01), vreinterpretq_u64_u8(v23)));
    const uint8x16_t vl13 = vshlq_n_u8(v13, 4);
    const uint8x16_t v0123 = vorrq_u8(v02, vl13);
    return veorq_u8(v0123, veor_mask);
}

void xnn_qb4_packw_gemm_goi_ukernel_x16c8__neoni8mm(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t bl,
  const uint8_t* weights,
  const int32_t* bias,
  const void* scale,
  int8_t* packed_weights,
  size_t extra_bytes_bl,
  size_t extra_bytes_n,
  const void* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 16);
  assert(kr == 8);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);
  assert(extra_bytes_bl == nr * sizeof(uint16_t));
  assert(extra_bytes_n == nr * sizeof(float));
  assert(params != NULL);
  assert(kc % bl == 0);
  size_t num_blocks = kc / bl;
  const int8x16_t vmask = vmovq_n_s8(INT8_C(0x0F));
  const uint8x16_t veor_mask = vmovq_n_u8(UINT8_C(0x88));
  const int32x4_t neg_zp = vmovq_n_s32(-64);
  const uint8x8_t vones = vmov_n_u8(UINT8_C(0x01));
  const uint8x8_t vzeros = vmov_n_u8(0);
  const uint8x16_t vone_zero = vcombine_u8(vones, vzeros);
  const uint8x16_t vzero_one = vcombine_u8(vzeros, vones);


  uint8_t* out = (uint8_t*) packed_weights;
  const int32_t* b = (const int32_t*) bias;
  const float32x4_t vzeropoint = vmovq_n_f32((float) (((const struct xnn_qs8_qc4w_packing_params*) params)->input_zero_point + 0));
  const float32x4_t vrecip_sixteen = vmovq_n_f32(1.0f/ 16.0f);

  do {
    // NC main loop multiple of 16
    const uint8_t* w0 = (const uint8_t*) weights;
    const uint16_t* s = (const uint16_t*) scale;
    size_t n = nc;
    for (;n >= 16; n -= 16) {
        float* packed_k_scaled_sum = (float*) out;
        float32x4_t packed_k_scaled_sums0123 = vdupq_n_f32(0.0f);
        float32x4_t packed_k_scaled_sums4567 = vdupq_n_f32(0.0f);
        float32x4_t packed_k_scaled_sums89AB = vdupq_n_f32(0.0f);
        float32x4_t packed_k_scaled_sumsCDEF = vdupq_n_f32(0.0f);
        out += 16 * sizeof(float);

        // KC/2 bytes is KC Nibbles
        const uint8_t* w1 = w0 + (kc >> 1);
        const uint8_t* w2 = w1 + (kc >> 1);
        const uint8_t* w3 = w2 + (kc >> 1);
        const uint8_t* w4 = w3 + (kc >> 1);
        const uint8_t* w5 = w4 + (kc >> 1);
        const uint8_t* w6 = w5 + (kc >> 1);
        const uint8_t* w7 = w6 + (kc >> 1);
        const uint8_t* w8 = w7 + (kc >> 1);
        const uint8_t* w9 = w8 + (kc >> 1);
        const uint8_t* w10 = w9 + (kc >> 1);
        const uint8_t* w11 = w10 + (kc >> 1);
        const uint8_t* w12 = w11 + (kc >> 1);
        const uint8_t* w13 = w12 + (kc >> 1);
        const uint8_t* w14 = w13 + (kc >> 1);
        const uint8_t* w15 = w14 + (kc >> 1);

        const uint16_t* s0 = s;

        size_t kb = kc;
        // Process k by blocks (bl)
        for (; kb >= bl; kb-=bl) {
            // Initialize KSum as subtracting bl zero points (8)
            int32x4_t ksum0123 = vdupq_n_s32(0);
            int32x4_t ksum4567 = vdupq_n_s32(0);
            int32x4_t ksum89AB = vdupq_n_s32(0);
            int32x4_t ksumCDEF = vdupq_n_s32(0);
            size_t k = bl;

            // KC Main loop multiple of 16x32
            for(; k >= 32; k-=32) {
                uint64x2_t w0x01 = vld1q_u64((uint64_t*) w0); w0 += 16;
                uint64x2_t w1x01 = vld1q_u64((uint64_t*) w1); w1 += 16;
                uint64x2_t w2x01 = vld1q_u64((uint64_t*) w2); w2 += 16;
                uint64x2_t w3x01 = vld1q_u64((uint64_t*) w3); w3 += 16;
                uint64x2_t w4x01 = vld1q_u64((uint64_t*) w4); w4 += 16;
                uint64x2_t w5x01 = vld1q_u64((uint64_t*) w5); w5 += 16;
                uint64x2_t w6x01 = vld1q_u64((uint64_t*) w6); w6 += 16;
                uint64x2_t w7x01 = vld1q_u64((uint64_t*) w7); w7 += 16;
                uint64x2_t w8x01 = vld1q_u64((uint64_t*) w8); w8 += 16;
                uint64x2_t w9x01 = vld1q_u64((uint64_t*) w9); w9 += 16;
                uint64x2_t wAx01 = vld1q_u64((uint64_t*) w10); w10 += 16;
                uint64x2_t wBx01 = vld1q_u64((uint64_t*) w11); w11 += 16;
                uint64x2_t wCx01 = vld1q_u64((uint64_t*) w12); w12 += 16;
                uint64x2_t wDx01 = vld1q_u64((uint64_t*) w13); w13 += 16;
                uint64x2_t wEx01 = vld1q_u64((uint64_t*) w14); w14 += 16;
                uint64x2_t wFx01 = vld1q_u64((uint64_t*) w15); w15 += 16;

                uint64x2_t v01_0 = vzip1q_u64(w0x01, w1x01);
                uint64x2_t v01_1 = vzip2q_u64(w0x01, w1x01);
                uint64x2_t v23_0 = vzip1q_u64(w2x01, w3x01);
                uint64x2_t v23_1 = vzip2q_u64(w2x01, w3x01);
                uint64x2_t v45_0 = vzip1q_u64(w4x01, w5x01);
                uint64x2_t v45_1 = vzip2q_u64(w4x01, w5x01);
                uint64x2_t v67_0 = vzip1q_u64(w6x01, w7x01);
                uint64x2_t v67_1 = vzip2q_u64(w6x01, w7x01);
                uint64x2_t v89_0 = vzip1q_u64(w8x01, w9x01);
                uint64x2_t v89_1 = vzip2q_u64(w8x01, w9x01);
                uint64x2_t vAB_0 = vzip1q_u64(wAx01, wBx01);
                uint64x2_t vAB_1 = vzip2q_u64(wAx01, wBx01);
                uint64x2_t vCD_0 = vzip1q_u64(wCx01, wDx01);
                uint64x2_t vCD_1 = vzip2q_u64(wCx01, wDx01);
                uint64x2_t vEF_0 = vzip1q_u64(wEx01, wFx01);
                uint64x2_t vEF_1 = vzip2q_u64(wEx01, wFx01);

                v01_0 = xnn_packed2planar(&ksum0123, v01_0, vmask, veor_mask, neg_zp, vone_zero);
                v23_0 = xnn_packed2planar(&ksum0123, v23_0, vmask, veor_mask, neg_zp, vzero_one);
                v45_0 = xnn_packed2planar(&ksum4567, v45_0, vmask, veor_mask, neg_zp, vone_zero);
                v67_0 = xnn_packed2planar(&ksum4567, v67_0, vmask, veor_mask, neg_zp, vzero_one);
                v89_0 = xnn_packed2planar(&ksum89AB, v89_0, vmask, veor_mask, neg_zp, vone_zero);
                vAB_0 = xnn_packed2planar(&ksum89AB, vAB_0, vmask, veor_mask, neg_zp, vzero_one);
                vCD_0 = xnn_packed2planar(&ksumCDEF, vCD_0, vmask, veor_mask, neg_zp, vone_zero);
                vEF_0 = xnn_packed2planar(&ksumCDEF, vEF_0, vmask, veor_mask, neg_zp, vzero_one);
                v01_1 = xnn_packed2planar(&ksum0123, v01_1, vmask, veor_mask, neg_zp, vone_zero);
                v23_1 = xnn_packed2planar(&ksum0123, v23_1, vmask, veor_mask, neg_zp, vzero_one);
                v45_1 = xnn_packed2planar(&ksum4567, v45_1, vmask, veor_mask, neg_zp, vone_zero);
                v67_1 = xnn_packed2planar(&ksum4567, v67_1, vmask, veor_mask, neg_zp, vzero_one);
                v89_1 = xnn_packed2planar(&ksum89AB, v89_1, vmask, veor_mask, neg_zp, vone_zero);
                vAB_1 = xnn_packed2planar(&ksum89AB, vAB_1, vmask, veor_mask, neg_zp, vzero_one);
                vCD_1 = xnn_packed2planar(&ksumCDEF, vCD_1, vmask, veor_mask, neg_zp, vone_zero);
                vEF_1 = xnn_packed2planar(&ksumCDEF, vEF_1, vmask, veor_mask, neg_zp, vzero_one);

                vst1q_u8(&out[0], v01_0);
                vst1q_u8(&out[16], v23_0);
                vst1q_u8(&out[32], v45_0);
                vst1q_u8(&out[48], v67_0);
                vst1q_u8(&out[64], v89_0);
                vst1q_u8(&out[80], vAB_0);
                vst1q_u8(&out[96], vCD_0);
                vst1q_u8(&out[112], vEF_0);
                vst1q_u8(&out[128], v01_1);
                vst1q_u8(&out[144], v23_1);
                vst1q_u8(&out[160], v45_1);
                vst1q_u8(&out[176], v67_1);
                vst1q_u8(&out[192], v89_1);
                vst1q_u8(&out[208], vAB_1);
                vst1q_u8(&out[224], vCD_1);
                vst1q_u8(&out[240], vEF_1);

                out += 256;
            }

            float32x4_t f_scales0123 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(s0 + 0), 16));
            float32x4_t f_scales4567 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(s0 + 4), 16));
            float32x4_t f_scales89AB = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(s0 + 8), 16));
            float32x4_t f_scalesCDEF = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(s0 + 12), 16));
            s0 += nc;

            float32x4_t f_ksum0123 = vcvtq_f32_s32(ksum0123);
            f_ksum0123 = vmulq_f32(f_ksum0123, vzeropoint);
            packed_k_scaled_sums0123 = vfmsq_f32(packed_k_scaled_sums0123, f_ksum0123, f_scales0123);
            float32x4_t f_ksum4567 = vcvtq_f32_s32(ksum4567);
            f_ksum4567 = vmulq_f32(f_ksum4567, vzeropoint);
            packed_k_scaled_sums4567 = vfmsq_f32(packed_k_scaled_sums4567, f_ksum4567, f_scales4567);
            float32x4_t f_ksum89AB = vcvtq_f32_s32(ksum89AB);
            f_ksum89AB = vmulq_f32(f_ksum89AB, vzeropoint);
            packed_k_scaled_sums89AB = vfmsq_f32(packed_k_scaled_sums89AB, f_ksum89AB, f_scales89AB);
            float32x4_t f_ksumCDEF = vcvtq_f32_s32(ksumCDEF);
            f_ksumCDEF = vmulq_f32(f_ksumCDEF, vzeropoint);
            packed_k_scaled_sumsCDEF = vfmsq_f32(packed_k_scaled_sumsCDEF, f_ksumCDEF, f_scalesCDEF);

            vst1q_f32(&packed_k_scaled_sum[0], packed_k_scaled_sums0123);
            vst1q_f32(&packed_k_scaled_sum[4], packed_k_scaled_sums4567);
            vst1q_f32(&packed_k_scaled_sum[8], packed_k_scaled_sums89AB);
            vst1q_f32(&packed_k_scaled_sum[12], packed_k_scaled_sumsCDEF);

            f_scales0123 = vmulq_f32(f_scales0123, vrecip_sixteen);
            f_scales4567 = vmulq_f32(f_scales4567, vrecip_sixteen);
            f_scales89AB = vmulq_f32(f_scales89AB, vrecip_sixteen);
            f_scalesCDEF = vmulq_f32(f_scalesCDEF, vrecip_sixteen);

            vst1_u16((uint16_t*)out+0, vshrn_n_s32(vreinterpretq_s32_f32(f_scales0123), 16));
            vst1_u16((uint16_t*)out+4, vshrn_n_s32(vreinterpretq_s32_f32(f_scales4567), 16));
            vst1_u16((uint16_t*)out+8, vshrn_n_s32(vreinterpretq_s32_f32(f_scales89AB), 16));
            vst1_u16((uint16_t*)out+12, vshrn_n_s32(vreinterpretq_s32_f32(f_scalesCDEF), 16));

            out += 16 * sizeof(uint16_t);
        }


        if XNN_LIKELY(b != NULL){
            const int32x4_t b0123 = vld1q_s32(&b[0]);
            vst1q_s32((int32_t*)out + 0, b0123);
            const int32x4_t b4567 = vld1q_s32(&b[4]);
            vst1q_s32((int32_t*)out + 4, b4567);
            const int32x4_t b89AB = vld1q_s32(&b[8]);
            vst1q_s32((int32_t*)out + 8, b89AB);
            const int32x4_t bCDEF = vld1q_s32(&b[12]);
            vst1q_s32((int32_t*)out + 12, bCDEF);
            b += 16;
        } else {
            vst1q_s32((int32_t*)out + 0, vdupq_n_s32(0));
            vst1q_s32((int32_t*)out + 4, vdupq_n_s32(0));
            vst1q_s32((int32_t*)out + 8, vdupq_n_s32(0));
            vst1q_s32((int32_t*)out + 12, vdupq_n_s32(0));
        }
        out += 16 * sizeof(uint32_t);
        w0 = w15;
        s += 16;
    }

    if XNN_UNLIKELY(n != 0){
        assert(n >= 1 && n < 16);
        float* packed_k_scaled_sum = (float*) out;
        float32x4_t packed_k_scaled_sums0123 = vdupq_n_f32(0.0f);
        float32x4_t packed_k_scaled_sums4567 = vdupq_n_f32(0.0f);
        float32x4_t packed_k_scaled_sums89AB = vdupq_n_f32(0.0f);
        float32x4_t packed_k_scaled_sumsCDEF = vdupq_n_f32(0.0f);
        out += 16 * sizeof(float);
        const uint8_t* w1 = w0 + (kc >> 1);
        if XNN_UNPREDICTABLE(n < 2) {
            w1 = w0;
        }
        const uint8_t* w2 = w1 + (kc >> 1);
        if XNN_UNPREDICTABLE(n <= 2) {
            w2 = w1;
        }
        const uint8_t* w3 = w2 + (kc >> 1);
        if XNN_UNPREDICTABLE(n < 4) {
            w3 = w2;
        }
        const uint8_t* w4 = w3 + (kc >> 1);
        if XNN_UNPREDICTABLE(n <= 4) {
            w4 = w3;
        }
        const uint8_t* w5 = w4 + (kc >> 1);
        if XNN_UNPREDICTABLE(n < 6) {
            w5 = w4;
        }
        const uint8_t* w6 = w5 + (kc >> 1);
        if XNN_UNPREDICTABLE(n <= 6) {
            w6 = w5;
        }
        const uint8_t* w7 = w6 + (kc >> 1);
        if XNN_UNPREDICTABLE(n < 8) {
            w7 = w6;
        }
        const uint8_t* w8 = w7 + (kc >> 1);
        if XNN_UNPREDICTABLE(n <= 8) {
            w8 = w7;
        }
        const uint8_t* w9 = w8 + (kc >> 1);
        if XNN_UNPREDICTABLE(n < 10) {
            w9 = w8;
        }
        const uint8_t* w10 = w9 + (kc >> 1);
        if XNN_UNPREDICTABLE(n <= 10) {
            w10 = w9;
        }
        const uint8_t* w11 = w10 + (kc >> 1);
        if XNN_UNPREDICTABLE(n < 12) {
            w11 = w10;
        }
        const uint8_t* w12 = w11 + (kc >> 1);
        if XNN_UNPREDICTABLE(n <= 12) {
            w12 = w11;
        }
        const uint8_t* w13 = w12 + (kc >> 1);
        if XNN_UNPREDICTABLE(n < 14) {
            w13 = w12;
        }
        const uint8_t* w14 = w13 + (kc >> 1);
        if XNN_UNPREDICTABLE(n <= 14) {
            w14 = w13;
        }
        const uint8_t* w15 = w14 + (kc >> 1);
        if XNN_UNPREDICTABLE(n < 16) {
            w15 = w14;
        }
        const uint16_t* s0 = s;

        size_t kb = kc;
        // Process k by blocks (bl)
        for (; kb >= bl; kb-=bl) {
            // Initialize KSum as subtracting bl zero points (8)
            int32x4_t ksum0123 = vdupq_n_s32(0);
            int32x4_t ksum4567 = vdupq_n_s32(0);
            int32x4_t ksum89AB = vdupq_n_s32(0);
            int32x4_t ksumCDEF = vdupq_n_s32(0);
            size_t k = bl;

            // KC Main loop multiple of 16x32
            for(; k >= 32; k-=32) {
                uint64x2_t w0x01 = vld1q_u64((uint64_t*) w0); w0 += 16;
                uint64x2_t w1x01 = vld1q_u64((uint64_t*) w1); w1 += 16;
                uint64x2_t w2x01 = vld1q_u64((uint64_t*) w2); w2 += 16;
                uint64x2_t w3x01 = vld1q_u64((uint64_t*) w3); w3 += 16;
                uint64x2_t w4x01 = vld1q_u64((uint64_t*) w4); w4 += 16;
                uint64x2_t w5x01 = vld1q_u64((uint64_t*) w5); w5 += 16;
                uint64x2_t w6x01 = vld1q_u64((uint64_t*) w6); w6 += 16;
                uint64x2_t w7x01 = vld1q_u64((uint64_t*) w7); w7 += 16;
                uint64x2_t w8x01 = vld1q_u64((uint64_t*) w8); w8 += 16;
                uint64x2_t w9x01 = vld1q_u64((uint64_t*) w9); w9 += 16;
                uint64x2_t wAx01 = vld1q_u64((uint64_t*) w10); w10 += 16;
                uint64x2_t wBx01 = vld1q_u64((uint64_t*) w11); w11 += 16;
                uint64x2_t wCx01 = vld1q_u64((uint64_t*) w12); w12 += 16;
                uint64x2_t wDx01 = vld1q_u64((uint64_t*) w13); w13 += 16;
                uint64x2_t wEx01 = vld1q_u64((uint64_t*) w14); w14 += 16;
                uint64x2_t wFx01 = vld1q_u64((uint64_t*) w15); w15 += 16;

                uint64x2_t v01_0 = vzip1q_u64(w0x01, w1x01);
                uint64x2_t v01_1 = vzip2q_u64(w0x01, w1x01);
                uint64x2_t v23_0 = vzip1q_u64(w2x01, w3x01);
                uint64x2_t v23_1 = vzip2q_u64(w2x01, w3x01);
                uint64x2_t v45_0 = vzip1q_u64(w4x01, w5x01);
                uint64x2_t v45_1 = vzip2q_u64(w4x01, w5x01);
                uint64x2_t v67_0 = vzip1q_u64(w6x01, w7x01);
                uint64x2_t v67_1 = vzip2q_u64(w6x01, w7x01);
                uint64x2_t v89_0 = vzip1q_u64(w8x01, w9x01);
                uint64x2_t v89_1 = vzip2q_u64(w8x01, w9x01);
                uint64x2_t vAB_0 = vzip1q_u64(wAx01, wBx01);
                uint64x2_t vAB_1 = vzip2q_u64(wAx01, wBx01);
                uint64x2_t vCD_0 = vzip1q_u64(wCx01, wDx01);
                uint64x2_t vCD_1 = vzip2q_u64(wCx01, wDx01);
                uint64x2_t vEF_0 = vzip1q_u64(wEx01, wFx01);
                uint64x2_t vEF_1 = vzip2q_u64(wEx01, wFx01);

                v01_0 = xnn_packed2planar(&ksum0123, v01_0, vmask, veor_mask, neg_zp, vone_zero);
                v23_0 = xnn_packed2planar(&ksum0123, v23_0, vmask, veor_mask, neg_zp, vzero_one);
                v45_0 = xnn_packed2planar(&ksum4567, v45_0, vmask, veor_mask, neg_zp, vone_zero);
                v67_0 = xnn_packed2planar(&ksum4567, v67_0, vmask, veor_mask, neg_zp, vzero_one);
                v89_0 = xnn_packed2planar(&ksum89AB, v89_0, vmask, veor_mask, neg_zp, vone_zero);
                vAB_0 = xnn_packed2planar(&ksum89AB, vAB_0, vmask, veor_mask, neg_zp, vzero_one);
                vCD_0 = xnn_packed2planar(&ksumCDEF, vCD_0, vmask, veor_mask, neg_zp, vone_zero);
                vEF_0 = xnn_packed2planar(&ksumCDEF, vEF_0, vmask, veor_mask, neg_zp, vzero_one);
                v01_1 = xnn_packed2planar(&ksum0123, v01_1, vmask, veor_mask, neg_zp, vone_zero);
                v23_1 = xnn_packed2planar(&ksum0123, v23_1, vmask, veor_mask, neg_zp, vzero_one);
                v45_1 = xnn_packed2planar(&ksum4567, v45_1, vmask, veor_mask, neg_zp, vone_zero);
                v67_1 = xnn_packed2planar(&ksum4567, v67_1, vmask, veor_mask, neg_zp, vzero_one);
                v89_1 = xnn_packed2planar(&ksum89AB, v89_1, vmask, veor_mask, neg_zp, vone_zero);
                vAB_1 = xnn_packed2planar(&ksum89AB, vAB_1, vmask, veor_mask, neg_zp, vzero_one);
                vCD_1 = xnn_packed2planar(&ksumCDEF, vCD_1, vmask, veor_mask, neg_zp, vone_zero);
                vEF_1 = xnn_packed2planar(&ksumCDEF, vEF_1, vmask, veor_mask, neg_zp, vzero_one);

                vst1q_u8(&out[0], v01_0);
                vst1q_u8(&out[16], v23_0);
                vst1q_u8(&out[32], v45_0);
                vst1q_u8(&out[48], v67_0);
                vst1q_u8(&out[64], v89_0);
                vst1q_u8(&out[80], vAB_0);
                vst1q_u8(&out[96], vCD_0);
                vst1q_u8(&out[112], vEF_0);
                vst1q_u8(&out[128], v01_1);
                vst1q_u8(&out[144], v23_1);
                vst1q_u8(&out[160], v45_1);
                vst1q_u8(&out[176], v67_1);
                vst1q_u8(&out[192], v89_1);
                vst1q_u8(&out[208], vAB_1);
                vst1q_u8(&out[224], vCD_1);
                vst1q_u8(&out[240], vEF_1);

                out += 256;
            }

            float32x4_t f_scales0123 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(s0 + 0), 16));
            float32x4_t f_scales4567 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(s0 + 4), 16));
            float32x4_t f_scales89AB = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(s0 + 8), 16));
            float32x4_t f_scalesCDEF = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(s0 + 12), 16));
            s0 += n;

            float32x4_t f_ksum0123 = vcvtq_f32_s32(ksum0123);
            f_ksum0123 = vmulq_f32(f_ksum0123, vzeropoint);
            packed_k_scaled_sums0123 = vfmsq_f32(packed_k_scaled_sums0123, f_ksum0123, f_scales0123);
            float32x4_t f_ksum4567 = vcvtq_f32_s32(ksum4567);
            f_ksum4567 = vmulq_f32(f_ksum4567, vzeropoint);
            packed_k_scaled_sums4567 = vfmsq_f32(packed_k_scaled_sums4567, f_ksum4567, f_scales4567);
            float32x4_t f_ksum89AB = vcvtq_f32_s32(ksum89AB);
            f_ksum89AB = vmulq_f32(f_ksum89AB, vzeropoint);
            packed_k_scaled_sums89AB = vfmsq_f32(packed_k_scaled_sums89AB, f_ksum89AB, f_scales89AB);
            float32x4_t f_ksumCDEF = vcvtq_f32_s32(ksumCDEF);
            f_ksumCDEF = vmulq_f32(f_ksumCDEF, vzeropoint);
            packed_k_scaled_sumsCDEF = vfmsq_f32(packed_k_scaled_sumsCDEF, f_ksumCDEF, f_scalesCDEF);

            vst1q_f32(&packed_k_scaled_sum[0], packed_k_scaled_sums0123);
            vst1q_f32(&packed_k_scaled_sum[4], packed_k_scaled_sums4567);
            vst1q_f32(&packed_k_scaled_sum[8], packed_k_scaled_sums89AB);
            vst1q_f32(&packed_k_scaled_sum[12], packed_k_scaled_sumsCDEF);

            f_scales0123 = vmulq_f32(f_scales0123, vrecip_sixteen);
            f_scales4567 = vmulq_f32(f_scales4567, vrecip_sixteen);
            f_scales89AB = vmulq_f32(f_scales89AB, vrecip_sixteen);
            f_scalesCDEF = vmulq_f32(f_scalesCDEF, vrecip_sixteen);

            vst1_u16((uint16_t*)out+0, vshrn_n_s32(vreinterpretq_s32_f32(f_scales0123), 16));
            vst1_u16((uint16_t*)out+4, vshrn_n_s32(vreinterpretq_s32_f32(f_scales4567), 16));
            vst1_u16((uint16_t*)out+8, vshrn_n_s32(vreinterpretq_s32_f32(f_scales89AB), 16));
            vst1_u16((uint16_t*)out+12, vshrn_n_s32(vreinterpretq_s32_f32(f_scalesCDEF), 16));

            out += 16 * sizeof(uint16_t);
        }

        if XNN_LIKELY(b != NULL){
            const int32x4_t b0123 = vld1q_s32(&b[0]);
            vst1q_s32((int32_t*)out + 0, b0123);
            const int32x4_t b4567 = vld1q_s32(&b[4]);
            vst1q_s32((int32_t*)out + 4, b4567);
            const int32x4_t b89AB = vld1q_s32(&b[8]);
            vst1q_s32((int32_t*)out + 8, b89AB);
            const int32x4_t bCDEF = vld1q_s32(&b[12]);
            vst1q_s32((int32_t*)out + 12, bCDEF);
            b += 16;
        } else {
            vst1q_s32((int32_t*)out + 0, vdupq_n_s32(0));
            vst1q_s32((int32_t*)out + 4, vdupq_n_s32(0));
            vst1q_s32((int32_t*)out + 8, vdupq_n_s32(0));
            vst1q_s32((int32_t*)out + 12, vdupq_n_s32(0));
        }
        out += 16 * sizeof(uint32_t);

    }
  } while (--g != 0);
}
