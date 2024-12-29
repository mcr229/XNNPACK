// Auto-generated file. Do not edit!
//   Template: src/qb4-packw/kr-neondot.c.in
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
    const uint8x16_t vones)
{
    const uint8x16_t vl = vshrq_n_u8(v, 4);    // isolate lower int 4
    const uint8x16_t vh = vandq_u8(v, vmask);  // isolate upper int 4
    *vacc = vreinterpretq_s32_u32(vdotq_u32(vreinterpretq_u32_s32(*vacc), vh, vones));
    *vacc = vreinterpretq_s32_u32(vdotq_u32(vreinterpretq_u32_s32(*vacc), vl, vones));
    *vacc = vaddq_s32(*vacc, neg_zp);
    const uint8x16_t v0123 = vzip1q_u8(vh, vl);
    const uint8x16_t v4567 = vzip2q_u8(vh, vl);
    const uint8x16_t v0246 = vreinterpretq_u8_u32(vuzp1q_u32(vreinterpretq_u32_u8(v0123), vreinterpretq_u32_u8(v4567)));
    const uint8x16_t v1357 = vreinterpretq_u8_u32(vuzp2q_u32(vreinterpretq_u32_u8(v0123), vreinterpretq_u32_u8(v4567)));
    const uint8x16_t vl1357 = vshlq_n_u8(v1357, 4);
    const uint8x16_t v01234567 = vorrq_u8(v0246, vl1357);
    return veorq_u8(v01234567, veor_mask);
}

void xnn_qb4_packw_gemm_goi_ukernel_x16c4__neondot(
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
  assert(kr == 4);
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
  const uint8x16_t vones = vmovq_n_u8(UINT8_C(0x01));

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
                uint32x4_t w0x0123 = vld1q_u32((uint32_t*) w0); w0 += 16;
                uint32x4_t w1x0123 = vld1q_u32((uint32_t*) w1); w1 += 16;
                uint32x4_t w2x0123 = vld1q_u32((uint32_t*) w2); w2 += 16;
                uint32x4_t w3x0123 = vld1q_u32((uint32_t*) w3); w3 += 16;
                uint32x4_t w4x0123 = vld1q_u32((uint32_t*) w4); w4 += 16;
                uint32x4_t w5x0123 = vld1q_u32((uint32_t*) w5); w5 += 16;
                uint32x4_t w6x0123 = vld1q_u32((uint32_t*) w6); w6 += 16;
                uint32x4_t w7x0123 = vld1q_u32((uint32_t*) w7); w7 += 16;
                uint32x4_t w8x0123 = vld1q_u32((uint32_t*) w8); w8 += 16;
                uint32x4_t w9x0123 = vld1q_u32((uint32_t*) w9); w9 += 16;
                uint32x4_t wAx0123 = vld1q_u32((uint32_t*) w10); w10 += 16;
                uint32x4_t wBx0123 = vld1q_u32((uint32_t*) w11); w11 += 16;
                uint32x4_t wCx0123 = vld1q_u32((uint32_t*) w12); w12 += 16;
                uint32x4_t wDx0123 = vld1q_u32((uint32_t*) w13); w13 += 16;
                uint32x4_t wEx0123 = vld1q_u32((uint32_t*) w14); w14 += 16;
                uint32x4_t wFx0123 = vld1q_u32((uint32_t*) w15); w15 += 16;

                uint32x4_t v01_02 = vtrn1q_u32(w0x0123, w1x0123);
                uint32x4_t v01_13 = vtrn2q_u32(w0x0123, w1x0123);
                uint32x4_t v23_02 = vtrn1q_u32(w2x0123, w3x0123);
                uint32x4_t v23_13 = vtrn2q_u32(w2x0123, w3x0123);
                uint32x4_t v45_02 = vtrn1q_u32(w4x0123, w5x0123);
                uint32x4_t v45_13 = vtrn2q_u32(w4x0123, w5x0123);
                uint32x4_t v67_02 = vtrn1q_u32(w6x0123, w7x0123);
                uint32x4_t v67_13 = vtrn2q_u32(w6x0123, w7x0123);
                uint32x4_t v89_02 = vtrn1q_u32(w8x0123, w9x0123);
                uint32x4_t v89_13 = vtrn2q_u32(w8x0123, w9x0123);
                uint32x4_t vAB_02 = vtrn1q_u32(wAx0123, wBx0123);
                uint32x4_t vAB_13 = vtrn2q_u32(wAx0123, wBx0123);
                uint32x4_t vCD_02 = vtrn1q_u32(wCx0123, wDx0123);
                uint32x4_t vCD_13 = vtrn2q_u32(wCx0123, wDx0123);
                uint32x4_t vEF_02 = vtrn1q_u32(wEx0123, wFx0123);
                uint32x4_t vEF_13 = vtrn2q_u32(wEx0123, wFx0123);

                uint32x4_t v0123_0 = vcombine_u32(vget_low_u32(v01_02), vget_low_u32(v23_02));
                uint32x4_t v0123_2 = vcombine_u32(vget_high_u32(v01_02), vget_high_u32(v23_02));
                uint32x4_t v0123_1 = vcombine_u32(vget_low_u32(v01_13), vget_low_u32(v23_13));
                uint32x4_t v0123_3 = vcombine_u32(vget_high_u32(v01_13), vget_high_u32(v23_13));
                uint32x4_t v4567_0 = vcombine_u32(vget_low_u32(v45_02), vget_low_u32(v67_02));
                uint32x4_t v4567_2 = vcombine_u32(vget_high_u32(v45_02), vget_high_u32(v67_02));
                uint32x4_t v4567_1 = vcombine_u32(vget_low_u32(v45_13), vget_low_u32(v67_13));
                uint32x4_t v4567_3 = vcombine_u32(vget_high_u32(v45_13), vget_high_u32(v67_13));
                uint32x4_t v89AB_0 = vcombine_u32(vget_low_u32(v89_02), vget_low_u32(vAB_02));
                uint32x4_t v89AB_2 = vcombine_u32(vget_high_u32(v89_02), vget_high_u32(vAB_02));
                uint32x4_t v89AB_1 = vcombine_u32(vget_low_u32(v89_13), vget_low_u32(vAB_13));
                uint32x4_t v89AB_3 = vcombine_u32(vget_high_u32(v89_13), vget_high_u32(vAB_13));
                uint32x4_t vCDEF_0 = vcombine_u32(vget_low_u32(vCD_02), vget_low_u32(vEF_02));
                uint32x4_t vCDEF_2 = vcombine_u32(vget_high_u32(vCD_02), vget_high_u32(vEF_02));
                uint32x4_t vCDEF_1 = vcombine_u32(vget_low_u32(vCD_13), vget_low_u32(vEF_13));
                uint32x4_t vCDEF_3 = vcombine_u32(vget_high_u32(vCD_13), vget_high_u32(vEF_13));

                v0123_0 = xnn_packed2planar(&ksum0123, v0123_0, vmask, veor_mask, neg_zp, vones);
                v0123_1 = xnn_packed2planar(&ksum0123, v0123_1, vmask, veor_mask, neg_zp, vones);
                v0123_2 = xnn_packed2planar(&ksum0123, v0123_2, vmask, veor_mask, neg_zp, vones);
                v0123_3 = xnn_packed2planar(&ksum0123, v0123_3, vmask, veor_mask, neg_zp, vones);
                v4567_0 = xnn_packed2planar(&ksum4567, v4567_0, vmask, veor_mask, neg_zp, vones);
                v4567_1 = xnn_packed2planar(&ksum4567, v4567_1, vmask, veor_mask, neg_zp, vones);
                v4567_2 = xnn_packed2planar(&ksum4567, v4567_2, vmask, veor_mask, neg_zp, vones);
                v4567_3 = xnn_packed2planar(&ksum4567, v4567_3, vmask, veor_mask, neg_zp, vones);
                v89AB_0 = xnn_packed2planar(&ksum89AB, v89AB_0, vmask, veor_mask, neg_zp, vones);
                v89AB_1 = xnn_packed2planar(&ksum89AB, v89AB_1, vmask, veor_mask, neg_zp, vones);
                v89AB_2 = xnn_packed2planar(&ksum89AB, v89AB_2, vmask, veor_mask, neg_zp, vones);
                v89AB_3 = xnn_packed2planar(&ksum89AB, v89AB_3, vmask, veor_mask, neg_zp, vones);
                vCDEF_0 = xnn_packed2planar(&ksumCDEF, vCDEF_0, vmask, veor_mask, neg_zp, vones);
                vCDEF_1 = xnn_packed2planar(&ksumCDEF, vCDEF_1, vmask, veor_mask, neg_zp, vones);
                vCDEF_2 = xnn_packed2planar(&ksumCDEF, vCDEF_2, vmask, veor_mask, neg_zp, vones);
                vCDEF_3 = xnn_packed2planar(&ksumCDEF, vCDEF_3, vmask, veor_mask, neg_zp, vones);

                vst1q_u8(&out[0], v0123_0);
                vst1q_u8(&out[16], v4567_0);
                vst1q_u8(&out[32], v89AB_0);
                vst1q_u8(&out[48], vCDEF_0);
                vst1q_u8(&out[64], v0123_1);
                vst1q_u8(&out[80], v4567_1);
                vst1q_u8(&out[96], v89AB_1);
                vst1q_u8(&out[112], vCDEF_1);
                vst1q_u8(&out[128], v0123_2);
                vst1q_u8(&out[144], v4567_2);
                vst1q_u8(&out[160], v89AB_2);
                vst1q_u8(&out[176], vCDEF_2);
                vst1q_u8(&out[192], v0123_3);
                vst1q_u8(&out[208], v4567_3);
                vst1q_u8(&out[224], v89AB_3);
                vst1q_u8(&out[240], vCDEF_3);

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
        const uint16_t* s0 = s;
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
                uint32x4_t w0x0123 = vld1q_u32((uint32_t*) w0);
                uint32x4_t w1x0123 = vld1q_u32((uint32_t*) w1);
                uint32x4_t w2x0123 = vld1q_u32((uint32_t*) w2);
                uint32x4_t w3x0123 = vld1q_u32((uint32_t*) w3);
                uint32x4_t w4x0123 = vld1q_u32((uint32_t*) w4);
                uint32x4_t w5x0123 = vld1q_u32((uint32_t*) w5);
                uint32x4_t w6x0123 = vld1q_u32((uint32_t*) w6);
                uint32x4_t w7x0123 = vld1q_u32((uint32_t*) w7);
                uint32x4_t w8x0123 = vld1q_u32((uint32_t*) w8);
                uint32x4_t w9x0123 = vld1q_u32((uint32_t*) w9);
                uint32x4_t wAx0123 = vld1q_u32((uint32_t*) w10);
                uint32x4_t wBx0123 = vld1q_u32((uint32_t*) w11);
                uint32x4_t wCx0123 = vld1q_u32((uint32_t*) w12);
                uint32x4_t wDx0123 = vld1q_u32((uint32_t*) w13);
                uint32x4_t wEx0123 = vld1q_u32((uint32_t*) w14);
                uint32x4_t wFx0123 = vld1q_u32((uint32_t*) w15);

                uint32x4_t v01_02 = vtrn1q_u32(w0x0123, w1x0123);
                uint32x4_t v01_13 = vtrn2q_u32(w0x0123, w1x0123);
                uint32x4_t v23_02 = vtrn1q_u32(w2x0123, w3x0123);
                uint32x4_t v23_13 = vtrn2q_u32(w2x0123, w3x0123);
                uint32x4_t v45_02 = vtrn1q_u32(w4x0123, w5x0123);
                uint32x4_t v45_13 = vtrn2q_u32(w4x0123, w5x0123);
                uint32x4_t v67_02 = vtrn1q_u32(w6x0123, w7x0123);
                uint32x4_t v67_13 = vtrn2q_u32(w6x0123, w7x0123);
                uint32x4_t v89_02 = vtrn1q_u32(w8x0123, w9x0123);
                uint32x4_t v89_13 = vtrn2q_u32(w8x0123, w9x0123);
                uint32x4_t vAB_02 = vtrn1q_u32(wAx0123, wBx0123);
                uint32x4_t vAB_13 = vtrn2q_u32(wAx0123, wBx0123);
                uint32x4_t vCD_02 = vtrn1q_u32(wCx0123, wDx0123);
                uint32x4_t vCD_13 = vtrn2q_u32(wCx0123, wDx0123);
                uint32x4_t vEF_02 = vtrn1q_u32(wEx0123, wFx0123);
                uint32x4_t vEF_13 = vtrn2q_u32(wEx0123, wFx0123);

                uint32x4_t v0123_0 = vcombine_u32(vget_low_u32(v01_02), vget_low_u32(v23_02));
                uint32x4_t v0123_2 = vcombine_u32(vget_high_u32(v01_02), vget_high_u32(v23_02));
                uint32x4_t v0123_1 = vcombine_u32(vget_low_u32(v01_13), vget_low_u32(v23_13));
                uint32x4_t v0123_3 = vcombine_u32(vget_high_u32(v01_13), vget_high_u32(v23_13));
                uint32x4_t v4567_0 = vcombine_u32(vget_low_u32(v45_02), vget_low_u32(v67_02));
                uint32x4_t v4567_2 = vcombine_u32(vget_high_u32(v45_02), vget_high_u32(v67_02));
                uint32x4_t v4567_1 = vcombine_u32(vget_low_u32(v45_13), vget_low_u32(v67_13));
                uint32x4_t v4567_3 = vcombine_u32(vget_high_u32(v45_13), vget_high_u32(v67_13));
                uint32x4_t v89AB_0 = vcombine_u32(vget_low_u32(v89_02), vget_low_u32(vAB_02));
                uint32x4_t v89AB_2 = vcombine_u32(vget_high_u32(v89_02), vget_high_u32(vAB_02));
                uint32x4_t v89AB_1 = vcombine_u32(vget_low_u32(v89_13), vget_low_u32(vAB_13));
                uint32x4_t v89AB_3 = vcombine_u32(vget_high_u32(v89_13), vget_high_u32(vAB_13));
                uint32x4_t vCDEF_0 = vcombine_u32(vget_low_u32(vCD_02), vget_low_u32(vEF_02));
                uint32x4_t vCDEF_2 = vcombine_u32(vget_high_u32(vCD_02), vget_high_u32(vEF_02));
                uint32x4_t vCDEF_1 = vcombine_u32(vget_low_u32(vCD_13), vget_low_u32(vEF_13));
                uint32x4_t vCDEF_3 = vcombine_u32(vget_high_u32(vCD_13), vget_high_u32(vEF_13));

                v0123_0 = xnn_packed2planar(&ksum0123, v0123_0, vmask, veor_mask, neg_zp, vones);
                v0123_1 = xnn_packed2planar(&ksum0123, v0123_1, vmask, veor_mask, neg_zp, vones);
                v0123_2 = xnn_packed2planar(&ksum0123, v0123_2, vmask, veor_mask, neg_zp, vones);
                v0123_3 = xnn_packed2planar(&ksum0123, v0123_3, vmask, veor_mask, neg_zp, vones);
                v4567_0 = xnn_packed2planar(&ksum4567, v4567_0, vmask, veor_mask, neg_zp, vones);
                v4567_1 = xnn_packed2planar(&ksum4567, v4567_1, vmask, veor_mask, neg_zp, vones);
                v4567_2 = xnn_packed2planar(&ksum4567, v4567_2, vmask, veor_mask, neg_zp, vones);
                v4567_3 = xnn_packed2planar(&ksum4567, v4567_3, vmask, veor_mask, neg_zp, vones);
                v89AB_0 = xnn_packed2planar(&ksum89AB, v89AB_0, vmask, veor_mask, neg_zp, vones);
                v89AB_1 = xnn_packed2planar(&ksum89AB, v89AB_1, vmask, veor_mask, neg_zp, vones);
                v89AB_2 = xnn_packed2planar(&ksum89AB, v89AB_2, vmask, veor_mask, neg_zp, vones);
                v89AB_3 = xnn_packed2planar(&ksum89AB, v89AB_3, vmask, veor_mask, neg_zp, vones);
                vCDEF_0 = xnn_packed2planar(&ksumCDEF, vCDEF_0, vmask, veor_mask, neg_zp, vones);
                vCDEF_1 = xnn_packed2planar(&ksumCDEF, vCDEF_1, vmask, veor_mask, neg_zp, vones);
                vCDEF_2 = xnn_packed2planar(&ksumCDEF, vCDEF_2, vmask, veor_mask, neg_zp, vones);
                vCDEF_3 = xnn_packed2planar(&ksumCDEF, vCDEF_3, vmask, veor_mask, neg_zp, vones);

                vst1q_u8(&out[0], v0123_0);
                vst1q_u8(&out[16], v4567_0);
                vst1q_u8(&out[32], v89AB_0);
                vst1q_u8(&out[48], vCDEF_0);
                vst1q_u8(&out[64], v0123_1);
                vst1q_u8(&out[80], v4567_1);
                vst1q_u8(&out[96], v89AB_1);
                vst1q_u8(&out[112], vCDEF_1);
                vst1q_u8(&out[128], v0123_2);
                vst1q_u8(&out[144], v4567_2);
                vst1q_u8(&out[160], v89AB_2);
                vst1q_u8(&out[176], vCDEF_2);
                vst1q_u8(&out[192], v0123_3);
                vst1q_u8(&out[208], v4567_3);
                vst1q_u8(&out[224], v89AB_3);
                vst1q_u8(&out[240], vCDEF_3);

                w0 += 16;
                w1 += 16;
                w2 += 16;
                w3 += 16;
                w4 += 16;
                w5 += 16;
                w6 += 16;
                w7 += 16;
                w8 += 16;
                w9 += 16;
                w10 += 16;
                w11 += 16;
                w12 += 16;
                w13 += 16;
                w14 += 16;
                w15 += 16;
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
