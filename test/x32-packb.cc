// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <string>

#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/packb.h"
#include "packb-microkernel-tester.h"

struct XnnTestParam {
  const char *name;
  bool (*isa_check)();
  xnn_x32_packb_gemm_ukernel_fn fn;
  size_t channel_tile, channel_subtile, channel_round;
};

class XnnTest : public testing::TestWithParam<XnnTestParam> {
};

std::string GetTestName(const testing::TestParamInfo<XnnTest::ParamType>& info) {
  return info.param.name;
}

const XnnTestParam xnn_test_params[] = {
  { "X32_PACKB_GEMM_2C1S1R__SCALAR_FLOAT", []() { return true; }, xnn_x32_packb_gemm_ukernel_2c1s1r__scalar_float, /*channel_tile=*/2, /*channel_subtile=*/1, /*channel_round=*/1 },
  { "X32_PACKB_GEMM_2C1S1R__SCALAR_INT", []() { return true; }, xnn_x32_packb_gemm_ukernel_2c1s1r__scalar_int, /*channel_tile=*/2, /*channel_subtile=*/1, /*channel_round=*/1 },
  { "X32_PACKB_GEMM_2C2S1R__SCALAR_FLOAT", []() { return true; }, xnn_x32_packb_gemm_ukernel_2c2s1r__scalar_float, /*channel_tile=*/2, /*channel_subtile=*/2, /*channel_round=*/1 },
  { "X32_PACKB_GEMM_2C2S1R__SCALAR_INT", []() { return true; }, xnn_x32_packb_gemm_ukernel_2c2s1r__scalar_int, /*channel_tile=*/2, /*channel_subtile=*/2, /*channel_round=*/1 },
  { "X32_PACKB_GEMM_4C1S1R__SCALAR_FLOAT", []() { return true; }, xnn_x32_packb_gemm_ukernel_4c1s1r__scalar_float, /*channel_tile=*/4, /*channel_subtile=*/1, /*channel_round=*/1 },
  { "X32_PACKB_GEMM_4C1S1R__SCALAR_INT", []() { return true; }, xnn_x32_packb_gemm_ukernel_4c1s1r__scalar_int, /*channel_tile=*/4, /*channel_subtile=*/1, /*channel_round=*/1 },
  { "X32_PACKB_GEMM_4C4S1R__SCALAR_FLOAT", []() { return true; }, xnn_x32_packb_gemm_ukernel_4c4s1r__scalar_float, /*channel_tile=*/4, /*channel_subtile=*/4, /*channel_round=*/1 },
  { "X32_PACKB_GEMM_4C4S1R__SCALAR_INT", []() { return true; }, xnn_x32_packb_gemm_ukernel_4c4s1r__scalar_int, /*channel_tile=*/4, /*channel_subtile=*/4, /*channel_round=*/1 },
};

TEST_P(XnnTest, n_eq_channel_tile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t k = 1; k < 4; k++) {
    PackBMicrokernelTester()
      .channels(GetParam().channel_tile)
      .kernel_tile(k)
      .channel_tile(GetParam().channel_tile)
      .channel_subtile(GetParam().channel_subtile)
      .channel_round(GetParam().channel_round)
      .Test(GetParam().fn);
  }
}

TEST_P(XnnTest, n_div_channel_tile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1) {
    GTEST_SKIP();
  }
  for (size_t k = 1; k < 4; k++) {
    PackBMicrokernelTester()
      .channels(GetParam().channel_tile * 2)
      .kernel_tile(k)
      .channel_tile(GetParam().channel_tile)
      .channel_subtile(GetParam().channel_subtile)
      .channel_round(GetParam().channel_round)
      .Test(GetParam().fn);
  }
}

TEST_P(XnnTest, n_lt_channel_tile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1) {
    GTEST_SKIP();
  }
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 1; n < GetParam().channel_tile; n++) {
      PackBMicrokernelTester()
        .channels(n)
        .kernel_tile(k)
        .channel_tile(GetParam().channel_tile)
        .channel_subtile(GetParam().channel_subtile)
        .channel_round(GetParam().channel_round)
        .Test(GetParam().fn);
    }
  }
}

TEST_P(XnnTest, n_gt_channel_tile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = GetParam().channel_tile + 1; n < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); n++) {
      PackBMicrokernelTester()
        .channels(n)
        .kernel_tile(k)
        .channel_tile(GetParam().channel_tile)
        .channel_subtile(GetParam().channel_subtile)
        .channel_round(GetParam().channel_round)
        .Test(GetParam().fn);
    }
  }
}

TEST_P(XnnTest, groups_gt_1) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t g = 2; g <= 3; g++) {
    for (size_t kernel_tile = 1; kernel_tile < 4; kernel_tile++) {
      for (size_t n = GetParam().channel_tile + 1; n < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); n++) {
        PackBMicrokernelTester()
          .groups(g)
          .channels(n)
          .kernel_tile(kernel_tile)
          .channel_tile(GetParam().channel_tile)
          .channel_subtile(GetParam().channel_subtile)
          .channel_round(GetParam().channel_round)
          .Test(GetParam().fn);
      }
    }
  }
}
INSTANTIATE_TEST_SUITE_P(x32_packb,
                         XnnTest,
                         testing::ValuesIn(xnn_test_params),
                         GetTestName);

