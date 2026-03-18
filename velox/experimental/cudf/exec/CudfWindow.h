/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cudf/aggregation.hpp>

#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/vector/CudfVector.h"
#include "velox/exec/Operator.h"

namespace facebook::velox::cudf_velox {

/// Minimal GPU-backed window implementation for rank-like functions.
///
/// This operator currently implements only a narrow but useful subset:
/// `row_number()`, `rank()`, and `dense_rank()` over
/// `(partition by ... order by ...)`.
/// It keeps all data on GPU, reuses pre-sorted input when available, otherwise
/// sorts once to establish window order, computes the window result column
/// using cuDF scan primitives, and returns a CudfVector so downstream GPU
/// operators can continue without an immediate CPU boundary.
class CudfWindow : public exec::Operator, public NvtxHelper {
 public:
  CudfWindow(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::WindowNode>& windowNode);

  bool needsInput() const override {
    return !noMoreInput_;
  }

  void addInput(RowVectorPtr input) override;

  void noMoreInput() override;

  RowVectorPtr getOutput() override;

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return finished_;
  }

  void close() override;

 private:
  std::unique_ptr<cudf::column> computeRankLikeColumn(
      cudf::table_view const& sortedInput,
      rmm::cuda_stream_view stream) const;

  std::unique_ptr<cudf::table> computeOutputTable(
      std::unique_ptr<cudf::table> input,
      rmm::cuda_stream_view stream) const;

  const std::shared_ptr<const core::WindowNode> windowNode_;
  const RowTypePtr inputType_;
  const std::vector<cudf::size_type> partitionKeyChannels_;
  const cudf::size_type sortKeyChannel_;
  const cudf::order sortOrder_;
  const cudf::null_order nullOrder_;
  const cudf::rank_method rankMethod_;

  std::vector<CudfVectorPtr> inputs_;
  CudfVectorPtr output_;
  bool finished_{false};
};

bool isSupportedCudfWindowNode(
    const std::shared_ptr<const core::WindowNode>& node);

} // namespace facebook::velox::cudf_velox
