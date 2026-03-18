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

#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::exec::test {

using core::QueryConfig;
using facebook::velox::test::BatchMaker;
using namespace common::testutil;

class ToCudfSelectionTest : public OperatorTestBase {
 protected:
  static void SetUpTestCase() {
    OperatorTestBase::SetUpTestCase();
    TestValue::enable();
  }

  void SetUp() override {
    OperatorTestBase::SetUp();
    filesystems::registerLocalFileSystem();
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    OperatorTestBase::TearDown();
  }

  std::vector<RowVectorPtr>
  makeVectors(const RowTypePtr& rowType, size_t size, int numVectors) {
    std::vector<RowVectorPtr> vectors;
    VectorFuzzer fuzzer({.vectorSize = size}, pool());
    for (int32_t i = 0; i < numVectors; ++i) {
      vectors.push_back(fuzzer.fuzzInputRow(rowType));
    }
    return vectors;
  }

  std::vector<RowVectorPtr> makeWindowVectors() {
    return {
        makeRowVector(
            {makeFlatVector<int64_t>({1, 2, 1, 2, 1, 2, 1, 2}),
             makeFlatVector<int16_t>({20, 7, 10, 5, 10, 9, 30, 5}),
             makeFlatVector<int32_t>({0, 1, 2, 3, 4, 5, 6, 7}),
             makeFlatVector<int64_t>({100, 101, 102, 103, 104, 105, 106, 107}),
             makeFlatVector<double>({0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}),
             makeFlatVector<double>({1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8}),
             makeFlatVector<StringView>({"a", "b", "c", "d", "e", "f", "g", "h"})})};
  }

  std::vector<RowVectorPtr> makeSortedWindowVectors() {
    return {
        makeRowVector(
            {makeFlatVector<int64_t>({1, 1, 1, 1, 2, 2, 2, 2}),
             makeFlatVector<int16_t>({10, 10, 20, 30, 5, 5, 7, 9}),
             makeFlatVector<int32_t>({0, 1, 2, 3, 4, 5, 6, 7}),
             makeFlatVector<int64_t>({100, 101, 102, 103, 104, 105, 106, 107}),
             makeFlatVector<double>({0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}),
             makeFlatVector<double>({1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8}),
             makeFlatVector<StringView>({"a", "b", "c", "d", "e", "f", "g", "h"})})};
  }

  bool wasCudfAggregationUsed(const std::shared_ptr<exec::Task>& task) {
    auto stats = task->taskStats();
    for (const auto& pipelineStats : stats.pipelineStats) {
      for (const auto& operatorStats : pipelineStats.operatorStats) {
        if (operatorStats.operatorType == "CudfAggregation") {
          return true;
        }
      }
    }
    return false;
  }

  bool wasDefaultHashAggregationUsed(const std::shared_ptr<exec::Task>& task) {
    auto stats = task->taskStats();
    for (const auto& pipelineStats : stats.pipelineStats) {
      for (const auto& operatorStats : pipelineStats.operatorStats) {
        if (operatorStats.operatorType == "Aggregation") {
          return true;
        }
      }
    }
    return false;
  }

  bool wasDefaultWindowUsed(const std::shared_ptr<exec::Task>& task) {
    auto stats = task->taskStats();
    for (const auto& pipelineStats : stats.pipelineStats) {
      for (const auto& operatorStats : pipelineStats.operatorStats) {
        if (operatorStats.operatorType == "Window") {
          return true;
        }
      }
    }
    return false;
  }

  bool wasCudfWindowUsed(const std::shared_ptr<exec::Task>& task) {
    auto stats = task->taskStats();
    for (const auto& pipelineStats : stats.pipelineStats) {
      for (const auto& operatorStats : pipelineStats.operatorStats) {
        if (operatorStats.operatorType == "CudfWindow") {
          return true;
        }
      }
    }
    return false;
  }

  int countOperators(
      const std::shared_ptr<exec::Task>& task,
      const std::string& operatorType) {
    int count = 0;
    auto stats = task->taskStats();
    for (const auto& pipelineStats : stats.pipelineStats) {
      for (const auto& operatorStats : pipelineStats.operatorStats) {
        if (operatorStats.operatorType == operatorType) {
          ++count;
        }
      }
    }
    return count;
  }

  RowTypePtr rowType_{
      ROW({"c0", "c1", "c2", "c3", "c4", "c5", "c6"},
          {BIGINT(),
           SMALLINT(),
           INTEGER(),
           BIGINT(),
           DOUBLE(),
           DOUBLE(),
           VARCHAR()})};
};

// Test supported aggregation should use CUDF
TEST_F(ToCudfSelectionTest, supportedAggregationUsesCudf) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {"c0"},
                      {"sum(c1)", "count(c2)", "min(c3)", "max(c4)", "avg(c5)"},
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config("cudf.enabled", true)
          .plan(plan)
          .assertResults(
              "SELECT c0, sum(c1), count(c2), min(c3), max(c4), avg(c5) FROM tmp GROUP BY c0");

  ASSERT_TRUE(wasCudfAggregationUsed(task));
  ASSERT_FALSE(wasDefaultHashAggregationUsed(task));
}

// Test unsupported aggregation should fall back to CPU
TEST_F(ToCudfSelectionTest, unsupportedAggregationFallsBack) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {"c0"},
                      {"stddev(c1)", "variance(c2)"},
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config("cudf.enabled", true)
          .plan(plan)
          .assertResults(
              "SELECT c0, stddev(c1), variance(c2) FROM tmp GROUP BY c0");

  ASSERT_FALSE(wasCudfAggregationUsed(task));
  ASSERT_TRUE(wasDefaultHashAggregationUsed(task));
}

// Test mixed supported/unsupported should fall back
TEST_F(ToCudfSelectionTest, mixedSupportFallsBack) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {"c0"},
                      {"sum(c1)", "stddev(c2)"},
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config("cudf.enabled", true)
          .plan(plan)
          .assertResults("SELECT c0, sum(c1), stddev(c2) FROM tmp GROUP BY c0");

  ASSERT_FALSE(wasCudfAggregationUsed(task));
  ASSERT_TRUE(wasDefaultHashAggregationUsed(task));
}

// Test supported global aggregation should use CUDF
TEST_F(ToCudfSelectionTest, supportedGlobalAggregationUsesCudf) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {},
                      {"sum(c1)", "count(c2)", "max(c3)"},
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config("cudf.enabled", true)
                  .plan(plan)
                  .assertResults("SELECT sum(c1), count(c2), max(c3) FROM tmp");

  ASSERT_TRUE(wasCudfAggregationUsed(task));
  ASSERT_FALSE(wasDefaultHashAggregationUsed(task));
}

// Test unsupported global aggregation should fall back
TEST_F(ToCudfSelectionTest, unsupportedGlobalAggregationFallsBack) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {},
                      {"stddev(c1)"},
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config("cudf.enabled", true)
                  .plan(plan)
                  .assertResults("SELECT stddev(c1) FROM tmp");

  ASSERT_FALSE(wasCudfAggregationUsed(task));
  ASSERT_TRUE(wasDefaultHashAggregationUsed(task));
}

// Test supported grouping key expressions should use CUDF
TEST_F(ToCudfSelectionTest, supportedGroupingKeyExpressionsUsesCudf) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .project(
                      {"c0",
                       "c1",
                       "c2",
                       "c3",
                       "c4",
                       "c5",
                       "c6",
                       "c0 + c1 as key1",
                       "length(c6) as key2"})
                  .aggregation(
                      {"key1", "key2"},
                      {"sum(c2)"},
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config("cudf.enabled", true)
          .plan(plan)
          .assertResults(
              "SELECT c0 + c1, length(c6), sum(c2) FROM tmp GROUP BY c0 + c1, length(c6)");

  ASSERT_TRUE(wasCudfAggregationUsed(task));
  ASSERT_FALSE(wasDefaultHashAggregationUsed(task));
}

// Test unsupported aggregation functions should fall back
TEST_F(ToCudfSelectionTest, unsupportedAggregationFunctionsFallsBack) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {"c0"},
                      {"sum(c1)", "stddev(c2)"},
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config("cudf.enabled", true)
          .plan(plan)
          .assertResults("SELECT c0, sum(c1), stddev(c2) FROM tmp GROUP BY c0");

  ASSERT_FALSE(wasCudfAggregationUsed(task));
  ASSERT_TRUE(wasDefaultHashAggregationUsed(task));
}

// Test complex grouping key expressions should fall back
TEST_F(ToCudfSelectionTest, complexGroupingKeyExpressionsFallsBack) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .project({"c0", "c1", "c2", "abs(c0) AS complex_key"})
                  .aggregation(
                      {"complex_key"},
                      {"sum(c2)"},
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config("cudf.enabled", true)
          .plan(plan)
          .assertResults("SELECT abs(c0), sum(c2) FROM tmp GROUP BY abs(c0)");

  ASSERT_FALSE(wasCudfAggregationUsed(task));
  ASSERT_TRUE(wasDefaultHashAggregationUsed(task));
}

// Test supported aggregation input expressions should use CUDF
TEST_F(ToCudfSelectionTest, supportedAggregationInputExpressionsUsesCudf) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {"c0"},
                      {"sum(c1)", "max(c2)"},
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config("cudf.enabled", true)
          .plan(plan)
          .assertResults("SELECT c0, sum(c1), max(c2) FROM tmp GROUP BY c0");

  ASSERT_TRUE(wasCudfAggregationUsed(task));
  ASSERT_FALSE(wasDefaultHashAggregationUsed(task));
}

// Test unsupported aggregation input expressions should fall back
TEST_F(ToCudfSelectionTest, unsupportedAggregationInputExpressionsFallsBack) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {"c0"},
                      {"sum(c1)", "variance(c2)"},
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config("cudf.enabled", true)
                  .plan(plan)
                  .assertResults(
                      "SELECT c0, sum(c1), variance(c2) FROM tmp GROUP BY c0");

  ASSERT_FALSE(wasCudfAggregationUsed(task));
  ASSERT_TRUE(wasDefaultHashAggregationUsed(task));
}

// Test CUDF disabled should always use regular aggregation
TEST_F(ToCudfSelectionTest, cudfDisabledUsesRegularAggregation) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {"c0"},
                      {"sum(c1)", "count(c2)"},
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config("cudf.enabled", false)
          .plan(plan)
          .assertResults("SELECT c0, sum(c1), count(c2) FROM tmp GROUP BY c0");

  ASSERT_FALSE(wasCudfAggregationUsed(task));
  ASSERT_TRUE(wasDefaultHashAggregationUsed(task));
}

TEST_F(ToCudfSelectionTest, supportedRankWindowUsesCudfWindow) {
  auto vectors = makeWindowVectors();
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .project({"c0", "c1", "c2", "c3", "c4", "c5", "c6"})
                  .window({"rank() over (partition by c0 order by c1) as rk"})
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config("cudf.enabled", true)
          .plan(plan)
          .assertResults(
              "SELECT c0, c1, c2, c3, c4, c5, c6, "
              "rank() OVER (PARTITION BY c0 ORDER BY c1) AS rk FROM tmp");

  ASSERT_TRUE(wasCudfWindowUsed(task));
  ASSERT_GE(countOperators(task, "CudfFromVelox"), 1);
  ASSERT_EQ(countOperators(task, "CudfToVelox"), 0);
}

TEST_F(ToCudfSelectionTest, supportedRowNumberWindowUsesCudfWindow) {
  auto vectors = makeWindowVectors();
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .project({"c0", "c1", "c2", "c3", "c4", "c5", "c6"})
                  .window({"row_number() over (partition by c0 order by c1) as rn"})
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config("cudf.enabled", true)
          .plan(plan)
          .assertResults(
              "SELECT c0, c1, c2, c3, c4, c5, c6, "
              "row_number() OVER (PARTITION BY c0 ORDER BY c1) AS rn FROM tmp");

  ASSERT_TRUE(wasCudfWindowUsed(task));
  ASSERT_GE(countOperators(task, "CudfFromVelox"), 1);
  ASSERT_EQ(countOperators(task, "CudfToVelox"), 0);
}

TEST_F(ToCudfSelectionTest, supportedGlobalRankWindowUsesCudfWindow) {
  auto vectors = makeWindowVectors();
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .project({"c0", "c1", "c2", "c3", "c4", "c5", "c6"})
                  .window({"rank() over (order by c1) as rk"})
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config("cudf.enabled", true)
          .plan(plan)
          .assertResults(
              "SELECT c0, c1, c2, c3, c4, c5, c6, "
              "rank() OVER (ORDER BY c1) AS rk FROM tmp");

  ASSERT_TRUE(wasCudfWindowUsed(task));
  ASSERT_GE(countOperators(task, "CudfFromVelox"), 1);
  ASSERT_EQ(countOperators(task, "CudfToVelox"), 0);
}

TEST_F(ToCudfSelectionTest, supportedDenseRankWindowUsesCudfWindow) {
  auto vectors = makeWindowVectors();
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .project({"c0", "c1", "c2", "c3", "c4", "c5", "c6"})
                  .window({"dense_rank() over (partition by c0 order by c1) as rk"})
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config("cudf.enabled", true)
          .plan(plan)
          .assertResults(
              "SELECT c0, c1, c2, c3, c4, c5, c6, "
              "dense_rank() OVER (PARTITION BY c0 ORDER BY c1) AS rk FROM tmp");

  ASSERT_TRUE(wasCudfWindowUsed(task));
  ASSERT_GE(countOperators(task, "CudfFromVelox"), 1);
  ASSERT_EQ(countOperators(task, "CudfToVelox"), 0);
}

TEST_F(ToCudfSelectionTest, supportedStreamingRankWindowUsesCudfWindow) {
  auto vectors = makeSortedWindowVectors();
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .project({"c0", "c1", "c2", "c3", "c4", "c5", "c6"})
                  .streamingWindow({"rank() over (partition by c0 order by c1) as rk"})
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config("cudf.enabled", true)
          .plan(plan)
          .assertResults(
              "SELECT c0, c1, c2, c3, c4, c5, c6, "
              "rank() OVER (PARTITION BY c0 ORDER BY c1) AS rk FROM tmp");

  ASSERT_TRUE(wasCudfWindowUsed(task));
  ASSERT_GE(countOperators(task, "CudfFromVelox"), 1);
  ASSERT_EQ(countOperators(task, "CudfToVelox"), 0);
}

TEST_F(ToCudfSelectionTest, supportedStreamingRowNumberWindowUsesCudfWindow) {
  auto vectors = makeSortedWindowVectors();
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .project({"c0", "c1", "c2", "c3", "c4", "c5", "c6"})
                  .streamingWindow(
                      {"row_number() over (partition by c0 order by c1) as rn"})
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config("cudf.enabled", true)
          .plan(plan)
          .assertResults(
              "SELECT c0, c1, c2, c3, c4, c5, c6, "
              "row_number() OVER (PARTITION BY c0 ORDER BY c1) AS rn FROM tmp");

  ASSERT_TRUE(wasCudfWindowUsed(task));
  ASSERT_GE(countOperators(task, "CudfFromVelox"), 1);
  ASSERT_EQ(countOperators(task, "CudfToVelox"), 0);
}

TEST_F(ToCudfSelectionTest, supportedGlobalDenseRankWindowUsesCudfWindow) {
  auto vectors = makeWindowVectors();
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .project({"c0", "c1", "c2", "c3", "c4", "c5", "c6"})
                  .window({"dense_rank() over (order by c1) as rk"})
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config("cudf.enabled", true)
          .plan(plan)
          .assertResults(
              "SELECT c0, c1, c2, c3, c4, c5, c6, "
              "dense_rank() OVER (ORDER BY c1) AS rk FROM tmp");

  ASSERT_TRUE(wasCudfWindowUsed(task));
  ASSERT_GE(countOperators(task, "CudfFromVelox"), 1);
  ASSERT_EQ(countOperators(task, "CudfToVelox"), 0);
}

TEST_F(ToCudfSelectionTest, unsupportedWindowFrameFallsBackToExplicitToVelox) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .project({"c0", "c1", "c2", "c3", "c4", "c5", "c6"})
                  .window(
                      {"sum(c3) over (partition by c0 order by c1 rows between 1 preceding and current row) as w"})
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config("cudf.enabled", true)
          .plan(plan)
          .assertResults(
              "SELECT c0, c1, c2, c3, c4, c5, c6, "
              "sum(c3) OVER (PARTITION BY c0 ORDER BY c1 "
              "ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS w FROM tmp");

  ASSERT_TRUE(wasDefaultWindowUsed(task));
  ASSERT_GE(countOperators(task, "CudfFromVelox"), 1);
  ASSERT_GE(countOperators(task, "CudfToVelox"), 1);
}

TEST_F(
    ToCudfSelectionTest,
    unsupportedWindowExpressionArgFallsBackToExplicitToVelox) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .project({"c0", "c1", "c2", "c3", "c4", "c5", "c6"})
                  .window(
                      {"sum(c3 + c2) over (partition by c0 order by c1) as w"})
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config("cudf.enabled", true)
          .plan(plan)
          .assertResults(
              "SELECT c0, c1, c2, c3, c4, c5, c6, "
              "sum(c3 + c2) OVER (PARTITION BY c0 ORDER BY c1) AS w FROM tmp");

  ASSERT_TRUE(wasDefaultWindowUsed(task));
  ASSERT_GE(countOperators(task, "CudfFromVelox"), 1);
  ASSERT_GE(countOperators(task, "CudfToVelox"), 1);
}

} // namespace facebook::velox::exec::test
