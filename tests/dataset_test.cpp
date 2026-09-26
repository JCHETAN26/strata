#include "strata/dataset.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "test_util.hpp"

namespace strata {
namespace {

class DatasetIo : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    dir_ = std::filesystem::temp_directory_path() /
           (std::string("strata_") + info->test_suite_name() + "_" + info->name());
    std::filesystem::remove_all(dir_);
    std::filesystem::create_directories(dir_);
  }
  void TearDown() override { std::filesystem::remove_all(dir_); }

  std::filesystem::path dir_;
};

TEST_F(DatasetIo, FbinRoundTrip) {
  const auto m = test::random_matrix(5, 3, 1);
  ASSERT_TRUE(write_fbin(dir_ / "m.fbin", m));
  auto read = read_fbin(dir_ / "m.fbin");
  ASSERT_TRUE(read.has_value()) << read.error().message;
  EXPECT_EQ(read->rows(), 5U);
  EXPECT_EQ(read->cols(), 3U);
  EXPECT_TRUE(std::equal(m.data().begin(), m.data().end(), read->data().begin()));
  EXPECT_EQ(std::filesystem::file_size(dir_ / "m.fbin"), 8U + 5U * 3U * sizeof(float));
}

TEST_F(DatasetIo, IbinRoundTrip) {
  const Matrix<std::int32_t> m(2, 2, {1, -2, 3, 2147483647});
  ASSERT_TRUE(write_ibin(dir_ / "m.ibin", m));
  auto read = read_ibin(dir_ / "m.ibin");
  ASSERT_TRUE(read.has_value());
  EXPECT_TRUE(std::equal(m.data().begin(), m.data().end(), read->data().begin()));
}

TEST_F(DatasetIo, EmptyMatrixRoundTrip) {
  ASSERT_TRUE(write_fbin(dir_ / "e.fbin", Matrix<float>(0, 128)));
  auto read = read_fbin(dir_ / "e.fbin");
  ASSERT_TRUE(read.has_value());
  EXPECT_EQ(read->rows(), 0U);
  EXPECT_EQ(read->cols(), 128U);
}

TEST_F(DatasetIo, MissingFile) {
  auto read = read_fbin(dir_ / "nope.fbin");
  ASSERT_FALSE(read.has_value());
  EXPECT_EQ(read.error().code, ErrorCode::kIoError);
}

TEST_F(DatasetIo, TruncatedFile) {
  ASSERT_TRUE(write_fbin(dir_ / "t.fbin", test::random_matrix(4, 4, 1)));
  std::filesystem::resize_file(dir_ / "t.fbin", 8 + 15 * sizeof(float));
  auto read = read_fbin(dir_ / "t.fbin");
  ASSERT_FALSE(read.has_value());
  EXPECT_EQ(read.error().code, ErrorCode::kCorruptData);
}

TEST_F(DatasetIo, FileShorterThanHeader) {
  std::ofstream(dir_ / "h.fbin", std::ios::binary) << "abc";
  auto read = read_fbin(dir_ / "h.fbin");
  ASSERT_FALSE(read.has_value());
  EXPECT_EQ(read.error().code, ErrorCode::kCorruptData);
}

TEST_F(DatasetIo, LoadDatasetChecksShapes) {
  ASSERT_TRUE(write_fbin(dir_ / "base.fbin", test::random_matrix(10, 4, 1)));
  ASSERT_TRUE(write_fbin(dir_ / "query.fbin", test::random_matrix(2, 4, 2)));
  ASSERT_TRUE(write_ibin(dir_ / "groundtruth.ibin", Matrix<std::int32_t>(2, 3)));
  EXPECT_TRUE(load_dataset(dir_).has_value());

  ASSERT_TRUE(write_fbin(dir_ / "query.fbin", test::random_matrix(2, 5, 2)));
  auto bad_dim = load_dataset(dir_);
  ASSERT_FALSE(bad_dim.has_value());
  EXPECT_EQ(bad_dim.error().code, ErrorCode::kDimensionMismatch);

  ASSERT_TRUE(write_fbin(dir_ / "query.fbin", test::random_matrix(3, 4, 2)));
  auto bad_rows = load_dataset(dir_);
  ASSERT_FALSE(bad_rows.has_value());
  EXPECT_EQ(bad_rows.error().code, ErrorCode::kCorruptData);
}

}  // namespace
}  // namespace strata
