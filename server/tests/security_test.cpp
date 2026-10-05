// Network exposure policy and transport security: TLS between coordinator and shards, and the
// shared-token check. Certificates come from scripts/make_dev_certs.sh, run at build time into
// STRATA_TEST_CERT_DIR: "good" (CA, localhost server certificate, token) and "other" (an unrelated
// CA a client must not trust).

#include "security.hpp"

#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "coordinator_service.hpp"
#include "shard_service.hpp"
#include "strata/collection.hpp"

namespace strata::server {
namespace {

namespace fs = std::filesystem;

constexpr std::size_t kDim = 4;
const fs::path kCerts = STRATA_TEST_CERT_DIR;

std::string must_read(const fs::path& path) {
  auto contents = read_file(path);
  EXPECT_TRUE(contents.has_value()) << path;
  return contents ? *contents : std::string();
}

// ---- Exposure policy (no network) ---------------------------------------------------------------

TEST(ExposurePolicy, LoopbackDetection) {
  for (const char* host : {"127.0.0.1", "127.1.2.3", "localhost", "::1", "[::1]"}) {
    EXPECT_TRUE(is_loopback(host)) << host;
  }
  for (const char* host : {"0.0.0.0", "10.0.0.5", "128.0.0.1", "::", "example.com", ""}) {
    EXPECT_FALSE(is_loopback(host)) << host;
  }
}

TEST(ExposurePolicy, HostPortBracketsIpv6) {
  EXPECT_EQ(host_port("127.0.0.1", 50051), "127.0.0.1:50051");
  EXPECT_EQ(host_port("::1", 50051), "[::1]:50051");
  EXPECT_EQ(host_port("[::1]", 50051), "[::1]:50051");
}

TEST(ExposurePolicy, DefaultIsLoopbackAndAllowed) {
  const ListenConfig config;
  EXPECT_EQ(config.host, "127.0.0.1");
  EXPECT_FALSE(exposure_error(config).has_value());
}

TEST(ExposurePolicy, ExposingRequiresTlsAndTokenOrInsecure) {
  ListenConfig config;
  config.host = "0.0.0.0";
  EXPECT_TRUE(exposure_error(config).has_value()) << "plaintext on all interfaces";

  config.tls_cert_file = "server.pem";
  config.tls_key_file = "server.key";
  EXPECT_TRUE(exposure_error(config).has_value()) << "TLS but anyone may call";

  config.token_file = "token";
  EXPECT_FALSE(exposure_error(config).has_value());

  ListenConfig insecure;
  insecure.host = "0.0.0.0";
  insecure.insecure = true;
  EXPECT_FALSE(exposure_error(insecure).has_value()) << "explicitly accepted";
}

TEST(ExposurePolicy, TokenNeedsTlsAndCertNeedsKey) {
  ListenConfig token_only;
  token_only.token_file = "token";
  EXPECT_TRUE(exposure_error(token_only).has_value());

  ListenConfig cert_only;
  cert_only.tls_cert_file = "server.pem";
  EXPECT_TRUE(exposure_error(cert_only).has_value());

  ShardClientConfig client;
  client.token_file = "token";
  EXPECT_TRUE(shard_client_error(client).has_value());
  client.ca_file = "ca.pem";
  EXPECT_FALSE(shard_client_error(client).has_value());
}

TEST(ExposurePolicy, StartServerRefusesUnsafeExposure) {
  ListenConfig config;
  config.host = "0.0.0.0";
  config.port = 0;
  v1::VectorService::Service service;
  int port = 0;
  auto server = start_server(config, &service, &port);
  ASSERT_FALSE(server.has_value());
  EXPECT_NE(server.error().message.find("refusing"), std::string::npos);
}

TEST(ExposurePolicy, SecondServerOnTheSamePortFailsToStart) {
  v1::VectorService::Service first_service;
  v1::VectorService::Service second_service;
  ListenConfig config;  // 127.0.0.1, plaintext
  int port = 0;
  auto first = start_server(config, &first_service, &port);
  ASSERT_TRUE(first.has_value()) << first.error().message;
  ASSERT_GT(port, 0);

  config.port = port;
  int second_port = 0;
  auto second = start_server(config, &second_service, &second_port);
  EXPECT_FALSE(second.has_value()) << "a second server shared port " << port;
  (*first)->Shutdown();
}

TEST(ExposurePolicy, ShortTokensAreRejected) {
  const fs::path path = fs::temp_directory_path() / "strata_short_token";
  {
    std::ofstream out(path);
    out << "short\n";
  }
  EXPECT_FALSE(read_token(path).has_value());
  fs::remove(path);
  auto good = read_token(kCerts / "good" / "token");
  ASSERT_TRUE(good.has_value()) << good.error().message;
  EXPECT_EQ(good->size(), 64u) << "trailing newline stripped";
}

// ---- TLS and token over real connections --------------------------------------------------------

// A shard serving TLS with the "good" certificate and requiring the "good" token.
class SecureShardTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           ("strata_security_test_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)));
    fs::create_directories(dir_);
    auto collection = Collection::open(dir_, kDim, Metric::kL2, {});
    ASSERT_TRUE(collection.has_value()) << collection.error().message;
    service_ = std::make_unique<ShardService>(std::move(*collection), kDim, Metric::kL2);

    ListenConfig config;
    config.port = 0;
    config.tls_cert_file = kCerts / "good" / "server.pem";
    config.tls_key_file = kCerts / "good" / "server.key";
    config.token_file = kCerts / "good" / "token";
    auto server = start_server(config, service_.get(), &port_);
    ASSERT_TRUE(server.has_value()) << server.error().message;
    server_ = std::move(*server);

    ca_ = must_read(kCerts / "good" / "ca.pem");
    token_ = *read_token(kCerts / "good" / "token");
  }

  void TearDown() override {
    if (server_) {
      server_->Shutdown();
    }
    std::error_code ec;
    fs::remove_all(dir_, ec);
  }

  std::string address() const { return "127.0.0.1:" + std::to_string(port_); }

  // A Stats call over `credentials`, with a deadline so a failed handshake cannot hang the test.
  grpc::Status stats(const std::shared_ptr<grpc::ChannelCredentials>& credentials) const {
    auto stub = v1::VectorService::NewStub(grpc::CreateChannel(address(), credentials));
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
    v1::StatsRequest request;
    v1::StatsResponse response;
    return stub->Stats(&context, request, &response);
  }

  fs::path dir_;
  std::unique_ptr<ShardService> service_;
  std::unique_ptr<grpc::Server> server_;
  int port_ = 0;
  std::string ca_;
  std::string token_;
};

TEST_F(SecureShardTest, AcceptsTrustedTlsWithTheToken) {
  grpc::Status status = stats(tls_channel_credentials(ca_, token_));
  EXPECT_TRUE(status.ok()) << status.error_message();
}

TEST_F(SecureShardTest, RejectsAMissingToken) {
  EXPECT_EQ(stats(tls_channel_credentials(ca_, "")).error_code(),
            grpc::StatusCode::UNAUTHENTICATED);
}

TEST_F(SecureShardTest, RejectsAWrongToken) {
  std::string wrong = token_;
  wrong.back() = wrong.back() == '0' ? '1' : '0';
  EXPECT_EQ(stats(tls_channel_credentials(ca_, wrong)).error_code(),
            grpc::StatusCode::UNAUTHENTICATED);
}

TEST_F(SecureShardTest, RejectsAPlaintextClient) {
  EXPECT_EQ(stats(grpc::InsecureChannelCredentials()).error_code(), grpc::StatusCode::UNAVAILABLE);
}

TEST_F(SecureShardTest, ClientRejectsAnUntrustedServerCertificate) {
  // The client trusts only the other CA, so the shard's certificate does not verify: the
  // handshake fails and the token is never sent.
  const std::string other_ca = must_read(kCerts / "other" / "ca.pem");
  EXPECT_EQ(stats(tls_channel_credentials(other_ca, token_)).error_code(),
            grpc::StatusCode::UNAVAILABLE);
}

TEST_F(SecureShardTest, CoordinatorReachesShardOverTlsWithToken) {
  CoordinatorOptions options;
  options.shard_credentials = tls_channel_credentials(ca_, token_);
  CoordinatorService coordinator({address()}, kDim, Metric::kL2, options);

  const std::vector<float> vector = {1.0f, 2.0f, 3.0f, 4.0f};
  v1::InsertRequest insert;
  insert.mutable_values()->Add(vector.begin(), vector.end());
  v1::InsertResponse inserted;
  grpc::Status status = coordinator.Insert(nullptr, &insert, &inserted);
  ASSERT_TRUE(status.ok()) << status.error_message();

  v1::SearchRequest search;
  search.mutable_values()->Add(vector.begin(), vector.end());
  search.set_k(1);
  v1::SearchResponse found;
  status = coordinator.Search(nullptr, &search, &found);
  ASSERT_TRUE(status.ok()) << status.error_message();
  ASSERT_EQ(found.neighbors_size(), 1);
  EXPECT_EQ(found.neighbors(0).id(), inserted.id());
}

TEST_F(SecureShardTest, CoordinatorWithAWrongTokenIsRejected) {
  CoordinatorOptions options;
  options.shard_credentials = tls_channel_credentials(ca_, std::string(64, 'f'));
  CoordinatorService coordinator({address()}, kDim, Metric::kL2, options);
  v1::StatsRequest request;
  v1::StatsResponse response;
  EXPECT_EQ(coordinator.Stats(nullptr, &request, &response).error_code(),
            grpc::StatusCode::UNAUTHENTICATED);
}

}  // namespace
}  // namespace strata::server
