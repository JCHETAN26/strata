#include "security.hpp"

#include <arpa/inet.h>
#include <grpcpp/security/auth_metadata_processor.h>

#include <cstddef>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <utility>

namespace strata::server {
namespace {

constexpr std::string_view kAuthHeader = "authorization";
constexpr std::string_view kBearer = "Bearer ";

// Compares without returning early at the first differing byte, so response timing does not
// reveal how much of a guessed token was right. (The length can leak; it is not secret.)
bool constant_time_equal(std::string_view a, std::string_view b) {
  if (a.size() != b.size()) {
    return false;
  }
  unsigned char diff = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    diff |= static_cast<unsigned char>(a[i]) ^ static_cast<unsigned char>(b[i]);
  }
  return diff == 0;
}

// Server side: runs before the service for every call on a TLS port and rejects calls without
// the right bearer token. gRPC only runs an AuthMetadataProcessor on secure credentials, which is
// one more reason a token requires TLS.
class TokenAuthProcessor final : public grpc::AuthMetadataProcessor {
 public:
  explicit TokenAuthProcessor(std::string token) : expected_(std::string(kBearer) + token) {}

  // A string compare: cheap enough to run on the call's own thread.
  bool IsBlocking() const override { return false; }

  grpc::Status Process(const InputMetadata& auth_metadata, grpc::AuthContext* /*context*/,
                       OutputMetadata* consumed_auth_metadata,
                       OutputMetadata* /*response_metadata*/) override {
    const auto it = auth_metadata.find(grpc::string_ref(kAuthHeader.data(), kAuthHeader.size()));
    if (it == auth_metadata.end() ||
        !constant_time_equal(std::string_view(it->second.data(), it->second.size()), expected_)) {
      return {grpc::StatusCode::UNAUTHENTICATED, "missing or invalid token"};
    }
    // Consumed: the token is hidden from the service handlers, which have no use for it.
    consumed_auth_metadata->insert({std::string(it->first.data(), it->first.size()),
                                    std::string(it->second.data(), it->second.size())});
    return grpc::Status::OK;
  }

 private:
  std::string expected_;
};

// Client side: attaches the bearer token to every call. gRPC refuses to send call credentials
// over an insecure channel, so this only ever travels inside TLS.
class TokenPlugin final : public grpc::MetadataCredentialsPlugin {
 public:
  explicit TokenPlugin(const std::string& token) : header_(std::string(kBearer) + token) {}

  bool IsBlocking() const override { return false; }

  grpc::Status GetMetadata(grpc::string_ref /*service_url*/, grpc::string_ref /*method_name*/,
                           const grpc::AuthContext& /*channel_auth_context*/,
                           std::multimap<std::string, std::string>* metadata) override {
    metadata->insert({std::string(kAuthHeader), header_});
    return grpc::Status::OK;
  }

 private:
  std::string header_;
};

}  // namespace

bool is_loopback(std::string_view host) {
  if (host == "localhost" || host == "::1" || host == "[::1]") {
    return true;
  }
  in_addr v4{};
  const std::string h(host);
  if (inet_pton(AF_INET, h.c_str(), &v4) == 1) {
    return (ntohl(v4.s_addr) >> 24) == 127;
  }
  return false;
}

std::string host_port(std::string_view host, int port) {
  const bool ipv6_literal = host.find(':') != std::string_view::npos && !host.starts_with('[');
  std::string out = ipv6_literal ? "[" + std::string(host) + "]" : std::string(host);
  return out + ":" + std::to_string(port);
}

std::optional<std::string> exposure_error(const ListenConfig& config) {
  const bool has_cert = !config.tls_cert_file.empty();
  const bool has_key = !config.tls_key_file.empty();
  if (has_cert != has_key) {
    return "--tls-cert and --tls-key must be given together";
  }
  const bool tls = has_cert;
  const bool token = !config.token_file.empty();
  if (token && !tls) {
    return "--token-file needs TLS (--tls-cert and --tls-key): the token would otherwise cross "
           "the network in plain text";
  }
  if (!is_loopback(config.host) && !(tls && token) && !config.insecure) {
    return "refusing to listen on " + config.host +
           " without TLS and a token: pass --tls-cert, --tls-key and --token-file, or --insecure "
           "to expose the server anyway";
  }
  if (config.max_threads < 0) {
    return "--max-threads must be positive";
  }
  return std::nullopt;
}

std::optional<std::string> shard_client_error(const ShardClientConfig& config) {
  if (!config.token_file.empty() && config.ca_file.empty()) {
    return "--shard-token-file needs --shard-ca: the token would otherwise cross the network in "
           "plain text";
  }
  return std::nullopt;
}

Expected<std::string> read_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return make_error(ErrorCode::kIoError, "cannot read " + path.string());
  }
  std::ostringstream contents;
  contents << in.rdbuf();
  if (in.bad()) {
    return make_error(ErrorCode::kIoError, "error reading " + path.string());
  }
  return contents.str();
}

Expected<std::string> read_token(const std::filesystem::path& path) {
  auto token = read_file(path);
  if (!token) {
    return token;
  }
  while (!token->empty() && (token->back() == '\n' || token->back() == '\r' ||
                             token->back() == ' ' || token->back() == '\t')) {
    token->pop_back();
  }
  if (token->size() < kMinTokenLength) {
    return make_error(ErrorCode::kInvalidArgument,
                      "token in " + path.string() + " is shorter than " +
                          std::to_string(kMinTokenLength) + " characters");
  }
  return token;
}

std::shared_ptr<grpc::ServerCredentials> tls_server_credentials(const std::string& cert_chain_pem,
                                                                const std::string& private_key_pem,
                                                                const std::string& token) {
  grpc::SslServerCredentialsOptions options;
  options.pem_key_cert_pairs.push_back({private_key_pem, cert_chain_pem});
  auto credentials = grpc::SslServerCredentials(options);
  if (!token.empty()) {
    credentials->SetAuthMetadataProcessor(std::make_shared<TokenAuthProcessor>(token));
  }
  return credentials;
}

std::shared_ptr<grpc::ChannelCredentials> tls_channel_credentials(const std::string& root_certs_pem,
                                                                  const std::string& token) {
  grpc::SslCredentialsOptions options;
  options.pem_root_certs = root_certs_pem;
  auto channel = grpc::SslCredentials(options);
  if (token.empty()) {
    return channel;
  }
  return grpc::CompositeChannelCredentials(
      channel, grpc::MetadataCredentialsFromPlugin(std::make_unique<TokenPlugin>(token)));
}

Expected<std::shared_ptr<grpc::ServerCredentials>> make_server_credentials(
    const ListenConfig& config) {
  if (config.tls_cert_file.empty()) {
    return grpc::InsecureServerCredentials();
  }
  auto cert = read_file(config.tls_cert_file);
  if (!cert) {
    return tl::unexpected(cert.error());
  }
  auto key = read_file(config.tls_key_file);
  if (!key) {
    return tl::unexpected(key.error());
  }
  std::string token;
  if (!config.token_file.empty()) {
    auto t = read_token(config.token_file);
    if (!t) {
      return tl::unexpected(t.error());
    }
    token = std::move(*t);
  }
  return tls_server_credentials(*cert, *key, token);
}

Expected<std::shared_ptr<grpc::ChannelCredentials>> make_shard_credentials(
    const ShardClientConfig& config) {
  if (config.ca_file.empty()) {
    return grpc::InsecureChannelCredentials();
  }
  auto ca = read_file(config.ca_file);
  if (!ca) {
    return tl::unexpected(ca.error());
  }
  std::string token;
  if (!config.token_file.empty()) {
    auto t = read_token(config.token_file);
    if (!t) {
      return tl::unexpected(t.error());
    }
    token = std::move(*t);
  }
  return tls_channel_credentials(*ca, token);
}

Expected<std::unique_ptr<grpc::Server>> start_server(const ListenConfig& config,
                                                     grpc::Service* service, int* bound_port) {
  if (auto refused = exposure_error(config)) {
    return make_error(ErrorCode::kInvalidArgument, *refused);
  }
  auto credentials = make_server_credentials(config);
  if (!credentials) {
    return tl::unexpected(credentials.error());
  }
  const std::string address = host_port(config.host, config.port);
  grpc::ServerBuilder builder;
  builder.AddListeningPort(address, *credentials, bound_port);
  builder.RegisterService(service);
  if (config.max_threads > 0) {
    grpc::ResourceQuota quota("strata_server");
    quota.SetMaxThreads(config.max_threads);
    builder.SetResourceQuota(quota);
  }
  std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
  if (!server || *bound_port == 0) {
    return make_error(ErrorCode::kIoError, "failed to listen on " + address);
  }
  return server;
}

}  // namespace strata::server
