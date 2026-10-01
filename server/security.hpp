#pragma once

#include <grpcpp/grpcpp.h>

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "strata/error.hpp"

namespace strata::server {

// Network exposure and transport security, shared by strata_shard and strata_coordinator.
//
// The default is safe: a server listens on 127.0.0.1 only, so nothing off the machine can reach
// it. Listening anywhere else (--listen) requires TLS and a shared token, or an explicit
// --insecure. TLS alone authenticates the server to the client and encrypts the traffic; the token
// authenticates the client to the server (anyone who can open a TLS connection could otherwise
// call Insert and Delete). A token is refused without TLS, because it would cross the network in
// plain text.
//
// Thread safety: the functions are pure or read files; the credential objects they return are
// gRPC's and safe to share between threads.

// Where and how a server listens.
struct ListenConfig {
  std::string host = "127.0.0.1";
  int port = 0;
  std::filesystem::path tls_cert_file;  // PEM certificate chain the server presents
  std::filesystem::path tls_key_file;   // PEM private key for it
  std::filesystem::path token_file;     // shared token clients must present (needs TLS)
  bool insecure = false;                // allow a non-loopback listen without TLS and a token
  int max_threads = 0;                  // cap on gRPC server threads; 0 = gRPC's default
};

// How the coordinator connects to its shards. Empty ca_file means plaintext (shards on loopback).
struct ShardClientConfig {
  std::filesystem::path ca_file;     // PEM root certificate(s) the shards' certificates chain to
  std::filesystem::path token_file;  // token sent to the shards (needs ca_file)
};

// Minimum token length. A token is a shared secret; this rejects obviously weak ones (the setup
// script generates 64 hex characters).
inline constexpr std::size_t kMinTokenLength = 16;

// True for "localhost", "::1" (with or without brackets), and any 127.0.0.0/8 address.
[[nodiscard]] bool is_loopback(std::string_view host);

// "host:port", bracketing an IPv6 literal ("[::1]:50051").
[[nodiscard]] std::string host_port(std::string_view host, int port);

// Checks a listen configuration against the exposure policy above, without touching files.
// Returns the reason it is refused, or nullopt if it is acceptable.
[[nodiscard]] std::optional<std::string> exposure_error(const ListenConfig& config);

// Same check for the coordinator's connections to shards: a token needs TLS.
[[nodiscard]] std::optional<std::string> shard_client_error(const ShardClientConfig& config);

// Reads a whole file. For tokens, use read_token, which also strips trailing whitespace.
[[nodiscard]] Expected<std::string> read_file(const std::filesystem::path& path);
[[nodiscard]] Expected<std::string> read_token(const std::filesystem::path& path);

// Credentials from in-memory PEM and token. An empty token means no token check / none sent.
// Server: TLS with `cert_chain_pem` and `private_key_pem`; with a token, every call must carry
// "authorization: Bearer <token>" or it fails with UNAUTHENTICATED before reaching the service.
[[nodiscard]] std::shared_ptr<grpc::ServerCredentials> tls_server_credentials(
    const std::string& cert_chain_pem, const std::string& private_key_pem,
    const std::string& token);
// Client: TLS trusting `root_certs_pem`; with a token, attaches it to every call.
[[nodiscard]] std::shared_ptr<grpc::ChannelCredentials> tls_channel_credentials(
    const std::string& root_certs_pem, const std::string& token);

// The same, built from configuration (reading the files). Insecure credentials when TLS is not
// configured. Call exposure_error / shard_client_error first.
[[nodiscard]] Expected<std::shared_ptr<grpc::ServerCredentials>> make_server_credentials(
    const ListenConfig& config);
[[nodiscard]] Expected<std::shared_ptr<grpc::ChannelCredentials>> make_shard_credentials(
    const ShardClientConfig& config);

// Parses the listen and security flags shared by both binaries (--listen, --port, --tls-cert,
// --tls-key, --token-file, --insecure, --max-threads). `next` returns the flag's value. Returns
// false if `flag` is not one of them. Throws std::exception on a malformed value, like the
// binaries' own parsers.
template <typename Next>
bool parse_listen_flag(std::string_view flag, Next&& next, ListenConfig& config) {
  if (flag == "--listen") {
    config.host = next();
  } else if (flag == "--port") {
    config.port = std::stoi(next());
  } else if (flag == "--tls-cert") {
    config.tls_cert_file = next();
  } else if (flag == "--tls-key") {
    config.tls_key_file = next();
  } else if (flag == "--token-file") {
    config.token_file = next();
  } else if (flag == "--insecure") {
    config.insecure = true;
  } else if (flag == "--max-threads") {
    config.max_threads = std::stoi(next());
  } else {
    return false;
  }
  return true;
}

// Usage text for the flags parse_listen_flag accepts.
inline constexpr std::string_view kListenUsage =
    "  --listen <host>        address to listen on (default 127.0.0.1; e.g. 0.0.0.0 to expose)\n"
    "  --tls-cert <pem>       serve TLS with this certificate chain (with --tls-key)\n"
    "  --tls-key <pem>        private key for --tls-cert\n"
    "  --token-file <path>    require this shared token from every client (needs TLS)\n"
    "  --insecure             allow a non-loopback --listen without TLS and a token\n"
    "  --max-threads <n>      cap on gRPC server threads (default: gRPC's)\n";

// Builds and starts a server for `service` per `config`, after checking the exposure policy.
// On success returns the server and sets *bound_port (useful when config.port is 0).
[[nodiscard]] Expected<std::unique_ptr<grpc::Server>> start_server(const ListenConfig& config,
                                                                   grpc::Service* service,
                                                                   int* bound_port);

}  // namespace strata::server
