#!/usr/bin/env bash
# Generates a private CA, a server certificate signed by it, and a shared token, for TLS and token
# auth between strata_coordinator and strata_shard (see README, "Securing the server").
#
#   scripts/make_dev_certs.sh <out_dir> [subjectAltName entries...]
#
#   scripts/make_dev_certs.sh certs                                  # localhost + 127.0.0.1
#   scripts/make_dev_certs.sh certs DNS:shard1.internal IP:10.0.0.5  # real hosts
#
# Writes into <out_dir>:
#   ca.pem       CA certificate: give to clients (--shard-ca)
#   ca.key       CA private key: keep offline; only needed to sign more certificates
#   server.pem   server certificate (--tls-cert); its SANs must name every host clients dial
#   server.key   server private key (--tls-key)
#   token        shared token, 64 hex characters (--token-file, --shard-token-file)
#
# One certificate for all shards keeps setup simple: list every shard host in its SANs. Keys and
# the token are created mode 600. The CA is valid for 10 years, the server certificate for 825
# days (the longest many TLS clients accept). Works with OpenSSL 3 and LibreSSL (macOS).
set -euo pipefail

if [[ $# -lt 1 ]]; then
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi
out=$1
shift
sans=("$@")
if [[ ${#sans[@]} -eq 0 ]]; then
  sans=(DNS:localhost IP:127.0.0.1)
fi
san_line=$(IFS=,; echo "${sans[*]}")

mkdir -p "$out"
umask 077
cnf="$out/openssl.cnf"
cat > "$cnf" <<EOF
[req]
distinguished_name = dn
prompt = no
[dn]
CN = Strata
[v3_ca]
basicConstraints = critical, CA:true
keyUsage = critical, keyCertSign, cRLSign
subjectKeyIdentifier = hash
[v3_server]
basicConstraints = critical, CA:false
keyUsage = critical, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = $san_line
EOF

openssl req -x509 -newkey rsa:2048 -nodes -sha256 -days 3650 -config "$cnf" -extensions v3_ca \
  -subj "/CN=Strata dev CA" -keyout "$out/ca.key" -out "$out/ca.pem" 2>/dev/null
openssl req -new -newkey rsa:2048 -nodes -sha256 -config "$cnf" \
  -subj "/CN=strata-server" -keyout "$out/server.key" -out "$out/server.csr" 2>/dev/null
openssl x509 -req -sha256 -days 825 -in "$out/server.csr" -CA "$out/ca.pem" -CAkey "$out/ca.key" \
  -CAcreateserial -extfile "$cnf" -extensions v3_server -out "$out/server.pem" 2>/dev/null
openssl rand -hex 32 > "$out/token"
rm -f "$out/server.csr" "$out/ca.srl" "$cnf"
chmod 644 "$out/ca.pem" "$out/server.pem"

echo "wrote $out/{ca.pem,ca.key,server.pem,server.key,token} (SAN: $san_line)"
