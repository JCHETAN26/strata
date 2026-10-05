# Runtime image for strata_shard and strata_coordinator. The binaries are built on the host first
# (they link gRPC statically from vcpkg), so the image only needs the C++ runtime:
#
#   cmake --preset linux-server-release && cmake --build --preset linux-server-release
#   docker build -f deploy/docker/Dockerfile -t strata-server build/linux-server-release/server
#
# The build context is the server build directory. The base image must provide a libstdc++ at
# least as new as the build compiler's (GCC 13: GLIBCXX_3.4.32) and a glibc at least as new as the
# build host's. Ubuntu 24.04 covers binaries built with GCC 13 on Ubuntu 22.04 or 24.04; stock
# ubuntu:22.04 does not (its libstdc++ is GCC 12's).
FROM ubuntu:24.04
# Compose runs the containers as the host user (STRATA_UID/STRATA_GID), so the mounted key and
# token can stay mode 600 and owned by that user. /data is writable by any uid, sticky like /tmp,
# so each shard's named volume (initialized from it) works under whichever uid runs it.
RUN mkdir -p /data && chmod 1777 /data
COPY strata_shard strata_coordinator /usr/local/bin/
USER 65534:65534
WORKDIR /data
