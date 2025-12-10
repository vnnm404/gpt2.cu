FROM nixos/nix:latest AS pure
RUN mkdir -p /root/.config/nix && \
  echo "experimental-features = nix-command flakes" >> /root/.config/nix/nix.conf && \
  echo "ssl-cert-file = /root/.nix-profile/etc/ssl/certs/ca-bundle.crt" >> /root/.config/nix/nix.conf
COPY . /app
WORKDIR /app
CMD ["nix", "run", ".#inference"]

FROM ubuntu:22.04 AS dev
RUN apt-get update && apt-get install -y curl xz-utils ca-certificates
RUN curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix \
    | sh -s -- install linux --no-confirm --init none
ENV PATH="/nix/var/nix/profiles/default/bin:${PATH}"

RUN apt-get update && apt-get install -y curl sudo ca-certificates && rm -rf /var/lib/apt/lists/*

RUN nix profile add nixpkgs#tini nixpkgs#gosu

COPY --chmod=440 <<-EOF /etc/sudoers.d/nopass
%sudo ALL=(ALL) NOPASSWD:ALL
EOF

COPY --chmod=755 <<-"EOF" /usr/local/bin/entrypoint
#!/usr/bin/env bash
set -eu

log() {
  printf "[entrypoint] %s\n" "$*";
}

if [ -d /usr/local/bin/entrypoint.d ]; then
  for f in /usr/local/bin/entrypoint.d/*.sh; do
    [ -e "$f" ] || continue
    log "running: $(basename "$f")"
    . "$f"
  done
fi

if declare -f entrypoint_exec >/dev/null 2>&1; then
  entrypoint_exec "$@"
fi

exec "$@"
EOF

# 00-user.sh
COPY --chmod=755 <<-"EOF" /usr/local/bin/entrypoint.d/00-user.sh
USER_NAME="${USER_NAME:-dev}"
USER_UID="${USER_UID:-1000}"
USER_GID="${USER_GID:-1000}"

log "setting up user '$USER_NAME' uid=$USER_UID gid=$USER_GID"

if ! grep -q "^${USER_GID}:" /etc/group; then
  groupadd -g "$USER_GID" "$USER_NAME"
fi

if ! id -u "$USER_UID" >/dev/null 2>&1; then
  useradd -m -u "$USER_UID" -g "$USER_GID" -s /bin/bash "$USER_NAME"
fi
usermod -aG sudo "$USER_NAME"
export USER_NAME USER_UID USER_GID

if [ -n "$USER_UID" ] && [ -n "$USER_GID" ]; then
    chown -R "$USER_UID:$USER_GID" /nix
    chown -R "$USER_UID:$USER_GID" /nix/var/nix
fi
EOF

# 10-gosu.sh
COPY <<-"EOF" /usr/local/bin/entrypoint.d/10-gosu.sh
entrypoint_exec() {
  # if interactive + no command OR command is just "bash" -> start real login shell
  if [[ -t 1 && ( "$#" -eq 0 || ( "$#" -eq 1 && "$1" = "bash" ) ) ]]; then
    exec gosu "$USER_UID:$USER_GID" bash --login
  fi
  # otherwise forward
  exec gosu "$USER_UID:$USER_GID" "$@"
}
EOF

WORKDIR /workspace
ENTRYPOINT ["tini", "--", "/usr/local/bin/entrypoint"]

