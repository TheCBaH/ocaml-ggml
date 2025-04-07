#!/bin/sh
set -eu
set -x

echo "Activating feature 'OCaml'"
PACKAGES=${PACKAGES:-$@}
SYSTEM_PACKAGES=${SYSTEM_PACKAGES:-}
OCAML_VERSION=${VERSION:-4.14.1}
OPAM_OPTIONS=''
if [ -n "${OPTIONS:-}" ]; then
    OPAM_OPTIONS="--packages=ocaml-variants.${OCAML_VERSION}+options,${OPTIONS}"
fi
echo "Selected OCaml:$OCAML_VERSION packages: $PACKAGES with ${OPAM_OPTIONS} ${SYSTEM_PACKAGES}"

# From https://github.com/devcontainers/features/blob/main/src/git/install.sh
apt_get_update()
{
    if [ "$(find /var/lib/apt/lists/* | wc -l)" = "0" ]; then
        echo "Running apt-get update..."
        apt-get update -y
    fi
}

# Checks if packages are installed and installs them if not
check_packages() {
    if ! dpkg -s "$@" > /dev/null 2>&1; then
        apt_get_update
        if ! apt-get -o Acquire::Retries=3 -y install --no-install-recommends "$@"; then
            apt-get update -y
            apt-get -o Acquire::Retries=3 -y install --no-install-recommends "$@"
        fi
    fi
}

export DEBIAN_FRONTEND=noninteractive

USERNAME="${USERNAME:-"${_REMOTE_USER:-"automatic"}"}"
UPDATE_RC="${UPDATE_RC:-"true"}"

# Determine the appropriate non-root user
if [ "${USERNAME}" = "auto" ] || [ "${USERNAME}" = "automatic" ]; then
    USERNAME=""
    POSSIBLE_USERS="vscode node codespace $(awk -v val=1000 -F ":" '$3==val{print $1}' /etc/passwd)"
    for CURRENT_USER in $POSSIBLE_USERS; do
        if id -u "${CURRENT_USER}" > /dev/null 2>&1; then
            USERNAME="${CURRENT_USER}"
            break
        fi
    done
    if [ "${USERNAME}" = "" ]; then
        USERNAME=root
    fi
elif [ "${USERNAME}" = "none" ] || ! id -u ${USERNAME} > /dev/null 2>&1; then
    USERNAME=root
fi

updaterc() {
    if [ "${UPDATE_RC}" = "true" ]; then
        echo "Updating /etc/bash.bashrc and /etc/zsh/zshrc..."
        if [ -f /etc/bash.bashrc ]; then
            /bin/echo -e "$1" >> /etc/bash.bashrc
        fi
        if [ -f "/etc/zsh/zshrc" ]; then
            /bin/echo -e "$1" >> /etc/zsh/zshrc
        fi
    fi
}

export OPAMROOT="/opt/opam"
export OPAMYES="true"
export OPAMCONFIRMLEVEL="unsafe-yes"

rc="$(cat << EOF
# >>> OCaml >>>
export OPAMROOT="$OPAMROOT"
# <<< OCaml <<<
EOF
)"
updaterc "$rc"

check_packages\
 ${SYSTEM_PACKAGES}\
 opam\

opam init --no-setup --disable-sandboxing --bare
eval $(opam env)
opam switch create $OCAML_VERSION ${OPAM_OPTIONS}
BASE_PACKAGES="\
 dune\
 ocaml-lsp-server\
 ocamlformat\
 ocamlformat-rpc\
"
ALL_PACKAGES="${BASE_PACKAGES} ${PACKAGES}"

opam install ${ALL_PACKAGES}

opam clean --repo-cache
opam list
chown -R ${USERNAME} $OPAMROOT

apt-get autoremove -y
apt-get clean -y
rm -rf /var/lib/apt/lists/*
