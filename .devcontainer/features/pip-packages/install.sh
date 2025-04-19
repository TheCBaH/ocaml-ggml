#!/bin/sh
set -e
set -x

echo "Activating feature 'pip_packages'"

PACKAGES=${PACKAGES:-$@}
OPTIONS=${OPTIONS:-$@}
echo "Selected packages: $PACKAGES with ${OPTIONS}"

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

check_packages python3 python3-pip

python3 -m pip config set global.break-system-packages true
python3 -m pip --no-cache-dir install $PACKAGES $OPTIONS
