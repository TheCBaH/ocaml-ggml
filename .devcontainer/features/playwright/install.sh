#!/bin/sh
set -eu
set -x

echo "Activating feature 'Playwright'"
BROWSERS=${BROWSERS:-$@}
USERNAME="${USERNAME:-"${_REMOTE_USER:-"automatic"}"}"
su ${USERNAME} -c "umask 0002 && . '${NVM_DIR}/nvm.sh' && npm install --global playwright"
su ${USERNAME} -c "umask 0002 && . '${NVM_DIR}/nvm.sh' && npx playwright install --with-deps ${BROWSERS}"
