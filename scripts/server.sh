#!/bin/sh
set -eu
set -x

case "$1" in
start)
    shift
    $@ >log 2>&1 &
    for i in $(seq 1 10); do
        pid=$(pgrep http-server || true)
        if [ -n "$pid" ]; then
            cat log
            exit 0
        fi
        sleep 1
    done
    exit 1
    ;;
stop)
    pid=$(pgrep http-server)
    kill $pid
    wait $pid
    ;;
*)
    echo "Not supported '$@'" >&2
    exit 1
    ;;
esac