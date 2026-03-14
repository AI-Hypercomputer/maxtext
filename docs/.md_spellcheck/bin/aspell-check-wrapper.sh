#!/bin/bash
# Wrapper to restore terminal for interactive use
exec < /dev/tty > /dev/tty 2>&1
exec "$(dirname "$0")/aspell-check.sh" "$@"
