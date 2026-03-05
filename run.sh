#!/usr/bin/env bash
# VoxelKit run script
# Usage: ./run.sh process <input.mov> [options]
#        ./run.sh info <input.botmap>
#        ./run.sh build        — build only
#        ./run.sh test         — run tests

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CMD="${1:-build}"

case "$CMD" in
    build)
        swift build -c release
        echo "Build complete: .build/release/voxelcli"
        ;;
    test)
        swift test
        ;;
    process|info|stream)
        swift build -c release 2>/dev/null
        .build/release/voxelcli "$@"
        ;;
    *)
        swift build -c release 2>/dev/null
        .build/release/voxelcli "$@"
        ;;
esac
