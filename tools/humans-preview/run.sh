#!/bin/sh
# swift build + run for humans-preview, working around SPM-CLI not compiling
# .metal resources: builds default.metallib into the VisualizerRendering
# resource bundle so makeDefaultLibrary(bundle:) succeeds outside Xcode.
set -e
cd "$(dirname "$0")/../.."
swift build --product humans-preview
BUNDLE=.build/arm64-apple-macosx/debug/VisualizerEngine_VisualizerRendering.bundle
if [ ! -f "$BUNDLE/default.metallib" ] || [ -n "$(find VisualizerRendering/Sources/VisualizerRendering/Shaders -name '*.metal' -newer "$BUNDLE/default.metallib" 2>/dev/null)" ]; then
    echo "compiling default.metallib..."
    AIR=$(mktemp -d)
    for f in "$BUNDLE"/*.metal; do
        xcrun -sdk macosx metal -c "$f" -o "$AIR/$(basename "$f" .metal).air" -I "$BUNDLE" &
    done
    wait
    xcrun -sdk macosx metallib "$AIR"/*.air -o "$BUNDLE/default.metallib"
    rm -rf "$AIR"
fi
exec .build/arm64-apple-macosx/debug/humans-preview "$@"
