#!/bin/sh
# Stage the CC0 MakeHuman data subset that import_makehuman.py bakes from.
# Everything copied here was explicitly released CC0 (headers in each file;
# LICENSE.ASSETS.md in the makehuman repo).
set -e
cd "$(dirname "$0")"
TMP=$(mktemp -d)
git clone --depth 1 https://github.com/makehumancommunity/makehuman.git "$TMP/mh"
mkdir -p mh-data
cp "$TMP/mh/makehuman/data/3dobjs/base.obj" mh-data/
cp -R "$TMP/mh/makehuman/data/rigs" mh-data/
cp -R "$TMP/mh/makehuman/data/targets/macrodetails" mh-data/
rm -rf "$TMP"
echo "staged $(du -sh mh-data | cut -f1) into mh-data/"
