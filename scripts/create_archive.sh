#!/usr/bin/env bash
# 将 diffold 项目完整打包压缩
set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PARENT="$(dirname "$ROOT")"
NAME="diffold"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="${ROOT}/archives"
mkdir -p "$OUT_DIR"
TMP_ARCHIVE=$(mktemp -u /tmp/diffold_archive_XXXXXX.tar.gz)

echo "==> 完整打包（约 20GB，请耐心等待）..."
ARCHIVE="${OUT_DIR}/${NAME}_full_${TIMESTAMP}.tar.gz"
tar czvf "$TMP_ARCHIVE" -C "$PARENT" "$NAME" && mv "$TMP_ARCHIVE" "$ARCHIVE"

echo ""
echo "==> 已生成: $ARCHIVE"
ls -lh "$ARCHIVE"
