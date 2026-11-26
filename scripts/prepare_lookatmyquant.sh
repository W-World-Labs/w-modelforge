#!/usr/bin/env bash
# Copyright (c) 2026 W-World-Labs contributors. SPDX-License-Identifier: MIT
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
upstream="upstream/lookatmyquant"
patch="$root/patches/lookatmyquant.patch"
target="$root/methods/lookatmyquant"
pin="$(git -C "$root" rev-parse ":$upstream")"
source_id="$pin $(git -C "$root" hash-object "$patch")"

if [[ -e "$target" || -L "$target" ]]; then
    if [[ -f "$target/.w-modelforge-source" &&
          "$(<"$target/.w-modelforge-source")" == "$source_id" ]]; then
        exit 0
    fi
    echo "Move $target aside before preparing a different source revision." >&2
    exit 1
fi

git -C "$root" submodule update --init --checkout -- "$upstream"
if [[ "$(git -C "$root/$upstream" rev-parse HEAD)" != "$pin" ]]; then
    echo "LookAtMyQuant checkout does not match the recorded upstream commit: $pin" >&2
    exit 1
fi

mkdir -p -- "$root/methods"
staging="$(mktemp -d "$root/methods/.lookatmyquant.XXXXXX")"
trap 'rm -rf -- "$staging"' EXIT
git -C "$root/$upstream" archive "$pin" | tar -xf - -C "$staging"
GIT_CEILING_DIRECTORIES="$root/methods" git -C "$staging" apply -p2 --whitespace=nowarn "$patch"
printf '%s\n' "$source_id" > "$staging/.w-modelforge-source"
python3 -c 'import os, sys; os.rename(sys.argv[1], sys.argv[2])' "$staging" "$target"
