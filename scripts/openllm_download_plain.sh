#!/usr/bin/env bash
# Plain resumable download of a public HF repo via hf-mirror into a clean dir.
# Usage: openllm_download_plain.sh <repo_id> <dest_dir>
set -u
REPO_ID="$1"; DEST="$2"
mkdir -p "$DEST"
API="https://hf-mirror.com/api/models/$REPO_ID"
SHA=$(curl -s -m 30 ${CURL_NOPROXY:---noproxy '*'} "$API" | python3 -c "import sys,json;print(json.load(sys.stdin)['sha'])")
FILES=$(curl -s -m 30 ${CURL_NOPROXY:---noproxy '*'} "$API" | python3 -c "
import sys,json
d=json.load(sys.stdin)
keep=('config.json','generation_config.json','tokenizer.json','tokenizer_config.json','tokenizer.model','special_tokens_map.json','vocab.json','merges.txt','.gitattributes','README.md')
for s in d['siblings']:
    f=s['rfilename']
    if f.endswith(('.safetensors','.bin')) or f in keep:
        print(f)
")
echo "[$REPO_ID] sha=$SHA files=$(echo "$FILES" | wc -l)"
echo "$FILES" | xargs -P 6 -I{} bash -c '
  f="{}"; out="'"$DEST"'/$f"
  mkdir -p "$(dirname "$out")"
  url="https://hf-mirror.com/'"$REPO_ID"'/resolve/'"$SHA"'/$f"
  for a in 1 2 3 4 5; do
    curl -sL ${CURL_NOPROXY:---noproxy "*"} -C - --retry 3 --retry-delay 2 -m 7200 -o "$out" "$url" && break
    echo "retry $a $f"; sleep 3
  done
  sz=$(stat -c%s "$out" 2>/dev/null || echo 0)
  echo "done $f $((sz/1048576))MB"
'
echo "[$REPO_ID] COMPLETE"
