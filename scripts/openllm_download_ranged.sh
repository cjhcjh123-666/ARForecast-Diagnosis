#!/usr/bin/env bash
# All-chunks-parallel resumable download from hf-mirror.
# Usage: openllm_download_ranged.sh <repo_id> <dest_dir> [chunks] [parallel]
set -u
REPO_ID="$1"; DEST="$2"; CH=${3:-8}; PAR=${4:-32}
mkdir -p "$DEST"
API="https://hf-mirror.com/api/models/$REPO_ID"
META=$(curl -s -m 30 --noproxy '*' "$API")
SHA=$(echo "$META" | python3 -c "import sys,json;print(json.load(sys.stdin)['sha'])")
echo "$META" | python3 -c "
import sys,json
d=json.load(sys.stdin)
keep=('config.json','generation_config.json','tokenizer.json','tokenizer_config.json','tokenizer.model','special_tokens_map.json','vocab.json','merges.txt','.gitattributes','README.md')
for s in d['siblings']:
    f=s['rfilename']
    if f.endswith(('.safetensors','.bin','index.json')) or f in keep: print(f, s.get('size',0))
" > "$DEST/.filelist"
echo "[$REPO_ID] sha=$SHA files=$(wc -l < $DEST/.filelist)"
# small files first
while read -r f sz; do
  [ -z "$f" ] && continue; [ "$sz" -ge 104857600 ] && continue
  out="$DEST/$f"; mkdir -p "$(dirname "$out")"
  [ -f "$out" ] && [ "$(stat -c%s "$out")" = "$sz" ] && continue
  curl -sL --noproxy '*' -C - --retry 5 --retry-delay 3 -m 3600 -o "$out" "https://hf-mirror.com/$REPO_ID/resolve/$SHA/$f" || true
done < "$DEST/.filelist"
# build chunk job list for big files
: > "$DEST/.jobs"
while read -r f sz; do
  [ -z "$f" ] && continue; [ "$sz" -lt 104857600 ] && continue
  out="$DEST/$f"; mkdir -p "$(dirname "$out")"
  if [ -f "$out" ] && [ "$(stat -c%s "$out")" = "$sz" ]; then echo "skip $f"; continue; fi
  for i in $(seq 0 $((CH-1))); do
    start=$(( sz*i/CH )); end=$(( sz*(i+1)/CH - 1 ))
    echo "$f $i $start $end" >> "$DEST/.jobs"
  done
done < "$DEST/.filelist"
cat "$DEST/.jobs" | xargs -P "$PAR" -L1 bash -c '
  set -u; f="$0"; i="$1"; start="$2"; end="$3"
  out="'"$DEST"'/$f"; pf="$out.part$i"; want=$(( end-start+1 ))
  [ -f "$pf" ] && [ "$(stat -c%s "$pf")" = "$want" ] && exit 0
  curl -sL --noproxy "*" -r $start-$end --retry 6 --retry-delay 3 -m 7200 -o "$pf" "https://hf-mirror.com/'"$REPO_ID"'/resolve/'"$SHA"'/$f" || true
'
# reassemble
while read -r f sz; do
  [ -z "$f" ] && continue; [ "$sz" -lt 104857600 ] && continue
  out="$DEST/$f"
  if [ -f "$out" ] && [ "$(stat -c%s "$out")" = "$sz" ]; then echo "ok $f"; continue; fi
  cat $(for i in $(seq 0 $((CH-1))); do echo "$out.part$i"; done) > "$out" 2>/dev/null
  got=$(stat -c%s "$out" 2>/dev/null||echo 0)
  if [ "$got" = "$sz" ]; then rm -f "$out".part*; echo "done $f $((got/1048576))MB"; else echo "INCOMPLETE $f $((got/1048576))/$((sz/1048576))MB"; fi
done < "$DEST/.filelist"
echo "[$REPO_ID] ALL_DONE"
