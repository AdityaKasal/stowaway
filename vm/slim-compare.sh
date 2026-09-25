#!/bin/bash
cd /tmp
for f in ref-big ref-tiny slim-big slim-tiny; do
  grep -v -E 't/s|Loading model|Exiting|^[[:space:]]*$' $f.txt | grep -v -E '^[^a-zA-Z0-9]*$' > $f.ans
  echo "$f: $(wc -c < $f.ans) bytes, $(md5sum < $f.ans | cut -c1-12)"
done
cmp -s ref-big.ans slim-big.ans && echo "big cache: identical before/after slim" || echo "big cache: DIFFERENT"
cmp -s ref-tiny.ans slim-tiny.ans && echo "tiny cache: identical before/after slim" || echo "tiny cache: DIFFERENT"
cmp -s ref-big.ans ref-tiny.ans && echo "big vs tiny cache: identical" || { echo "big vs tiny cache: DIFFERENT"; diff ref-big.ans ref-tiny.ans | head -8; }
grep -h "didn't fit" slim-tiny.txt.err ref-tiny.txt.err
echo "--- answer:"; head -c 600 slim-tiny.ans; echo
