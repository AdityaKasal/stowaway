#!/bin/bash
cd /tmp
for v in mtp2 helper; do
  echo "=== none vs $v (story)"
  diff <(sed 's/\[ Prompt.*//' m35-story-none.out) <(sed 's/\[ Prompt.*//' m35-story-$v.out) | head -12
done
echo "=== first lines of none:"; head -c 600 m35-story-none.out
