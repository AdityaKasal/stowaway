#!/bin/bash
# Delete the model copies made for the VM tests (originals live on the Windows drives), then mark the space free.
echo "before: $(du -sh /root/moe/models 2>/dev/null | cut -f1) in /root/moe/models"
rm -rf /root/moe/models /root/mtpcheck
echo "deleted; /root/moe now: $(du -sh /root/moe | cut -f1)"
fstrim -v / 2>&1 | tail -1
df -h / | tail -1
