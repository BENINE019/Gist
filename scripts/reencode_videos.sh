#!/usr/bin/env bash
# Re-encode every simulator video under out/ to browser-playable H.264 (idempotent).
set -u
n=0
for f in $(find out -name episode.mp4); do
  codec=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "$f")
  if [ "$codec" != "h264" ]; then
    ffmpeg -loglevel error -y -i "$f" -c:v libx264 -pix_fmt yuv420p -crf 23 -movflags +faststart "${f%.mp4}_h264.mp4" && mv "${f%.mp4}_h264.mp4" "$f"
    n=$((n+1))
  fi
done
echo "re-encoded $n video(s)"
