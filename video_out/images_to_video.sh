#!/bin/bash
ffmpeg -r:v 12 -i "frame_%06d.png" -codec:v libx264 -preset veryslow -pix_fmt yuv420p -crf 28 -an "simple_cnn.mp4"
