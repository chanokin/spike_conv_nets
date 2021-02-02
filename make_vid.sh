#! /bin/bash

#ffmpeg -r 1 -i "test_img_sim_output_*.png" -vcodec mpeg4 -y -r 1 movie.mp4
cat test_img_sim_output_*.png | ffmpeg -f image2pipe -r 1 -i - -c:v copy -r 2 movie.mp4