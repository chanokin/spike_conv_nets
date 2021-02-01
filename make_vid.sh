#! /bin/bash

ffmpeg -i sim_output_%010d.png -vcodec mpeg4 -y -r 2 movie.mp4
