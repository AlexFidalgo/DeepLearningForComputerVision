set -v
g++ -std=gnu++14 $1.cpp -o $1 -fmax-errors=2 `pkg-config opencv4 --libs --cflags` -O3 -s
set +v
