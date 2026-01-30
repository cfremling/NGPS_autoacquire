## Build on Ubuntu

rm -rf build/
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

./build/ngps_acq --help
