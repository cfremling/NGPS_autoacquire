## Build on Ubuntu

### Install dependencies
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config \
  libcfitsio-dev wcslib-dev

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

./build/ngps_acq --help
