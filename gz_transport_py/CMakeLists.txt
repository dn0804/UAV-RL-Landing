cmake_minimum_required(VERSION 3.12)
project(gz_transport_py)

# ── Dependencies ──

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(gz-transport12 REQUIRED)

# Garden = gz-msgs9.  If yours differs, cmake will say which
# version it found — just change the number below.
find_package(gz-msgs9 REQUIRED)

# ── Build ──

pybind11_add_module(gz_transport_py gz_transport_py.cpp)

target_link_libraries(gz_transport_py PRIVATE
    gz-transport12::gz-transport12
    gz-msgs9::gz-msgs9
)

# ── Install (optional — you can also just copy the .so) ──

install(TARGETS gz_transport_py
    LIBRARY DESTINATION "${Python3_SITELIB}"
)
