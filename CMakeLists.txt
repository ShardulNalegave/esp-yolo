cmake_minimum_required(VERSION 3.5)

set(c_srcs
    "src/esp_yolo.c"
    "src/esp_yolo_model.cc"
)

set(priv_includes "src")

idf_component_register(
    SRCS ${c_srcs}
    INCLUDE_DIRS "include"
    PRIV_INCLUDE_DIRS ${priv_includes}
)