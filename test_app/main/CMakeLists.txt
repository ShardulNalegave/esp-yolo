
set(c_srcs
    "main.cc"
    "image.cc"
)

idf_component_register(SRCS ${c_srcs}
                    INCLUDE_DIRS ".")