
#ifndef ESP_YOLO_H
#define ESP_YOLO_H

#include <stdint.h>

typedef struct esp_yolo_box {
    uint16_t x1;
    uint16_t y1;
    uint16_t x2;
    uint16_t y2;
    float score;
    uint8_t class_id;
};

void preprocess_raw_img(unsigned char raw_img[]);

#endif