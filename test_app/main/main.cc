
#include <esp_log.h>
#include <freertos/FreeRTOS.h>

#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include <esp_yolo.h>
#include <esp_yolo_model.h>
#include "image.h"

static const char* TAG = "esp_yolo_test";

namespace {
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;

    constexpr int kTensorArenaSize = 2000;
    uint8_t tensor_arena[kTensorArenaSize];
}

extern "C" void app_main(void)
{
    model = tflite::GetModel(esp_yolo_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model provided is schema version %d not equal to supported version %d.", model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    static tflite::MicroMutableOpResolver<1> resolver;
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        MicroPrintf("AllocateTensors() failed");
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    int tensor_size = input->dims->data[1] *  // height
        input->dims->data[2] *  // width
        input->dims->data[3];  // channels
    
    int zero_point = ((TfLiteAffineQuantization*)input->quantization.params)->zero_point->data[0];

    for (int i = 0; i < tensor_size; i++)
    {
        input->data.int8[i] = (int8_t)((image_raw[i] / 255.0) - zero_point);
    }
    

    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        MicroPrintf("Invoke failed");
        return;
    }

    ESP_LOGI(TAG, "Invoke was successful");
}