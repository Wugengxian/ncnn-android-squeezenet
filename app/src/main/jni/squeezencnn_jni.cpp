// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>

// ncnn
#include "net.h"
#include "benchmark.h"

#include "squeezenet_v1.1.id.h"

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

static ncnn::Net squeezenet;
static ncnn::Net squeezenetint8;

static std::vector<std::string> split_string(const std::string &str, const std::string &delimiter) {
    std::vector<std::string> strings;

    std::string::size_type pos = 0;
    std::string::size_type prev = 0;
    while ((pos = str.find(delimiter, prev)) != std::string::npos) {
        strings.push_back(str.substr(prev, pos - prev));
        prev = pos + 1;
    }

    // To get the last substring (or only, if delimiter is not found)
    strings.push_back(str.substr(prev));

    return strings;
}

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "JNI_OnLoad");

    ncnn::create_gpu_instance();

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM *vm, void *reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "JNI_OnUnload");

    ncnn::destroy_gpu_instance();
}

// public native boolean Init(AssetManager mgr);
JNIEXPORT jboolean JNICALL
Java_com_tencent_squeezencnn_SqueezeNcnn_Init(JNIEnv *env, jobject thiz, jobject assetManager) {
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 4;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;

    // use vulkan compute
    if (ncnn::get_gpu_count() != 0)
        opt.use_vulkan_compute = true;

    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);

    squeezenet.opt = opt;
    squeezenetint8.opt = opt;

    // init param
    {
        int ret = squeezenet.load_param(mgr, "icnet.param");
        squeezenetint8.load_param(mgr, "icnet-int8.param");
        if (ret != 0) {
            __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "load_param_bin failed");
            return JNI_FALSE;
        }
    }

    // init bin
    {
        int ret = squeezenet.load_model(mgr, "icnet.bin");
        squeezenetint8.load_model(mgr, "icnet-int8.bin");
        if (ret != 0) {
            __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "load_model failed");
            return JNI_FALSE;
        }
    }

    return JNI_TRUE;
}

void draw_objects(JNIEnv *env, ncnn::Mat map, jobject bgr) {
    AndroidBitmapInfo bmpInfo = {0};
    if (AndroidBitmap_getInfo(env, bgr, &bmpInfo) < 0)
        return;
    int *dataFromBmp = NULL;
    if (AndroidBitmap_lockPixels(env, bgr, (void **) &dataFromBmp))
        return;

    const uint8_t color[] = {128, 255, 128, 244, 35, 232};
    const uint8_t color_count = sizeof(color) / sizeof(int);
    int alpha = 0xFF << 24;
    int width = map.w;
    int height = map.h;
    int size = map.c;
    int img_index2 = 0;
    float threshold = 0.45;
    const float *ptr2 = map;
    int red;
    int green;
    int blue;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float maxima = threshold;
            int index = -1;
            for (int c = 0; c < size; c++) {
                //const float* ptr3 = map.channel(c);
                const float *ptr3 = ptr2 + c * width * height;
                if (ptr3[img_index2] > maxima) {
                    maxima = ptr3[img_index2];
                    index = c;
                }
            }
            if (index > -1) {
                int color_index = (index) * 3;
                if (color_index < color_count) {
                    uint8_t b = color[color_index];
                    uint8_t g = color[color_index + 1];
                    uint8_t r = color[color_index + 2];
                    int color = dataFromBmp[img_index2];
                    red = ((color & 0x00FF0000) >> 17) + (r>>1);
                    green = ((color & 0x0000FF00) >> 9) + (g>>1);
                    blue = ((color & 0x000000FF) >> 1) + (b>>1);
                    dataFromBmp[img_index2] = alpha | (red << 16) | (green << 8) | blue;
                }
            }
            img_index2++;
        }
    }
    AndroidBitmap_unlockPixels(env,bgr);
}

// public native String Detect(Bitmap bitmap, boolean use_gpu);
JNIEXPORT jstring JNICALL
Java_com_tencent_squeezencnn_SqueezeNcnn_Detect(JNIEnv *env, jobject thiz, jobject bitmap,
                                                jboolean use_gpu, jboolean int8) {
    if (use_gpu == JNI_TRUE && ncnn::get_gpu_count() == 0) {
        return NULL;
    }

    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
    int width = info.width;
    int height = info.height;
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return NULL;

    // ncnn from bitmap
    ncnn::Mat in = ncnn::Mat::from_android_bitmap_resize(env, bitmap, ncnn::Mat::PIXEL_BGR, 640, 480);

    // squeezenet
    char *tmp;
    std::vector<float> cls_scores;
    {
        const float mean_vals[3] = {104.f, 117.f, 123.f};
        const float norm_vals[3] = {0.017f, 0.017f, 0.017f};
        in.substract_mean_normalize(mean_vals, norm_vals);
        double start_time = ncnn::get_current_time();
        ncnn::Extractor ex = squeezenet.create_extractor();
        if (int8 == JNI_FALSE) {
            ex = squeezenetint8.create_extractor();
        }

        ex.set_vulkan_compute(use_gpu);

        ex.input("input", in);

        ncnn::Mat out;
        ex.extract("output", out);
        double elasped = ncnn::get_current_time() - start_time;
        __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "%.2fms   detect", elasped);
        sprintf(tmp, "耗时：%.3f ms", elasped);
        
        ncnn::Mat seg_out;
        ncnn::resize_bilinear(out, seg_out, width, height);

        draw_objects(env, seg_out, bitmap);
    }
    std::string result_str = tmp;
    jstring result = env->NewStringUTF(result_str.c_str());

    return result;
}

}
