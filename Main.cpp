#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <chrono>
#include <thread>

#include "BitField.hpp"

#include "resize.hpp"
#include "toinvgrayscale.hpp"
#include "NNConvert.hpp"
#include "Conv2d.hpp"
#include "BatchNorm2d.hpp"
#include "ReLu.hpp"
#include "MaxPool2d.hpp"
#include "Bottleneck.hpp"
#include "Vector2.hpp"
#include "Normalize.hpp"
#include "Copy.hpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
    /* pytorch export into bit_field */
    /*
    struct bit_field nn;
    bit_field_init(&nn, 16, 256);
    read_into_bit_field_and_save(&nn, "./101seg_data/", "./101seg_data.position", "./101seg_data.data");
    */

    // read nn from file
    std::map<std::string, int> nn_positions = read_position_file("./101seg_data.position");

    struct bit_field nn;

    bit_field_load_from_disk(&nn, "./101seg_data.data");
    bit_field_register_device(&nn, 0);
    bit_field_update_device(&nn, 0);

    /*
    std::map<std::string, int>::iterator it = nn_positions.begin();
    // Iterate over the map using Iterator till end.
    while (it != nn_positions.end())
    {
        std::string word = it->first;
        int nn_position = it->second;
        int count = nn.data[nn_position];
        printf("name: %s\tcount: %d\n", word.c_str(), count);
        it++;
    }
    */

    //input layer
    struct conv2d bb_conv1;
    conv2d_init(&bb_conv1, 3, 64, struct vector2<unsigned int>(7, 7), struct vector2<unsigned int>(2, 2), struct vector2<unsigned int>(3, 3), struct vector2<unsigned int>(1, 1), nn_positions["backbone.conv1.weight"] + 1);
    struct vector3<unsigned int> bb_conv1_output_dim = conv2d_get_output_dim(&bb_conv1, struct vector2<unsigned int>(224, 224));

    struct bn2d bb_bn1;
    batchnorm2d_init(&bb_bn1, 64, nn_positions["backbone.bn1.weight"] + 1, nn_positions["backbone.bn1.bias"] + 1);

    struct maxpool2d bb_mp;
    maxpool2d_init(&bb_mp, 64, struct vector2<unsigned int>(3, 3), struct vector2<unsigned int>(2, 2), struct vector2<unsigned int>(1, 1), struct vector2<unsigned int>(1, 1));
    struct vector3<unsigned int> bb_mp_output_dim = maxpool2d_get_output_dim(&bb_mp, struct vector2<unsigned int>(bb_conv1_output_dim[0], bb_conv1_output_dim[1]));

    //layer 1
    struct bottleneck layer1_b_0;
    conv2d_init(&layer1_b_0.conv1, 64, 64, struct vector2<unsigned int>(1, 1), struct vector2<unsigned int>(1, 1), struct vector2<unsigned int>(0, 0), struct vector2<unsigned int>(1, 1), nn_positions["backbone.layer1.0.conv1.weight"] + 1);
    batchnorm2d_init(&layer1_b_0.bn1, 64, nn_positions["backbone.layer1.0.bn1.weight"] + 1, nn_positions["backbone.layer1.0.bn1.bias"] + 1);
    conv2d_init(&layer1_b_0.conv2, 64, 64, struct vector2<unsigned int>(3, 3), struct vector2<unsigned int>(1, 1), struct vector2<unsigned int>(1, 1), struct vector2<unsigned int>(1, 1), nn_positions["backbone.layer1.0.conv2.weight"] + 1);
    batchnorm2d_init(&layer1_b_0.bn2, 64, nn_positions["backbone.layer1.0.bn2.weight"] + 1, nn_positions["backbone.layer1.0.bn2.bias"] + 1);
    conv2d_init(&layer1_b_0.conv3, 64, 256, struct vector2<unsigned int>(1, 1), struct vector2<unsigned int>(1, 1), struct vector2<unsigned int>(0, 0), struct vector2<unsigned int>(1, 1), nn_positions["backbone.layer1.0.conv3.weight"] + 1);
    batchnorm2d_init(&layer1_b_0.bn3, 256, nn_positions["backbone.layer1.0.bn3.weight"] + 1, nn_positions["backbone.layer1.0.bn3.bias"] + 1);
    conv2d_init(&layer1_b_0.conv_ds, 64, 256, struct vector2<unsigned int>(1, 1), struct vector2<unsigned int>(1, 1), struct vector2<unsigned int>(0, 0), struct vector2<unsigned int>(1, 1), nn_positions["backbone.layer1.0.downsample.0.weight"] + 1);
    batchnorm2d_init(&layer1_b_0.bn_ds, 256, nn_positions["backbone.layer1.0.downsample.1.weight"] + 1, nn_positions["backbone.layer1.0.downsample.1.bias"] + 1);
    struct vector3<unsigned int> layer1_b_0_max_output_dim = bottleneck_get_max_size(&layer1_b_0, struct vector2<unsigned int>(bb_mp_output_dim[0], bb_mp_output_dim[1]));

    //:video input webcam
    string input_file_webcam("video.mp4");
    float fps_webcam = 60.00f;
    int crop_x1_webcam = 600;
    int crop_x2_webcam = 1240;
    int crop_y1_webcam = 150;
    int crop_y2_webcam = 510;
    int width_webcam = 640;
    int height_webcam = 360;

    //:video target
    int width_target = 640;
    int height_target = 360;
    float fps_targetf = 60.00f;
    int x1_webcam_target = 0;
    int y1_webcam_target = 0;

    //create window
    static const std::string kWinGName = "video target";
    namedWindow(kWinGName, WINDOW_NORMAL);
    resizeWindow(kWinGName, Size(width_target, height_target));

    for (int i = 0; i < 64; i++) {
        stringstream ss_;
        ss_ << i;
        std::string kWinFName = "feature_" + ss_.str();
        namedWindow(kWinFName, WINDOW_NORMAL);
        resizeWindow(kWinFName, Size(bb_mp_output_dim[0]*2, bb_mp_output_dim[1]*2));
        moveWindow(kWinFName, (i % 10) * (bb_mp_output_dim[0]*2 + 10), (i / 10) * (bb_mp_output_dim[1]*2 + 10));
    }

    //webcam capture
    VideoCapture cap;
    cap.open(input_file_webcam);
        
    Mat video_source;
    cap >> video_source;
    if (video_source.empty()) {
        exit(0);
    }
    int channels = video_source.channels();

    //input allocations
    struct bit_field bf_input;
    bit_field_init(&bf_input, 16, 1024);
    bit_field_register_device(&bf_input, 0);

    int video_source_size = video_source.cols * video_source.rows * channels;
    int video_source_size_in_bf = (int)ceilf(video_source_size / (float)sizeof(unsigned int));
    int video_source_position = bit_field_add_bulk_zero(&bf_input, video_source_size_in_bf)+1;

    unsigned char* video_source_in_bf = (unsigned char*)&(bf_input.data[video_source_position]);
    memcpy(video_source_in_bf, video_source.data, video_source_size);
    video_source.data = video_source_in_bf;

    //intermediate allocations
    struct bit_field bf_intermediate;
    bit_field_init(&bf_intermediate, 16, 1024);
    bit_field_register_device(&bf_intermediate, 0);

    int video_cr_size = width_webcam * height_webcam * channels;
    int video_cr_size_in_bf = (int)ceilf(video_cr_size / (float)sizeof(unsigned int));
    int video_cr_position = bit_field_add_bulk_zero(&bf_intermediate, video_cr_size_in_bf)+1;

    int video_nn_input_size = 224 * 224 * 3;
    int video_nn_input_size_in_bf = (int)ceilf(video_nn_input_size / (float)sizeof(unsigned int));
    int video_nn_input_position = bit_field_add_bulk_zero(&bf_intermediate, video_nn_input_size_in_bf)+1;

    int video_nn_input_normalized_size = 224 * 224 * 3 * sizeof(float);
    int video_nn_input_normalized_size_in_bf = (int)ceilf(video_nn_input_normalized_size / (float)sizeof(unsigned int));
    int video_nn_input_normalized_position = bit_field_add_bulk_zero(&bf_intermediate, video_nn_input_normalized_size_in_bf)+1;

    int bb_conv1_size = bb_conv1_output_dim[0] * bb_conv1_output_dim[1] * bb_conv1_output_dim[2] * sizeof(float);
    int bb_conv1_size_in_bf = (int)ceilf(bb_conv1_size / (float)sizeof(unsigned int));
    int bb_conv1_position = bit_field_add_bulk_zero(&bf_intermediate, bb_conv1_size_in_bf)+1;

    int bb_mp_size = bb_mp_output_dim[0] * bb_mp_output_dim[1] * bb_mp_output_dim[2] * sizeof(float);
    int bb_mp_size_in_bf = (int)ceilf(bb_mp_size / (float)sizeof(unsigned int));
    int bb_mp_position = bit_field_add_bulk_zero(&bf_intermediate, bb_mp_size_in_bf)+1;

    int layer1_b0_size = layer1_b_0_max_output_dim[0] * layer1_b_0_max_output_dim[1] * layer1_b_0_max_output_dim[2] * sizeof(float);

    int layer1_b0_size_in_bf = (int)ceilf(layer1_b0_size / (float)sizeof(unsigned int));
    int layer1_b0_position = bit_field_add_bulk_zero(&bf_intermediate, layer1_b0_size_in_bf)+1;
    int layer1_b0__position = bit_field_add_bulk_zero(&bf_intermediate, layer1_b0_size_in_bf)+1;

    bit_field_update_device(&bf_intermediate, 0);

    //output allocations
    struct bit_field bf_output;
    bit_field_init(&bf_output, 16, 1024);
    bit_field_register_device(&bf_output, 0);

    int video_output_size = width_target * height_target * 3;
    int video_output_size_in_bf = (int)ceilf(video_output_size / (float)sizeof(unsigned int));
    int video_output_position = bit_field_add_bulk_zero(&bf_output, video_output_size_in_bf)+1;

    //int feature_output_size = layer1_b_0.conv3_output_dim[0] * layer1_b_0.conv3_output_dim[1] * layer1_b_0.conv3_output_dim[2] * sizeof(float);
    //int feature_output_size = bb_conv1_output_dim[0] * bb_conv1_output_dim[1] * bb_conv1_output_dim[2] * sizeof(float);
    int feature_output_size = bb_mp_output_dim[0] * bb_mp_output_dim[1] * bb_mp_output_dim[2] * sizeof(float);
    int feature_output_size_in_bf = (int)ceilf(feature_output_size / (float)sizeof(unsigned int));
    int feature_output_position = bit_field_add_bulk_zero(&bf_output, feature_output_size_in_bf)+1;

    unsigned char* video_output_in_bf = (unsigned char*)&(bf_output.data[video_output_position]);
    Mat video_output = cv::Mat(height_target, width_target, CV_8UC3, video_output_in_bf);

    float* feature_output_in_bf = (float*)&(bf_output.data[feature_output_position]);
    stringstream ss;
    Mat feature_output[64];
    for (int i = 0; i < 64; i++) {
        //feature_output[i] = cv::Mat(layer1_b_0.conv3_output_dim[1], layer1_b_0.conv3_output_dim[0], CV_32FC1, &feature_output_in_bf[i * layer1_b_0.conv3_output_dim[0] * layer1_b_0.conv3_output_dim[1]]);
        feature_output[i] = cv::Mat(bb_mp_output_dim[1], bb_mp_output_dim[0], CV_32FC1, &feature_output_in_bf[i*bb_mp_output_dim[0]*bb_mp_output_dim[1]]);
        //feature_output[i] = cv::Mat(bb_conv1_output_dim[1], bb_conv1_output_dim[0], CV_32FC1, &feature_output_in_bf[i * bb_conv1_output_dim[0] * bb_conv1_output_dim[1]]);
    }

    bit_field_update_device(&bf_output, 0);

    double time_step = 1000.0 / fps_targetf;
    long long time_start = _Query_perf_counter();
    long long time_now = time_start;
    long long time_last = time_now;

    while (waitKey(1) < 0) {
        //updating video source
        bit_field_invalidate_bulk(&bf_input, video_source_position, video_source_size_in_bf);
        bit_field_update_device(&bf_input, 0);

        //resize source to desired output
        launch_resize(bf_input.device_data[0], 
                        video_source_position, 
                            video_source.cols, video_source.rows, channels,
                            crop_x1_webcam, crop_x2_webcam, crop_y1_webcam, crop_y2_webcam, 
                        bf_intermediate.device_data[0], video_cr_position, 
                            width_webcam, height_webcam);

        //resize and normalize desired output to nn_input
        launch_resize(bf_intermediate.device_data[0], video_cr_position, width_webcam, height_webcam, channels, 0, width_target, 0, height_target, bf_intermediate.device_data[0], video_nn_input_position, 224, 224);
        launch_normalize(bf_intermediate.device_data[0], video_nn_input_position, 224, 224, 3, bf_intermediate.device_data[0], video_nn_input_normalized_position, struct vector3<float> (0.485f, 0.456f, 0.406f), struct vector3<float>(0.229f, 0.224f, 0.225f));

        //run nn
        //input layer
        launch_conv2d(nn.device_data[0], bb_conv1.parameters_position, bf_intermediate.device_data[0], video_nn_input_normalized_position, 224, 224, bf_intermediate.device_data[0], bb_conv1_position, &bb_conv1);
        //launch_copy(bf_intermediate.device_data[0], bb_conv1_position, bb_conv1_output_dim[0], bb_conv1_output_dim[1], bb_conv1_output_dim[2], bf_output.device_data[0], feature_output_position, bb_conv1_output_dim[0], bb_conv1_output_dim[1], bb_conv1_output_dim[2], 0, 0, 255, 1);
        launch_batchnorm2d(nn.device_data[0], bb_bn1.position_weights, bb_bn1.position_bias, bf_intermediate.device_data[0], bb_conv1_position, bb_conv1_output_dim[0], bb_conv1_output_dim[1], bf_intermediate.device_data[0], bb_conv1_position, bb_conv1_output_dim[2]);
        launch_relu(bf_intermediate.device_data[0], bb_conv1_position, bb_conv1_output_dim[0], bb_conv1_output_dim[1], bf_intermediate.device_data[0], bb_conv1_position, bb_conv1_output_dim[2]);
        launch_maxpool2d(bf_intermediate.device_data[0], bb_conv1_position, bb_conv1_output_dim[0], bb_conv1_output_dim[1], bf_intermediate.device_data[0], bb_mp_position, &bb_mp);

        //layer 1 bottleneck 0
        launch_conv2d(nn.device_data[0], layer1_b_0.conv1.parameters_position, 
            bf_intermediate.device_data[0], bb_mp_position, bb_mp_output_dim[0], bb_mp_output_dim[1], 
            bf_intermediate.device_data[0], layer1_b0_position, &layer1_b_0.conv1);
        launch_batchnorm2d(nn.device_data[0], layer1_b_0.bn1.position_weights, layer1_b_0.bn1.position_bias, 
            bf_intermediate.device_data[0], layer1_b0_position, layer1_b_0.conv1_output_dim[0], layer1_b_0.conv1_output_dim[1], 
            bf_intermediate.device_data[0], layer1_b0_position, layer1_b_0.conv1_output_dim[2]);
        launch_conv2d(nn.device_data[0], layer1_b_0.conv2.parameters_position, 
            bf_intermediate.device_data[0], layer1_b0_position, layer1_b_0.conv1_output_dim[0], layer1_b_0.conv1_output_dim[1], 
            bf_intermediate.device_data[0], layer1_b0__position, &layer1_b_0.conv2);
        launch_batchnorm2d(nn.device_data[0], layer1_b_0.bn2.position_weights, layer1_b_0.bn2.position_bias, 
            bf_intermediate.device_data[0], layer1_b0__position, layer1_b_0.conv2_output_dim[0], layer1_b_0.conv2_output_dim[1], 
            bf_intermediate.device_data[0], layer1_b0__position, layer1_b_0.conv2_output_dim[2]);
        launch_conv2d(nn.device_data[0], layer1_b_0.conv3.parameters_position, 
            bf_intermediate.device_data[0], layer1_b0__position, layer1_b_0.conv2_output_dim[0], layer1_b_0.conv2_output_dim[1], 
            bf_intermediate.device_data[0], layer1_b0_position, &layer1_b_0.conv3);
        launch_batchnorm2d(nn.device_data[0], layer1_b_0.bn3.position_weights, layer1_b_0.bn3.position_bias, 
            bf_intermediate.device_data[0], layer1_b0_position, layer1_b_0.conv3_output_dim[0], layer1_b_0.conv3_output_dim[1], 
            bf_intermediate.device_data[0], layer1_b0_position, layer1_b_0.conv3_output_dim[2]);
        launch_conv2d(nn.device_data[0], layer1_b_0.conv_ds.parameters_position, bf_intermediate.device_data[0], bb_mp_position, bb_mp_output_dim[0], bb_mp_output_dim[1], bf_intermediate.device_data[0], layer1_b0__position, &layer1_b_0.conv_ds);
        launch_batchnorm2d(nn.device_data[0], layer1_b_0.bn_ds.position_weights, layer1_b_0.bn_ds.position_bias, bf_intermediate.device_data[0], layer1_b0__position, layer1_b_0.conv3_output_dim[0], layer1_b_0.conv3_output_dim[1], bf_intermediate.device_data[0], layer1_b0__position, layer1_b_0.conv3_output_dim[2]);
        launch_add(bf_intermediate.device_data[0], layer1_b0__position, layer1_b_0.conv3_output_dim[0], layer1_b_0.conv3_output_dim[1], layer1_b_0.conv3_output_dim[2], bf_intermediate.device_data[0], layer1_b0_position, layer1_b_0.conv3_output_dim[0], layer1_b_0.conv3_output_dim[1], layer1_b_0.conv3_output_dim[2], 1);
        launch_relu(bf_intermediate.device_data[0], layer1_b0_position, layer1_b_0.conv3_output_dim[0], layer1_b_0.conv3_output_dim[1], bf_intermediate.device_data[0], layer1_b0_position, layer1_b_0.conv3_output_dim[2]);

        //output
        //launch_copy(bf_intermediate.device_data[0], layer1_b0_position, layer1_b_0.conv3_output_dim[0], layer1_b_0.conv3_output_dim[1], layer1_b_0.conv3_output_dim[2], bf_output.device_data[0], feature_output_position, layer1_b_0.conv3_output_dim[0], layer1_b_0.conv3_output_dim[1], layer1_b_0.conv3_output_dim[2], 0, 0, 255, 1);
        launch_copy(bf_intermediate.device_data[0], bb_mp_position, bb_mp_output_dim[0], bb_mp_output_dim[1], bb_mp_output_dim[2], bf_output.device_data[0], feature_output_position, bb_mp_output_dim[0], bb_mp_output_dim[1], bb_mp_output_dim[2], 0, 0, 255, 1);
        launch_copy(bf_intermediate.device_data[0], video_cr_position, width_webcam, height_webcam, channels, bf_output.device_data[0], video_output_position, width_target, height_target, 3, x1_webcam_target, y1_webcam_target, 255, 0);

        bit_field_update_host(&bf_output, 0);

        /*
        for (int y = 0; y < bb_conv1_output_dim[1]; y++) {
            for (int x = 0; x < bb_conv1_output_dim[0]; x++) {
                printf("%0.2f ", feature_output_in_bf[y * (bb_conv1_output_dim[0]) + x]);
            }
            printf("\n");
        }*/

        imshow(kWinGName, video_output);
        for (int i = 0; i < 64; i++) {
            stringstream ss_;
            ss_ << i;
            std::string kWinFName = "feature_" + ss_.str();
            imshow(kWinFName, feature_output[i]);
        }

        cap >> video_source;
        if (video_source.empty()) {
            printf("total_time: %lf\n", (time_now - time_start)*1000.0/(getTickFrequency()));
            waitKey();
            break;
        }
    }
}

