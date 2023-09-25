#include <stdint.h>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "../stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb/stb_image_write.h"
#include "../NeuralNetLite/Linear.hpp"
#include "../NeuralNetLite/MatUtils.hpp"
#include <iostream>
#include <vector>
#include <ctime>
#include <random>
#include <cstdlib>

int main(){
    int width, height, bpp;

    uint8_t* data = stbi_load("/home/mpradyumna/Documents/GitHub/NeuralNetLite/image.png", &width, &height, &bpp, 3);

    std::cout<<width<<" "<<height<<" "<<bpp<<"\n";

    unsigned bytePerPixel = 1;
    size_t pixels = width*height;
    std::vector<double_t> red_channel;
    std::vector<double_t> green_channel;
    std::vector<double_t> blue_channel;
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            unsigned char* pixelOffset = data + (i + width * j) * 3;
            unsigned int r = static_cast<unsigned int>(pixelOffset[0]);
            unsigned int g = static_cast<unsigned int>(pixelOffset[1]);
            unsigned int b = static_cast<unsigned int>(pixelOffset[2]);
            red_channel.push_back(r);
            blue_channel.push_back(b);
            green_channel.push_back(g);
            // std::cout<<"("<<static_cast<unsigned int>(r)<<","<<static_cast<unsigned int>(g)<<","<<static_cast<unsigned int>(b)<<"),";
        }
        std::cout<<"\n";
    }

    std::vector<std::vector<double_t>> channels;
    channels.push_back(red_channel);
    channels.push_back(green_channel);
    channels.push_back(blue_channel);
    std::vector<std::vector<double_t>> conv_channels;
    for(auto channel: channels){
        TensorLite inputs = TensorLite({1, 750, 579});
        inputs.setData(channel);
        TensorLite kernel = TensorLite({1, 3, 3});
        kernel.setData({1, 1, 1, 1, 1, 1, 1, 1, 1});
        kernel.multiplyByConstantInPlace(0.11111111111);
        Conv convolution_layer({579, 750, 1}, {3, 3, 1});
        convolution_layer.setKernel(kernel);
        TensorLite output = convolution_layer.forward(inputs);
        output.print();
        conv_channels.push_back(output.data);
    }

    

    // Save processed image
    // Fill the processed_data array with the rgb values
    uint8_t* processed_data = new uint8_t[748 * 577 * bpp];
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int index = i * width + j;
            processed_data[index * 3] = static_cast<uint8_t>(conv_channels[0][index]);
            processed_data[index * 3 + 1] = static_cast<uint8_t>(conv_channels[1][index]);
            processed_data[index * 3 + 2] = static_cast<uint8_t>(conv_channels[2][index]);
        }
    }

    // Save the processed image as a new PNG file
    stbi_write_jpg("/home/mpradyumna/Documents/GitHub/NeuralNetLite/processed_image.png", 748,577, 3,  processed_data, 100);
    return 0;
}