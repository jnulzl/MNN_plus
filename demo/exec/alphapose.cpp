//
// Created by lizhaoliang-os on 2020/6/9.
//

#include <iostream>
#include <string>
#include <memory>

#include <MNN/Interpreter.hpp>

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Usage: ./multiPose.out model.mnn input.jpg pose.jpg" << std::endl;
    }

    //1. Variables
    std::unique_ptr<MNN::Interpreter> net_;
    MNN::Session* session_;
    std::string model_path_ = "";
    std::string input_name_ = "";
    std::string output_name_ = "";
    int net_inp_channels = 1;
    int net_inp_height = 160;
    int net_inp_width = 160;


    //2. Init
    net_ = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_path_.c_str());
    MNN::ScheduleConfig netConfig;
    netConfig.type      = MNN_FORWARD_CPU;
    netConfig.numThread = 4;
    session_        = net_->createSession(netConfig);

    MNN::Tensor* inputTensor_tmp = net_->getSessionInput(session_, input_name_.c_str());

    if (inputTensor_tmp->elementSize() <= 4)
    {
        net_->resizeTensor(inputTensor_tmp, {1, net_inp_channels, net_inp_height, net_inp_width});
        net_->resizeSession(session_);
    }

    //3. process

    std::vector<int> inp_shape = {1, net_inp_channels, net_inp_height, net_inp_width};
    float* data_in_ = nullptr; // processed data nchw
    MNN::Tensor* nchw_Tensor = MNN::Tensor::create<float>(inp_shape, data_in_, MNN::Tensor::CAFFE);
    MNN::Tensor* inputTensor  = net_->getSessionInput(session_,  input_name_.c_str());
    inputTensor->copyFromHostTensor(nchw_Tensor);

    //4. run network
    net_->runSession(session_);

    //5. get output
    MNN::Tensor* out_tensor = net_->getSessionOutput(session_, output_name_.c_str());
    MNN::Tensor out(out_tensor, MNN::Tensor::CAFFE);
    out_tensor->copyToHostTensor(&out);

    std::cout << "batch:    " << out_tensor->batch()    << std::endl
              << "channels: " << out_tensor->channel()  << std::endl
              << "height:   " << out_tensor->height()   << std::endl
              << "width:    " << out_tensor->width()    << std::endl
              << "type:     " << out_tensor->getDimensionType() << std::endl;

    //out.host<float>() out.width(), out.height(), out.channel()

    return 0;
}
