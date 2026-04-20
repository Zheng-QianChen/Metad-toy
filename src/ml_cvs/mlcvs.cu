#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>

void mlcvs_test() {
    // 1. 初始化环境
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "GPU_Test");

    // 2. 获取所有可用的执行提供者 (Providers)
    std::vector<std::string> providers = Ort::GetAvailableProviders();

    std::cout << "--- ONNX Runtime Available Providers ---" << std::endl;
    bool cuda_available = false;
    for (const auto& name : providers) {
        std::cout << "[Provider]: " << name << std::endl;
        if (name == "CUDAExecutionProvider") {
            cuda_available = true;
        }
    }
    std::cout << "---------------------------------------" << std::endl;

    if (cuda_available) {
        std::cout << "SUCCESS: CUDA Execution Provider is available!" << std::endl;

        // 尝试创建一个带 CUDA 选项的 SessionOptions 以进一步确认
        try {
            Ort::SessionOptions session_options;
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0; // 使用第一个 GPU
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "SUCCESS: Successfully appended CUDA provider options." << std::endl;
        } catch (const std::exception& e) {
            std::cout << "ERROR during CUDA options setup: " << e.what() << std::endl;
        }
    } else {
        std::cout << "FAILURE: CUDA Execution Provider NOT found." << std::endl;
        std::cout << "Check if LD_LIBRARY_PATH includes the 'lib' directory and CUDA is installed." << std::endl;
    }
}