#ifndef TIMER_CUH
#define TIMER_CUH

#include <iostream>
#include <ctime>
#include <chrono>

class CPUTimer{
    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> begin;
        std::chrono::time_point<std::chrono::high_resolution_clock> end;

    public:
        CPUTimer(){};
        ~CPUTimer(){};

        void start(){
            begin = std::chrono::high_resolution_clock::now();
        };

        void stop(){
            end = std::chrono::high_resolution_clock::now();
        };

        float getDuration(){
            std::chrono::duration<double> difference = end - begin;
            std::chrono::microseconds mics = std::chrono::duration_cast<std::chrono::microseconds>(difference);
            return (float) mics.count();
        };
};

class GPUTimer{
    private:
        cudaStream_t stream;
        cudaEvent_t begin, stop;

    public:
        GPUTimer(){
            cudaEventCreate(&begin);
		    cudaEventCreate(&stop);
            cudaStreamCreate(&stream);
        };
        
        ~GPUTimer(){
            cudaEventDestroy(begin);
		    cudaEventDestroy(stop);
            cudaStreamDestroy(stream);
        };

        cudaStream_t getStream(){
            return stream;
        };

        void GPUstart(){
            cudaEventRecord(begin,stream);
        };

        void GPUstop(){
            cudaEventRecord(stop,stream);
            cudaEventSynchronize(stop);
        };

        float getGPUDuration(){
            float duration;
		    cudaEventElapsedTime(&duration, begin, stop);
		    return duration / 1000;
        };
};

#endif