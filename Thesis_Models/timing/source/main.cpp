#include <chrono>
#include <iostream>
#include <torch/script.h>
#include <vector>
#include <string>

#include "main.h"
#include "test.cpp"

/*
 * main()
 * function to drive the overall timing system
 */
int main() {

    time_fitting(100000);

    time_ml("../../model.pt", 100000);
    time_ml("../../long_model.pt", 100000);

    std::cout << testfunc(2) << std::endl;

    return 0;
}

/*
 * function(double params[10], double mu_exp, double a_aicd, double flow_exp)
 * params: array of ten parameters holding [rate, oil_fraction, rhoo, muo, gas fraction, ...
 *          rhog, mug, water_fraction, rhow, muw]
 * mu_exp: viscosity exponent
 * a_aicd: aicd factor
 * flow_exp: flow exponent
 */
float function(std::vector<double>& params, double mu_exp, double a_aicd, double flow_exp) {
    auto rho_mix = params[1] * params[2] + params[4] * params[5] + params[7] * params[8];
    auto mu_mix = params[1] * params[3] + params[4] * params[6] + params[7] * params[9];
    auto f = pow(rho_mix,2) * pow(1/mu_mix, mu_exp);
    return f;
}


/* 
 * void time_fitting(int count)
 * Time how long the 'function' function takes to run on average over count iterations
 * count: number of runs to average over
 */
void time_fitting(int count) {
    double result, total_time;
    
    double mu_exp = 0.952;
    double a_aicd = 0.008;
    double flow_exp = 2.742;
    std::vector<double> params = {4.63,0.25,0.755,0.95,0.0,0.1118,0.0183,0.75,1.021,0.439};

    total_time = 0;

    // Initial run of the function to allow for setup delays
    result = function(params, mu_exp, a_aicd, flow_exp);

    // Start the actual timing of the fit function
    for (int i =0; i < count; i++){
        auto t1 = std::chrono::high_resolution_clock::now();
        result = function(params, mu_exp, a_aicd, flow_exp);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_double = t2 - t1;
        total_time = total_time + ms_double.count();
    }
    
    // Print the results of the timing
    std::cout << "Tendeka fitting, run " << count << " times took on average " <<
            total_time / count << " milliseconds" << std::endl;
}


/* 
 * void time_ml(std::string model_path, int count)
 * Time how long the ml implementation takes to run on average over count iterations
 * count: number of runs to average over
 */
void time_ml(std::string model_path, int count) {
    double total_time;
    std::vector<torch::jit::IValue> inputs;
    //std::vector<double> params = {4.63,0.25,0.755,0.95,0.0,0.1118,0.0183,0.75,1.021,0.439};
    inputs.push_back(torch::tensor({4.63,0.25,0.755,0.95,0.0,0.1118,0.0183,0.75,1.021,0.439}));

    total_time = 0;


    //Import the model
    torch::jit::script::Module mod;
    //mod = torch::jit::load("../model.pt");
    mod = torch::jit::load(model_path);

    // Initial run of the function to allow for setup delays
    auto result = mod.forward(inputs);

    // Start the actual timing of the fit function
    for (int i =0; i < count; i++){
        auto t1 = std::chrono::high_resolution_clock::now();
        result = mod.forward(inputs);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_double = t2 - t1;
        total_time = total_time + ms_double.count();
    }
    
    // Print the results of the timing
    std::cout << "ML fitting, run " << count << " times took on average " <<
            total_time / count << " milliseconds" << std::endl;
}
