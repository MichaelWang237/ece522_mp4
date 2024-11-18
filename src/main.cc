/* File: main.cc
 * -------------
 * This file defines the main() routine for the program and not much else.
 * You should not need to modify this file.
 */


#include <chrono>
#include <string>
#include <math.h>
#include <random>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <ctype.h>
#include <iostream>
#include <unistd.h>
#include <algorithm>
#include "analysis.h"
#include "simulationComponents.h"
#include "simulator.h"
#include "printUtils.h"

#define YYDEBUG 1
#define RUN_SIM 0
using std::chrono::duration;
using std::chrono::high_resolution_clock;

// CPU sim param
extern double CPU_PCIe_bandwidth_GBps;
// GPU sim param
extern double GPU_PCIe_bandwidth_GBps;
extern double GPU_frequency_GHz;
extern double GPU_memory_size_GB;
extern double GPU_malloc_uspB;
// SSD sim param
extern double SSD_PCIe_bandwidth_GBps;
// PCIe sim param
extern int PCIe_batch_size_in_page;
// Other sim param
extern bool use_movement_hints;
extern std::string migration_policy_str;
extern std::string eviction_policy_str;
extern Simulator::MigPolicy migration_policy;
extern Simulator::GPUPageTable::EvcPolicy eviction_policy;
extern int prefetch_degree;
extern int num_candidate;
extern double system_latency_us;
extern bool is_ideal;

// Other param
//   In codegen, is_UVM specifies whether to use cudaMallocManaged
//   In simulation, is_UVM specifies whether setup is ideal (i.e. all tensor in GPU mem)
bool is_UVM = true;
//   In codegen, num_iteration specifies number of iterations to profile
//   In simulation, num_iteration specifies number of iterations to run
int num_iteration = -1; // Specified in config file. No need to change here.
int is_transformer = -1;
int borden = 184;

// 
extern double CPU_memory_line_GB;
extern double SSD_read_latency_us;
extern double SSD_write_latency_us;
extern double SSD_latency_us; // Upper bound
extern double delta_parameter;

// Tensor configurations
extern long long memory_offset_intermediate;
extern long long memory_offset_weights;

//
// extern std::vector<Model_Layer*> forward_layers;
// extern std::vector<Model_OP*> forward_ops;
extern std::vector<CUDAKernel> kernel_list;
extern std::vector<Tensor*> tensor_list;
extern std::vector<InactivePeriod*> inactive_periods_list;
extern std::vector<EvictionGuideEntry> EvictionGuideTable;
extern std::vector<long> GPU_resident_memory_estimation;
extern std::vector<double> kernel_time_table;

// input specifications
std::string tensor_info_file;
std::string kernel_info_file;
std::string kernel_aux_time_file;
// output specifications
std::string stat_output_file;
std::string output_folder_name;
// simulation switches
bool is_simulation = true;
bool output_override = false;
// profiling switches
bool is_compile = true;
bool is_run = true;
int compile_max_thread_num = -1;
bool is_cudnn = false;

// random devices
std::mt19937 rand_device;
double kernel_time_std_dev = 0;
unsigned int ran_seed = 1;
double kernel_speedup = 1;

class RedirStdOut {
    public:
        RedirStdOut(std::string filename) {
            info_file = output_folder_name + "/statistics/" + filename;
            buffer.str("");
            old_cout_buf = std::cout.rdbuf();
            cout_buf = std::cout.rdbuf(buffer.rdbuf());
            printf("Saving %s\n", filename.c_str());
        }
        ~RedirStdOut() {
            std::ofstream fout(info_file.c_str());
            fout << buffer.str();
            fout.close();
            std::cout.rdbuf(old_cout_buf);
        }
    private:
        std::string info_file;
        std::stringstream buffer;
        std::streambuf *old_cout_buf;
        std::streambuf *cout_buf;
};

void CheckVar(double var, std::string variable_name, bool gt=true) {
    if ((gt && var < 0) || (!gt && var > 0)) {
        eprintf("Invalid or missing <%s>, current value: %f, should be %s than 0, aborting\n",
                variable_name.c_str(), var, gt ? "greater" : "less");
        assert(false);
    }
}

void SimulationParamSanityCheck() {
    // parameter validation (existence)
    CheckVar(PCIe_batch_size_in_page, "PCIe_batch_size_in_page");
    CheckVar(CPU_PCIe_bandwidth_GBps, "CPU_PCIe_bandwidth_GBps");
    CheckVar(GPU_PCIe_bandwidth_GBps, "GPU_PCIe_bandwidth_GBps");
    CheckVar(SSD_PCIe_bandwidth_GBps, "SSD_PCIe_bandwidth_GBps");
    CheckVar(GPU_frequency_GHz, "GPU_frequency_GHz");
    CheckVar(GPU_memory_size_GB, "GPU_memory_size_GB");
    CheckVar(GPU_malloc_uspB, "GPU_malloc_uspB");

    if (migration_policy == Simulator::MigPolicy::DEEPUM)
        assert(eviction_policy == Simulator::GPUPageTable::EvcPolicy::DEEPUM);
    if (eviction_policy == Simulator::GPUPageTable::EvcPolicy::DEEPUM)
        assert(migration_policy == Simulator::MigPolicy::DEEPUM);
    if (migration_policy == Simulator::MigPolicy::DEEPUM)
        CheckVar(prefetch_degree, "prefetch_degree");
    else
        CheckVar(prefetch_degree, "prefetch_degree", false);
    if (eviction_policy == Simulator::GPUPageTable::EvcPolicy::GUIDED)
        CheckVar(num_candidate, "num_candidate");
    else
        CheckVar(num_candidate, "num_candidate", false);
    std::cout << "num iter" << num_iteration << std::endl;
    CheckVar(num_iteration, "num_iteration");

    // parameter validation (value)
    if (SSD_PCIe_bandwidth_GBps > GPU_PCIe_bandwidth_GBps) {
        eprintf("Invalid SSD Bandwidth [%f] > GPU Bandwidth [%f]\n",
                SSD_PCIe_bandwidth_GBps, GPU_PCIe_bandwidth_GBps);
        assert(false);
    }
    if (CPU_PCIe_bandwidth_GBps > GPU_PCIe_bandwidth_GBps) {
        eprintf("Invalid CPU Bandwidth [%f] > GPU Bandwidth [%f]\n",
                SSD_PCIe_bandwidth_GBps, GPU_PCIe_bandwidth_GBps);
        assert(false);
    }
    if (SSD_PCIe_bandwidth_GBps > CPU_PCIe_bandwidth_GBps) {
        eprintf("Unsupported SSD Bandwidth [%f] > CPU Bandwidth [%f]\n",
                SSD_PCIe_bandwidth_GBps, CPU_PCIe_bandwidth_GBps);
        assert(false);
    }
    if (GPU_PCIe_bandwidth_GBps > SSD_PCIe_bandwidth_GBps + CPU_PCIe_bandwidth_GBps) {
        eprintf("Unsupported GPU Bandwidth [%f] > SSD Bandwidth [%f] + CPU Bandwidth [%f]\n",
                GPU_PCIe_bandwidth_GBps, SSD_PCIe_bandwidth_GBps, CPU_PCIe_bandwidth_GBps);
        assert(false);
    }
    if (kernel_speedup <= 0) {
        eprintf("Invalid kernel speedup [%f]\n", kernel_speedup);
        assert(false);
    }
}

void SetupOutputFolder() {
    if (output_override)
        wprintf("Overriding output folder <%s>...\n", output_folder_name.c_str());
    assert(system(("mkdir -p " + output_folder_name).c_str()) == 0);
    assert(system(("find " + output_folder_name + "/statistics -name \"*.config\" -type f | xargs rm -f").c_str()) == 0);
    // clean up dirs
    if (output_override && !is_simulation) {
        assert(system(("rm -rf " + output_folder_name + "/include").c_str()) == 0);
        assert(system(("rm -rf " + output_folder_name + "/src").c_str()) == 0);
        assert(system(("rm -rf " + output_folder_name + "/bin").c_str()) == 0);
        assert(system(("rm -rf " + output_folder_name + "/scripts").c_str()) == 0);
        assert(system(("rm -rf " + output_folder_name + "/profiling_src").c_str()) == 0);
        assert(system(("rm -f " + output_folder_name + "/main.cu").c_str()) == 0);
        assert(system(("rm -f " + output_folder_name + "/main").c_str()) == 0);
    }
    // make dirs
    assert(system(("mkdir -p " + output_folder_name + "/statistics").c_str()) == 0);
}

void loadKernelInfo() {
    double GPU_frequency_Hz = GPU_frequency_GHz * pow(10, 9);
    std::string line;

    {
        // load tensor info
        std::ifstream tensor_info_fin(tensor_info_file);
        assert(tensor_info_fin.good());
        iprintf("Loading tensor info from file <%s>\n", tensor_info_file.c_str());

        string tensor_id, tensor_size, tensor_global;
        tensor_list.clear();
        while (std::getline(tensor_info_fin, line)) {
            std::stringstream ss(line);
            tensor_id.clear();
            ss >> tensor_id >> tensor_size >> tensor_global;
            if (!tensor_id.size()) continue;

            // populate tensor list
            tensor_list.push_back(new Tensor(
                std::stoll(tensor_size), tensor_global == "true"
            ));
        }
    } {
        // load kernel info
        std::ifstream kinfo_fin(kernel_info_file);
        assert(kinfo_fin.good());
        iprintf("Loading kernel info from file <%s>\n", kernel_info_file.c_str());

        string kernel_idx, ktype, exe_time, input_tensor_list, output_tensor_list, workspace;
        kernel_list.clear();
        while (std::getline(kinfo_fin, line)) {
            std::stringstream ss(line);
            kernel_idx.clear();
            ss >> kernel_idx >> ktype >> exe_time >> input_tensor_list >> output_tensor_list >> workspace;
            if (!kernel_idx.size()) continue;

            std::vector<Tensor*> inputs;
            std::vector<Tensor*> outputs;
            Tensor* workspace_tensor = nullptr;

            auto list_to_vec_tensor = [](std::vector<Tensor*> &vec_tensor, const std::string &str_list) {
                assert(str_list.front() == '[' && str_list.back() == ']');
                std::string tensor_id;
                std::istringstream ss(str_list.substr(1, str_list.size() - 2));
                while(std::getline(ss, tensor_id, ',')) {
                    int tensor_id_i = std::stoi(tensor_id);
                    assert(tensor_id_i < tensor_list.size());
                    vec_tensor.push_back(tensor_list[tensor_id_i]);
                }
            };
            // input tensor list
            list_to_vec_tensor(inputs, input_tensor_list);
            // output tensor list
            list_to_vec_tensor(outputs, output_tensor_list);
            // workspace tensor
            if (workspace.size()) {
                int workspace_tensor_idx = std::stoi(workspace);
                assert(workspace_tensor_idx < tensor_list.size());
                workspace_tensor = tensor_list[std::stoi(workspace)];
                outputs.push_back(workspace_tensor);
            } else {
                workspace_tensor = nullptr;
            }

            // populate kernel list
            kernel_list.push_back(CUDAKernel(
                std::stoi(kernel_idx), kernel_type_revmap.at(ktype), inputs, outputs, workspace_tensor
            ));
            kernel_list.back().execution_cycles = std::stod(exe_time) * GPU_frequency_Hz / 1000.0;
        }
    } {
        // load kernel aux info
        std::ifstream auxtime_fin(kernel_aux_time_file);
        assert(auxtime_fin.good());
        iprintf("Loading aux kernel info from file <%s>\n", kernel_aux_time_file.c_str());

        string kernel_idx, input_pf_exe_time, pf_exe_time;
        while (std::getline(auxtime_fin, line)) {
            std::stringstream ss(line);
            kernel_idx.clear();
            ss >> kernel_idx >> input_pf_exe_time >> pf_exe_time;
            if (!kernel_idx.size()) continue;

            // populate aux timing info for performance model
            int kernel_idx_i = std::stoi(kernel_idx);
            kernel_list[kernel_idx_i].input_pf_execution_cycles = std::stod(input_pf_exe_time) * GPU_frequency_Hz / 1000.0;
            kernel_list[kernel_idx_i].pf_execution_cycles = std::stod(pf_exe_time) * GPU_frequency_Hz / 1000.0;
        }
    } {
        // validation
        for (unsigned tensor_idx = 0; tensor_idx < kernel_list.size(); tensor_idx++) {
            assert(tensor_list[tensor_idx]->tensor_id == tensor_idx);
        }
        for (unsigned kernel_idx = 0; kernel_idx < kernel_list.size(); kernel_idx++) {
            assert(kernel_list[kernel_idx].kernel_id == kernel_idx);
            assert(kernel_list[kernel_idx].input_pf_execution_cycles >= kernel_list[kernel_idx].execution_cycles);
            assert(kernel_list[kernel_idx].pf_execution_cycles >= kernel_list[kernel_idx].input_pf_execution_cycles);
        }
    }
    iprintf("Total %d tensors found\n", tensor_list.size());
    iprintf("Total %d kernels found\n", kernel_list.size());
    iprintf("Simulation data loading done\n", "");
}

void parse_config_args(std::string config_file_path) {
    std::ifstream config_file(config_file_path);
    if (!config_file.good()) {
        eprintf("Config file <%s> does not exist\n", config_file_path.c_str());
        assert(false);
    }
    // parse config file
    std::string line;
    std::string command;
    std::string value;
    printf("\nConfigs:\n");
    while (std::getline(config_file, line)) {
        std::stringstream ss(line);
        command.clear();
        value.clear();
        ss >> command >> value;
        if (command != "#" && command != "")
            printf("%27s: <%s>\n", command.c_str(), value.c_str());

        // general settings
        if (command == "output_folder")                 { output_folder_name = value; }
        else if (command == "output_override")          { output_override = std::stoi(value) != 0; }
        else if (command == "is_simulation")            { is_simulation = std::stoi(value) != 0; }
        else if (command == "is_profiling")             { is_simulation = std::stoi(value) == 0; }
        // profiling general settings
        else if (command == "is_compile")               { is_compile = std::stoi(value) != 0; }
        else if (command == "compile_max_thread_num")   { compile_max_thread_num = std::stoi(value); }
        else if (command == "is_run")                   { is_run = std::stoi(value) != 0; }
        else if (command == "is_cudnn")                 { is_cudnn = std::stoi(value) != 0; }
        // simulation general settings
        else if (command == "is_ideal")                 { is_ideal = std::stoi(value) != 0; }
        else if (command == "use_movement_hints")       { use_movement_hints = std::stoi(value) != 0; }
        else if (command == "tensor_info_file")         { tensor_info_file = value; }
        else if (command == "kernel_info_file")         { kernel_info_file = value; }
        else if (command == "kernel_aux_time_file")     { kernel_aux_time_file = value; }
        else if (command == "stat_output_file")         { stat_output_file = value; }
        else if (command == "migration_policy")         { migration_policy_str = value; }
        else if (command == "eviction_policy")          { eviction_policy_str = value; }
        else if (command == "num_candidate")            { num_candidate = std::stoul(value); }
        else if (command == "prefetch_degree")          { prefetch_degree = std::stoi(value); }
        else if (command == "delta_parameter")          { delta_parameter = std::stod(value); }
        else if (command == "system_latency_us")        { system_latency_us = std::stod(value); }
        else if (command == "num_iteration")            { num_iteration = std::stoi(value); }
        // simulation CPU statistics
        else if (command == "CPU_PCIe_bandwidth_GBps")  { CPU_PCIe_bandwidth_GBps = std::stod(value); }
        else if (command == "CPU_memory_line_GB")       { CPU_memory_line_GB = std::stod(value); }
        // simulation GPU statistics
        else if (command == "GPU_PCIe_bandwidth_GBps")  { GPU_PCIe_bandwidth_GBps = std::stod(value); }
        else if (command == "GPU_memory_size_GB")       { GPU_memory_size_GB = std::stod(value); }
        else if (command == "GPU_frequency_GHz")        { GPU_frequency_GHz = std::stod(value); }
        else if (command == "GPU_malloc_uspB")          { GPU_malloc_uspB = std::stod(value); }
        // simulation SSD statistics
        else if (command == "SSD_PCIe_bandwidth_GBps")  { SSD_PCIe_bandwidth_GBps = std::stod(value); }
        else if (command == "SSD_read_latency_us")      { SSD_read_latency_us = std::stod(value); }
        else if (command == "SSD_write_latency_us")     { SSD_write_latency_us = std::stod(value); }
        else if (command == "SSD_latency_us")           { SSD_latency_us = std::stod(value); }
        // simulation PCIe statistics
        else if (command == "PCIe_batch_size_page")     { PCIe_batch_size_in_page = std::stoi(value); }
        // simulation Timing sentivity statistics
        else if (command == "kernel_time_std_dev")      { kernel_time_std_dev = std::stod(value); }
        else if (command == "ran_seed")                 { ran_seed = std::stoi(value); }
        else if (command == "kernel_speedup")           { kernel_speedup = std::stod(value); }
        // comments or empty line
        else if (command == "#" || command == "")       {}
        else {
          eprintf("Error: Invalid config entry <%s>, aborting...\n", command.c_str());
          assert(false);
        }
    }
}

int main(int argc, char *argv[]) {
    // config file should be the first argument
    if (argc == 1) {
        eprintf("Please specify a config file\n", "");
        assert(false);
    }
    // exit if config file does not exist
    std::string config_file_path = string(argv[1]);

    parse_config_args(config_file_path);
    // sanity check
    assert((int) Simulator::GPUPageTable::EvcPolicy::DEEPUM != (int) Simulator::MigPolicy::DEEPUM);

    // parameter transformation
    if (output_folder_name.back() == '/') output_folder_name.pop_back();
    stat_output_file = output_folder_name + "/" + stat_output_file;

    if (is_simulation) {
        // eviction policy
        std::transform(eviction_policy_str.begin(), eviction_policy_str.end(), eviction_policy_str.begin(), ::toupper);
        if (eviction_policy_str == "RANDOM") {
            eviction_policy = Simulator::GPUPageTable::EvcPolicy::RANDOM;
        } else if (eviction_policy_str == "LRU" || eviction_policy_str == "TOLERANT") {
            eviction_policy = Simulator::GPUPageTable::EvcPolicy::LRU;
        } else if (eviction_policy_str == "GUIDED") {
            eviction_policy = Simulator::GPUPageTable::EvcPolicy::GUIDED;
        } else if (eviction_policy_str == "DEEPUM") {
            eviction_policy = Simulator::GPUPageTable::EvcPolicy::DEEPUM;
        } else {
            wprintf("Defaulting eviction policy to be LRU\n", "");
            eviction_policy = Simulator::GPUPageTable::EvcPolicy::LRU;
        }
        // migration policy
        std::transform(migration_policy_str.begin(), migration_policy_str.end(), migration_policy_str.begin(), ::toupper);
        if (migration_policy_str == "DEEPUM") {
            migration_policy = Simulator::MigPolicy::DEEPUM;
        } else {
            wprintf("Defaulting migration policy to be OURS\n", "");
            migration_policy = Simulator::MigPolicy::OURS;
        }
    }

    // parameter validation
    if (is_simulation) {
        SimulationParamSanityCheck();
    }
    printf("End configs\n\n");

    // set random seed
    srand(0);

    bool output_folder_exists = system(("test -d " + output_folder_name).c_str()) == 0;
    if (output_folder_exists && !output_override) {
        wprintf("Output folder <%s> exists\n", output_folder_name.c_str());
    }

    // cout redirection
    RedirStdOut* r;

    // ParseCommandLine(argc, argv);

    SetupOutputFolder();

    loadKernelInfo();

    if (is_simulation) {
        // tensor info
        r = new RedirStdOut("tensors.config");
        for (size_t i = 0; i < tensor_list.size(); i++) {
            tensor_list[i]->print();
        }
        delete r;

        // kernel info
        r = new RedirStdOut("kernels.config");
        for (size_t i = 0; i < kernel_list.size(); i++) {
            kernel_list[i].print();
        }
        delete r;

        nprintf("Global Memory amount:       %12lld B (%8.2f GB)\n", memory_offset_weights, memory_offset_weights / pow(1024, 3));
        nprintf("Intermediate Memory amount: %12lld B (%8.2f GB)\n", memory_offset_intermediate, memory_offset_intermediate / pow(1024, 3));
        nprintf("Memory Overcommitment:      %lld B/%lld B, %f GB/%f GB (%f%%)\n",
                memory_offset_intermediate + memory_offset_weights, (long long) (GPU_memory_size_GB * std::pow(1024, 3)),
                (memory_offset_intermediate + memory_offset_weights) / std::pow(1024, 3), GPU_memory_size_GB,
                (memory_offset_intermediate + memory_offset_weights) / (GPU_memory_size_GB * std::pow(1024, 3)) * 100);
        long max_num_pages = 0;
        CUDAKernel *max_mem_usage_kernel = nullptr;
        for (auto it = kernel_list.begin(); it != kernel_list.end(); ++it) {
            CUDAKernel *current_kernel = &(*it);
            vector<Tensor *> required_tensors;
            current_kernel->getRequiredTensors(required_tensors);
            long num_pages = 0;
            for (Tensor *tensor : required_tensors) {
                num_pages += std::ceil((float) tensor->size_in_byte / PAGE_SIZE);
            }
            if (num_pages > max_num_pages) {
                max_num_pages = num_pages;
                max_mem_usage_kernel = current_kernel;
            }
        }
        double max_memory_usage_GB = max_num_pages * PAGE_SIZE / std::pow(1024, 3);
        assert(max_mem_usage_kernel != nullptr);
        nprintf("Memory Usage Maximized at Kernel%d: %lld B (%f GB)\n",
                max_mem_usage_kernel->kernel_id, max_num_pages * PAGE_SIZE,
                max_memory_usage_GB);
        if (max_memory_usage_GB > GPU_memory_size_GB) {
            eprintf("Single kernel memory usage %f GB greater than total GPU memory size %f GB, aborting\n",
                    max_memory_usage_GB, GPU_memory_size_GB);
            assert(false);
        }

        tensor_first_pass_liveness_analysis();
        tensor_second_pass_interval_formation();
        get_inactive_periods_time();

        // life cycle info
        r = new RedirStdOut("interval.config");
        for (int i = 0; i < tensor_list.size(); i++) {
            tensor_list[i]->print_liveness();
            tensor_list[i]->print_inactive_periods();
        }
        delete r;

        if (use_movement_hints)
            scheduling_movement_hints();

        // nprintf("Average interval time: %f ms\n", inactive_periods_list[(inactive_periods_list.size() - 1) / 2]->time_estimated);
        iprintf("Checking output stat files\n", "");
        Simulator::Stat stat(stat_output_file);
        
        #if RUN_SIM 
            if (!stat.outputFileExists()) {
                if (kernel_time_std_dev != 0) {
                    printf("Kernel time variation with std %f\n", kernel_time_std_dev);
                    std::uniform_real_distribution<double> distribution(1 - kernel_time_std_dev, 1 + kernel_time_std_dev);
                    if (ran_seed != 1)
                    {
                        rand_device.seed((unsigned int)(ran_seed));
                    }
                    // rand_device.seed((unsigned int)(100*kernel_time_std_dev));
                    for (int i = 0; i < kernel_list.size(); i++) {
                        double ratio = distribution(rand_device);
                        if (ratio < 0.1) ratio = 0.1; // prevent normal distribution to produce a negative number
                        if (ratio > 1.9) ratio = 1.9; // ensure that the mean is still around 1.0
                        kernel_list[i].execution_cycles *= ratio;
                        kernel_list[i].input_pf_execution_cycles *= ratio;
                        kernel_list[i].pf_execution_cycles *= ratio;
                        assert(kernel_list[i].execution_cycles > 0);
                        assert(kernel_list[i].input_pf_execution_cycles > 0);
                        assert(kernel_list[i].pf_execution_cycles > 0);
                    }
                }
                iprintf("\nPerforming Simulation\n", "");
                Simulator::EventSimulator *sim = new Simulator::EventSimulator(stat_output_file);
                sim->run(num_iteration);
                delete sim; // make sure stats are written back to the files
            }
            iprintf("\nPerforming Analysis\n", "");
            stat.prepareOutputFiles(true);
            stat.analyzeStat();
        #endif
    }

    // for (int i = 0; i < forward_layers.size(); i++)
    // {
    //   delete forward_layers[i];
    // }
    // for (int i = 0; i < tensor_list.size(); i++)
    // {
    //   delete tensor_list[i];
    // }


    // return (ReportError::NumErrors() == 0? 0 : -1);
}
