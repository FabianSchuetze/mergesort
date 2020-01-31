#ifndef storage_h
#define storage_h
#include <memory>
#include <vector>
typedef int dtype;
class Storage {
   public:
    explicit Storage();
    explicit Storage(const std::vector<int>&);
    // Storage(const Storage&) = delete;
    Storage& operator=(Storage other) = delete;
    // Storage(const Eigen::MatrixXd&&) // I need to provide that!
    ~Storage();
    const dtype* cpu_pointer_const();
    const dtype* gpu_pointer_const();
    //void update_cpu_data(const dtype);
    //void update_gpu_data(const dtype);
    //void update_gpu_data(const dtype*);
    //void update_gpu_data(const dtype*, const unsigned int, const unsigned int);
    dtype* cpu_pointer();
    dtype* gpu_pointer();
    int size() {return _data.size();}
    std::vector<int>& return_data();
    const std::vector<int>& return_data_const();

   private:
    std::vector<int> _data;
    dtype* _cpu_pointer;
    dtype* _gpu_pointer;
    std::string recent_head;
    void initialize_gpu_memory();
    void sync_to_cpu();
    void sync_to_gpu();
};
#endif
