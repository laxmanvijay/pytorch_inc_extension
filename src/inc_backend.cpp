#include "inc_backend.hpp"
#include <iostream>
#include <fmt/core.h>
#include <incc/master.h>
namespace c10d {


bool IncBackendWork::isCompleted() {
  return true;
}

bool IncBackendWork::isSuccess() const {
  return true;
}

bool IncBackendWork::wait(std::chrono::milliseconds /* unused */) {
  return true;
}

c10::intrusive_ptr<c10::ivalue::Future> IncBackendWork::getFuture() {
  return future_;
}

// If necessary, pass store/rank/size to the ctor and exchange connection
// information here
IncBackend::IncBackend(int rank, int size)
    : Backend(rank, size) {}

// This is a dummy allreduce that sets all output tensors to zero
// Modify the implementation to conduct real communication asynchronously
c10::intrusive_ptr<Work> IncBackend::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {

  std::vector<int> data; 

    for (int i = 0; i < 40; i++) {
        data.push_back(i);
    }

  int res = Scheduler::schedule(data);
  fmt::print("Result: {}\n", res);
  std::cout << "hello from inc backend" << std::endl;
  for (auto& tensor : tensors) {
      tensor.zero_();
  }

  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::TensorType::get()));
  future->markCompleted(c10::IValue(tensors));
  return c10::make_intrusive<IncBackendWork>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Backend> IncBackend::createIncBackend(
    const c10::intrusive_ptr<::c10d::Store>& /* unused */,
    int rank,
    int size,
    const std::chrono::duration<float>& /* unused */) {
  return c10::make_intrusive<IncBackend>(rank, size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createIncBackend", &IncBackend::createIncBackend);
}

}