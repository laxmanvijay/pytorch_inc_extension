#pragma once

#include <torch/extension.h>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>

#include <pybind11/chrono.h>

namespace c10d {

class IncBackend : public Backend {
 public:

  IncBackend(int rank, int size);

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  static c10::intrusive_ptr<Backend> createIncBackend(
      const c10::intrusive_ptr<::c10d::Store>& store,
      int rank,
      int size,
      const std::chrono::duration<float>& timeout);

  static void IncBackendConstructor() __attribute__((constructor)) {
    py::object module = py::module::import("torch.distributed");
    py::object register_backend =
        module.attr("Backend").attr("register_backend");
    register_backend("inc_backend", py::cpp_function(createIncBackend), false, "cpu");
  }
};

class IncBackendWork : public Work {
    friend class IncBackend;
public:
    IncBackendWork(
        OpType opType,
        c10::intrusive_ptr<c10::ivalue::Future> future) // future of the output
        : Work(
              -1, // rank, only used by recvAnySource, irrelevant in this demo
              opType),
          future_(std::move(future)) {}
    bool isCompleted() override;
    bool isSuccess() const override;
    bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override;
    virtual c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

private:
    c10::intrusive_ptr<c10::ivalue::Future> future_;
};

}