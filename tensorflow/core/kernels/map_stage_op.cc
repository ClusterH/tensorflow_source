#include <map>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

namespace {

// General Template Definition
template <bool Ordered, typename Key, typename Data>
struct MapTraits {};

// Partially specialise for ordered
template <typename Key, typename Data>
struct MapTraits<true, Key, Data>
{
  typedef Key key_type;
  typedef Data data_type;
  typedef std::map<Key, Data> map_type;
};

// Partially specialise for unordered
template <typename Key, typename Data>
struct MapTraits<false, Key, Data>
{
  typedef Key key_type;
  typedef Data data_type;
  typedef std::unordered_map<Key, Data> map_type;
};

// Wrapper around map/unordered_map
template <bool Ordered>
class StagingMap : public ResourceBase
{
public:
  // Public typedefs
  typedef MapTraits<Ordered, int64, std::vector<Tensor>> map_traits;
  typedef typename map_traits::map_type map_type;
  typedef typename map_traits::key_type key_type;
  typedef typename map_traits::data_type Tuple;

private:
  // Private variables
  int capacity_;
  mutex mu_;
  condition_variable not_empty_;
  condition_variable full_;
  map_type map_ GUARDED_BY(mu_);

public:
  // public methods
  explicit StagingMap(int capacity) : capacity_(capacity) {}

  bool has_bounded_capacity()
    { return capacity_ > 0; }

  bool full()
    { return map_.size() >= capacity_; }

  void put(key_type* key, Tuple* tuple)
  {
    mutex_lock l(mu_);

    // If map capacity is bounded wait until map is not full
    if(has_bounded_capacity())
      { full_.wait(l, [this]() { return !this->full(); }); }

    // Insert key and tuples into the map
    map_.insert({*key, std::move(*tuple)});

    notify_removers(l);
  }

  Status get(key_type* key, Tuple* tuple)
  {
    mutex_lock l(mu_);

    typename map_type::const_iterator it;

    // Wait until the element with the requested key is present
    not_empty_.wait(l, [&, this]() {
      it = map_.find(*key);
      return it != map_.end();
    });

    // Copy tensors into the tuple
    for(const auto & tensor : it->second)
      { tuple->push_back(tensor); }

    return Status::OK();
  }

  Status pop(key_type* key, Tuple* tuple)
  {
    mutex_lock l(mu_);

    typename map_type::iterator it;

    // Wait until the element with the requested key is present
    not_empty_.wait(l, [&, this]() {
      it = map_.find(*key);
      return it != this->map_.end();
    });

    // Move from the entry as its erased anyway
    *tuple = std::move(it->second);

    // Remove
    map_.erase(it);

    notify_inserters_if_bounded(l);

    return Status::OK();
  }

  Status popitem(key_type* key, Tuple* tuple)
  {
    mutex_lock l(mu_);

    // Wait until map is not empty
    not_empty_.wait(l, [this]() { return !this->map_.empty(); });

    // Move from the first element and erase it
    *tuple = std::move(map_.begin()->second);
    *key = map_.begin()->first;
    map_.erase(map_.begin());

    notify_inserters_if_bounded(l);

    return Status::OK();
  }

  Status clear()
  {
    mutex_lock l(mu_);
    map_.clear();
    notify_inserters_if_bounded(l);

    return Status::OK();
  }

  size_t size()
  {
    // Lock the map and return the size
    mutex_lock l(mu_);
    return map_.size();
  }

  string DebugString()
  {
    return "StagingMap";
  }

private:
  // private methods

  // If map is configured for bounded capacity, notify
  // waiting inserters that space is now available
  void notify_inserters_if_bounded(mutex_lock & l)
  {
    if(has_bounded_capacity())
    {
      l.unlock();
      full_.notify_one();
    }
  }

  // Notify any removers waiting to extract values
  // that data is now available
  void notify_removers(mutex_lock & l)
  {
      l.unlock();
      not_empty_.notify_one();
  }
};

template <bool Ordered>
Status GetStagingMap(OpKernelContext* ctx,
                    const NodeDef& ndef,
                    StagingMap<Ordered>** map)
{
  auto rm = ctx->resource_manager();
  ContainerInfo cinfo;

  // Lambda for creating the Staging Area
  auto create_fn = [&ndef](StagingMap<Ordered>** ret) -> Status
  {
    int capacity;
    TF_RETURN_IF_ERROR(GetNodeAttr(ndef, "capacity", &capacity));
    *ret = new StagingMap<Ordered>(capacity);
    return Status::OK();
  };

  TF_RETURN_IF_ERROR(cinfo.Init(rm, ndef, true /* use name() */));
  TF_RETURN_IF_ERROR(rm->LookupOrCreate<StagingMap<Ordered>>(
                        cinfo.container(), cinfo.name(),
                        map, create_fn));
  return Status::OK();
}

template <bool Ordered>
class MapStageOp : public OpKernel
{
 public:
  explicit MapStageOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    StagingMap<Ordered>* map = nullptr;
    OP_REQUIRES_OK(ctx, GetStagingMap(ctx, def(), &map));
    core::ScopedUnref scope(map);
    typename StagingMap<Ordered>::Tuple tuple;

    const Tensor * key_tensor;
    OpInputList values_tensor;

    OP_REQUIRES_OK(ctx, ctx->input("key", &key_tensor));
    OP_REQUIRES_OK(ctx, ctx->input_list("values", &values_tensor));

    // Obtain the key
    auto key = key_tensor->scalar<typename StagingMap<Ordered>::key_type>()();

    // Create the tuple to store
    for (int i = 0; i < values_tensor.size(); ++i) {
      tuple.push_back(values_tensor[i]);
    }

    // Store the tuple in the map
    map->put(&key, &tuple);
  }
};

REGISTER_KERNEL_BUILDER(Name("MapStage").Device(DEVICE_CPU),
                      MapStageOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapStage").Device(DEVICE_CPU),
                      MapStageOp<true>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("MapStage").HostMemory("key")
                      .Device(DEVICE_GPU), MapStageOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapStage").HostMemory("key")
                      .Device(DEVICE_GPU), MapStageOp<true>);
#endif
#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("MapStage").HostMemory("key")
                      .Device(DEVICE_SYCL), MapStageOp<false>);
REGISTER_KERNEL_BUILDER(Name("MapStage").HostMemory("key")
                      .Device(DEVICE_SYCL), MapStageOp<true>);

#endif // TENSORFLOW_USE_SYCL

template <bool Ordered>
class MapUnstageOp : public OpKernel
{
 public:
  explicit MapUnstageOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  // Using this op in such a way that it blocks forever
  // is an error.  As such cancellation is not handled.
  void Compute(OpKernelContext* ctx) override {
    StagingMap<Ordered>* map = nullptr;
    OP_REQUIRES_OK(ctx, GetStagingMap(ctx, def(), &map));
    core::ScopedUnref scope(map);
    typename StagingMap<Ordered>::Tuple tuple;

    const Tensor * key_tensor;
    OpInputList values_tensor;

    OP_REQUIRES_OK(ctx, ctx->input("key", &key_tensor));

    auto key = key_tensor->scalar<typename StagingMap<Ordered>::key_type>()();

    OP_REQUIRES_OK(ctx, map->pop(&key, &tuple));

    OP_REQUIRES(
        ctx, tuple.size() == (size_t)ctx->num_outputs(),
        errors::InvalidArgument("Mismatch stage/unstage: ", tuple.size(),
                                " vs. ", ctx->num_outputs()));
    for (size_t i = 0; i < tuple.size(); ++i) {
      ctx->set_output(i, tuple[i]);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("MapUnstage").Device(DEVICE_CPU),
                            MapUnstageOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapUnstage").Device(DEVICE_CPU),
                            MapUnstageOp<true>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("MapUnstage").HostMemory("key")
                            .Device(DEVICE_GPU), MapUnstageOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapUnstage").HostMemory("key")
                            .Device(DEVICE_GPU), MapUnstageOp<true>);
#endif
#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("MapUnstage").HostMemory("key")
                            .Device(DEVICE_SYCL), MapUnstageOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapUnstage").HostMemory("key")
                            .Device(DEVICE_SYCL), MapUnstageOp<true>);
#endif // TENSORFLOW_USE_SYCL

template <bool Ordered>
class MapPeekOp : public OpKernel
{
 public:
  explicit MapPeekOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  // Using this op in such a way that it blocks forever
  // is an error.  As such cancellation is not handled.
  void Compute(OpKernelContext* ctx) override {
    StagingMap<Ordered>* map = nullptr;
    OP_REQUIRES_OK(ctx, GetStagingMap(ctx, def(), &map));
    core::ScopedUnref scope(map);
    typename StagingMap<Ordered>::Tuple tuple;

    const Tensor * key_tensor;
    OpInputList values_tensor;

    OP_REQUIRES_OK(ctx, ctx->input("key", &key_tensor));

    auto key = key_tensor->scalar<typename StagingMap<Ordered>::key_type>()();

    OP_REQUIRES_OK(ctx, map->get(&key, &tuple));

    OP_REQUIRES(
        ctx, tuple.size() == (size_t)ctx->num_outputs(),
        errors::InvalidArgument("Mismatch stage/unstage: ", tuple.size(),
                                " vs. ", ctx->num_outputs()));
    for (size_t i = 0; i < tuple.size(); ++i) {
      ctx->set_output(i, tuple[i]);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("MapPeek").Device(DEVICE_CPU),
                      MapPeekOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapPeek").Device(DEVICE_CPU),
                      MapPeekOp<true>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("MapPeek").HostMemory("key")
                      .Device(DEVICE_GPU), MapPeekOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapPeek").HostMemory("key")
                      .Device(DEVICE_GPU), MapPeekOp<true>);
#endif
#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("MapPeek").HostMemory("key")
                      .Device(DEVICE_SYCL), MapPeekOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapPeek").HostMemory("key")
                      .Device(DEVICE_SYCL), MapPeekOp<true>);
#endif // TENSORFLOW_USE_SYCL



template <bool Ordered>
class MapUnstageNoKeyOp : public OpKernel
{
 public:
  explicit MapUnstageNoKeyOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  // Using this op in such a way that it blocks forever
  // is an error.  As such cancellation is not handled.
  void Compute(OpKernelContext* ctx) override {
    StagingMap<Ordered>* map = nullptr;
    OP_REQUIRES_OK(ctx, GetStagingMap(ctx, def(), &map));
    core::ScopedUnref scope(map);

    // Pop a random (key, value) off the map
    typename StagingMap<Ordered>::key_type key;
    typename StagingMap<Ordered>::Tuple tuple;
    OP_REQUIRES_OK(ctx, map->popitem(&key, &tuple));

    // Allocate a key tensor and assign the key as the first output
    Tensor * key_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}),
                                                     &key_tensor));
    key_tensor->scalar<typename StagingMap<Ordered>::key_type>()() = key;
    ctx->set_output(0, *key_tensor);

    // Set the rest of the outputs to the tuple Tensors
    OP_REQUIRES(ctx,
      tuple.size() == (size_t)ctx->num_outputs()-1,
      errors::InvalidArgument("Mismatch stage/unstage: ", tuple.size(),
                              " vs. ", ctx->num_outputs()-1));
    for (size_t i = 0; i < tuple.size(); ++i)
    {
      ctx->set_output(i+1, tuple[i]);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("MapUnstageNoKey").Device(DEVICE_CPU),
                      MapUnstageNoKeyOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapUnstageNoKey").Device(DEVICE_CPU),
                      MapUnstageNoKeyOp<true>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("MapUnstageNoKey").HostMemory("key")
                      .Device(DEVICE_GPU), MapUnstageNoKeyOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapUnstageNoKey").HostMemory("key")
                      .Device(DEVICE_GPU), MapUnstageNoKeyOp<true>);

#endif
#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("MapUnstageNoKey").HostMemory("key")
                      .Device(DEVICE_SYCL), MapUnstageNoKeyOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapUnstageNoKey").HostMemory("key")
                      .Device(DEVICE_SYCL), MapUnstageNoKeyOp<true>);
#endif // TENSORFLOW_USE_SYCL


template <bool Ordered>
class MapSizeOp : public OpKernel
{
 public:
  explicit MapSizeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override
  {
    StagingMap<Ordered>* map = nullptr;
    OP_REQUIRES_OK(ctx, GetStagingMap(ctx, def(), &map));
    core::ScopedUnref scope(map);

    // Allocate size output tensor
    Tensor * size = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}),
                                                     &size));

    // Set it to the actual size
    size->scalar<int32>().setConstant(map->size());
  }
};

REGISTER_KERNEL_BUILDER(Name("MapSize").Device(DEVICE_CPU),
                        MapSizeOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapSize").Device(DEVICE_CPU),
                        MapSizeOp<true>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("MapSize").Device(DEVICE_GPU)
                        .HostMemory("size"), MapSizeOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapSize").Device(DEVICE_GPU)
                        .HostMemory("size"), MapSizeOp<true>);
#endif
#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("MapSize").Device(DEVICE_SYCL)
                        .HostMemory("size"), MapSizeOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapSize").Device(DEVICE_SYCL)
                        .HostMemory("size"), MapSizeOp<true>);
#endif // TENSORFLOW_USE_SYCL

template <bool Ordered>
class MapClearOp : public OpKernel
{
 public:
  explicit MapClearOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override
  {
    StagingMap<Ordered>* map = nullptr;
    OP_REQUIRES_OK(ctx, GetStagingMap(ctx, def(), &map));
    core::ScopedUnref scope(map);

    OP_REQUIRES_OK(ctx, map->clear());
  }
};

REGISTER_KERNEL_BUILDER(Name("MapClear").Device(DEVICE_CPU),
                        MapClearOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapClear").Device(DEVICE_CPU),
                        MapClearOp<true>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("MapClear").Device(DEVICE_GPU),
                        MapClearOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapClear").Device(DEVICE_GPU),
                        MapClearOp<true>);
#endif
#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("MapClear").Device(DEVICE_SYCL),
                        MapClearOp<false>);
REGISTER_KERNEL_BUILDER(Name("OrderedMapClear").Device(DEVICE_SYCL),
                        MapClearOp<true>);
#endif // TENSORFLOW_USE_SYCL

}  // namespace

}  // namespace tensorflow
