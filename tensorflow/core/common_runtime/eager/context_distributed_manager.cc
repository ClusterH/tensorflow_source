/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/common_runtime/eager/context_distributed_manager.h"

#include <algorithm>

#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/protobuf/device_filters.pb.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/util/device_name_utils.h"

#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/distributed_runtime/eager/cluster_function_library_runtime.h"
#include "tensorflow/core/distributed_runtime/eager/eager_client.h"
#include "tensorflow/core/distributed_runtime/eager/remote_mgr.h"
#include "tensorflow/core/distributed_runtime/master_env.h"
#include "tensorflow/core/distributed_runtime/remote_device.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/distributed_runtime/session_mgr.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#endif  // !IS_MOBILE_PLATFORM

namespace tensorflow {
#if !defined(IS_MOBILE_PLATFORM)
namespace {

// A dummy collective is run using this group key during the setup of a cluster.
// Must ensure that this group key is never re-used anywhere else later in the
// application to prevent collisions.
constexpr int kDummyHostCollectiveGroupKey = INT_MAX;

int GetClusterSize(const ServerDef& server_def) {
  int cluster_size = 0;
  for (const auto& job : server_def.cluster().job()) {
    cluster_size += job.tasks_size();
  }
  return cluster_size;
}

Status IsLocalDevice(const std::string& device_name,
                     const ServerDef& server_def, bool* is_local) {
  DeviceNameUtils::ParsedName parsed_name;
  if (!DeviceNameUtils::ParseFullOrLocalName(device_name, &parsed_name)) {
    return errors::InvalidArgument(
        "Invalid device name ", device_name,
        " while parsing local devices in TFE_EnableCollectiveOps.");
  }

  *is_local = (parsed_name.job == server_def.job_name() ||
               parsed_name.job == "localhost") &&
              parsed_name.task == server_def.task_index();
  return Status::OK();
}

Status ExecuteCollectiveForIncarnationIds(EagerContext* context,
                                          const ServerDef& server_def,
                                          tensorflow::Tensor input_tensor,
                                          AbstractTensorPtr* output_tensor,
                                          int group_size) {
  auto op = std::make_unique<EagerOperation>(context);
  auto executor = absl::make_unique<EagerExecutor>(false);
  context->SetExecutorForThread(executor.get());

  TensorHandle* tensor_handle = TensorHandle::CreateLocalHandle(input_tensor);
  TF_RETURN_IF_ERROR(
      op->Reset("CollectiveGather", context->HostCPUName().c_str()));
  TF_RETURN_IF_ERROR(op->AddInput(tensor_handle));
  TF_RETURN_IF_ERROR(op->SetAttrType("T", tensor_handle->DataType()));
  TF_RETURN_IF_ERROR(op->SetAttrInt("group_size", group_size));
  TF_RETURN_IF_ERROR(op->SetAttrInt("group_key", kDummyHostCollectiveGroupKey));
  TF_RETURN_IF_ERROR(op->SetAttrInt("instance_key", 1));
  tensorflow::TensorShape shape;
  TF_RETURN_IF_ERROR(tensor_handle->Shape(&shape));
  int num_dims;
  TF_RETURN_IF_ERROR(tensor_handle->NumDims(&num_dims));

  std::unique_ptr<int64_t[]> shape_list;
  shape_list.reset(new int64_t[num_dims]);
  auto dim_sizes = shape.dim_sizes();
  for (int i = 0; i < num_dims; ++i) {
    shape_list[i] = static_cast<int64_t>(dim_sizes[i]);
  }
  TF_RETURN_IF_ERROR(op->SetAttrFloat("timeout_seconds", 300));
  TF_RETURN_IF_ERROR(op->SetAttrShape("shape", shape_list.get(), num_dims));

  int num_retvals = 1;
  std::vector<AbstractTensorHandle*> retvals(1);
  TF_RETURN_IF_ERROR(
      op->Execute(absl::Span<AbstractTensorHandle*>(retvals), &num_retvals));
  Status status;
  output_tensor->reset(
      reinterpret_cast<ImmediateExecutionTensorHandle*>(retvals[0])
          ->Resolve(&status));

  tensor_handle->Unref();
  retvals[0]->Unref();
  return status;
}

tensorflow::Tensor GenerateIncarnationIdTensor(
    int32 worker_id, std::vector<DeviceAttributes> local_devices) {
  tensorflow::Tensor tensor(
      tensorflow::DT_INT64,
      tensorflow::TensorShape({static_cast<int64_t>(local_devices.size()), 2}));
  std::vector<int64_t> incarnation_vec;
  for (const auto& local_device : local_devices) {
    incarnation_vec.push_back(worker_id);
    incarnation_vec.push_back(local_device.incarnation());
  }
  memcpy(tensor.flat<int64_t>().data(), incarnation_vec.data(),
         incarnation_vec.size() * sizeof(int64_t));
  return tensor;
}

// Runs dummy collectives on every device to force the CompleteGroup RPCs to
// gather the DeviceAttributes of all the devices in the cluster.
// Returns the first error obtained from a collective op, if any.
Status FetchAllDeviceAttributes(
    EagerContext* context, const ServerDef& server_def,
    std::vector<DeviceAttributes> local_devices,
    std::vector<DeviceAttributes>* cluster_devices) {
  // Execute an AllGather collective on the cluster to exchange device
  // incarnations and learn about all the remote devices in the cluster.
  // An AllGather collective op is started on the host device of each worker,
  // and the incarnation IDs of each of the devices is shares with all the
  // workers.
  // This method assumes that all workers will have the same distribution of
  // devices and the order of the incarnation IDs in each of the input tensors
  // is fixed i.e CPU devices followed by GPU/TPU devices in ascending order of
  // device_id.
  const int cluster_size = GetClusterSize(server_def);
  std::sort(local_devices.begin(), local_devices.end(),
            [](const DeviceAttributes& lhs, const DeviceAttributes& rhs) {
              return lhs.name() < rhs.name();
            });
  const tensorflow::Tensor& tensor =
      GenerateIncarnationIdTensor(server_def.task_index(), local_devices);
  AbstractTensorPtr output_tensor;
  TF_RETURN_IF_ERROR(ExecuteCollectiveForIncarnationIds(
      context, server_def, tensor, &output_tensor, cluster_size));

  DCHECK_EQ(output_tensor->NumDims(), 2);
  int64_t* data = reinterpret_cast<int64_t*>(output_tensor->Data());
  int device_type_tracker = 0;
  for (int i = 0; i < output_tensor->NumElements(); i += 2) {
    // Since the order of the devices is the same on all workers, we use the
    // local devices list to infer the device_type and number.
    DeviceNameUtils::ParsedName parsed_name;
    const std::string& local_device_name =
        local_devices[device_type_tracker].name();
    if (!DeviceNameUtils::ParseFullName(local_device_name, &parsed_name)) {
      return errors::InvalidArgument("Failed to parse device name: ",
                                     local_device_name);
    }
    parsed_name.task = data[i];
    auto name = DeviceNameUtils::ParsedNameToString(parsed_name);

    DeviceAttributes dev_attr;
    dev_attr.set_name(name);
    dev_attr.set_incarnation(data[i + 1]);
    dev_attr.set_device_type(parsed_name.type);
    cluster_devices->push_back(dev_attr);
    device_type_tracker = (device_type_tracker + 1) % local_devices.size();
  }

  return Status::OK();
}

// Fetch the device attributes of remote devices in the cluster.
Status UpdateDeviceManager(EagerContext* context, const ServerDef& server_def,
                           std::vector<DeviceAttributes> local_devices) {
  std::vector<DeviceAttributes> device_attrs;
  TF_RETURN_IF_ERROR(FetchAllDeviceAttributes(context, server_def,
                                              local_devices, &device_attrs));

  std::vector<std::unique_ptr<Device>> remote_devices;
  for (const auto& device_attr : device_attrs) {
    bool is_local;
    auto status = IsLocalDevice(device_attr.name(), server_def, &is_local);
    if (!status.ok()) return status;
    if (!is_local) {
      remote_devices.emplace_back(
          NewRemoteDevice(context->TFEnv(), device_attr));
    }
  }

  return context->AddDevices(std::move(remote_devices));
}

bool AreLocalDevicesCompatible(const EagerContext* context,
                               const ServerDef& server_def) {
  if (server_def.job_name() != context->HostCPU()->parsed_name().job) {
    return false;
  }
  return server_def.default_session_config().SerializeAsString() ==
         context->session_options().config.SerializeAsString();
}

Status AddRemoteDevicesToMgr(const std::vector<string>& added_remote_workers,
                             WorkerCacheInterface* worker_cache,
                             DynamicDeviceMgr* remote_device_mgr) {
  std::vector<std::unique_ptr<Device>> remote_devices;
  mutex remote_devices_mu;
  int num_added_workers = added_remote_workers.size();
  BlockingCounter counter(num_added_workers);
  std::vector<Status> statuses(num_added_workers);
  for (int i = 0; i < num_added_workers; i++) {
    NewRemoteDevices(
        Env::Default(), worker_cache, added_remote_workers[i],
        [i, &statuses, &counter, &remote_devices, &remote_devices_mu](
            const Status& s, std::vector<Device*>* devices) {
          statuses[i] = s;
          if (s.ok()) {
            mutex_lock l(remote_devices_mu);
            for (Device* d : *devices) {
              remote_devices.emplace_back(d);
            }
          }
          counter.DecrementCount();
        });
  }
  counter.Wait();
  for (int i = 0; i < num_added_workers; i++) {
    TF_RETURN_IF_ERROR(statuses[i]);
  }

  TF_RETURN_IF_ERROR(remote_device_mgr->AddDevices(std::move(remote_devices)));
  return Status::OK();
}

Status GetAllRemoteDevices(const std::vector<string>& remote_workers,
                           WorkerCacheInterface* worker_cache,
                           std::unique_ptr<DynamicDeviceMgr>* device_mgr) {
  auto remote_device_mgr = std::make_unique<DynamicDeviceMgr>();
  TF_RETURN_IF_ERROR(AddRemoteDevicesToMgr(remote_workers, worker_cache,
                                           remote_device_mgr.get()));
  *device_mgr = std::move(remote_device_mgr);
  return Status::OK();
}

Status RemoveRemoteDevicesFromMgr(
    const std::vector<string>& removed_remote_workers,
    DynamicDeviceMgr* remote_device_mgr) {
  const std::vector<Device*> remote_devices =
      (remote_device_mgr->ListDevices());
  std::vector<Device*> devices_to_remove;
  for (Device* d : remote_devices) {
    for (const string& remote_worker : removed_remote_workers) {
      if (DeviceNameUtils::IsSameAddressSpace(remote_worker, d->name())) {
        devices_to_remove.emplace_back(d);
        break;
      }
    }
  }
  TF_RETURN_IF_ERROR(remote_device_mgr->RemoveDevices(devices_to_remove));
  return Status::OK();
}

Status ListRemoteWorkers(ServerInterface* server, const string& local_worker,
                         std::vector<string>* remote_workers) {
  server->master_env()->worker_cache->ListWorkers(remote_workers);
  remote_workers->erase(
      std::remove(remote_workers->begin(), remote_workers->end(), local_worker),
      remote_workers->end());
  return Status::OK();
}

void DifferentiateWorkerLists(const std::vector<string>* current_list,
                              const std::vector<string>* new_list,
                              std::vector<string>* added,
                              std::vector<string>* removed,
                              std::vector<string>* existing) {
  // Get STL set_difference and set_intersection with one list traversal.
  // Similar to the set_difference library function, the input lists
  // (`current_list` and `new_list`) must be sorted before calling the function.
  added->resize(new_list->size());
  removed->resize(current_list->size());
  existing->resize(current_list->size());
  std::vector<string>::const_iterator curr_it = current_list->begin();
  std::vector<string>::const_iterator new_it = new_list->begin();
  std::vector<string>::iterator added_it = added->begin();
  std::vector<string>::iterator removed_it = removed->begin();
  std::vector<string>::iterator existing_it = existing->begin();
  while (curr_it != current_list->end() && new_it != new_list->end()) {
    if (*curr_it < *new_it) {
      *removed_it++ = *curr_it++;
    } else if (*curr_it > *new_it) {
      *added_it++ = *new_it++;
    } else {
      *existing_it++ = *curr_it++;
      new_it++;
    }
  }
  removed_it = std::copy(curr_it, current_list->end(), removed_it);
  added_it = std::copy(new_it, new_list->end(), added_it);
  added->resize(added_it - added->begin());
  removed->resize(removed_it - removed->begin());
  existing->resize(existing_it - existing->begin());
}

Status GetReplacedFromExistingWorkers(
    const std::vector<string>* existing_workers, uint64 context_id,
    uint64 context_view_id, const ServerDef& server_def,
    eager::EagerClientCache* client_cache,
    std::vector<string>* replaced_workers) {
  BlockingCounter counter(existing_workers->size());
  std::vector<Status> statuses(existing_workers->size());
  eager::KeepAliveRequest request;
  request.set_context_id(context_id);
  std::vector<eager::KeepAliveResponse> responses(existing_workers->size());
  for (int i = 0; i < existing_workers->size(); i++) {
    core::RefCountPtr<eager::EagerClient> eager_client;
    statuses[i] =
        client_cache->GetClient(existing_workers->at(i), &eager_client);
    if (!statuses[i].ok()) {
      counter.DecrementCount();
      continue;
    }
    eager_client->KeepAliveAsync(&request, &responses[i],
                                 [i, &statuses, &counter](const Status& s) {
                                   statuses[i] = s;
                                   counter.DecrementCount();
                                 });
  }
  counter.Wait();
  for (int i = 0; i < existing_workers->size(); i++) {
    // If the RPC fails (indicating that the requested ID doesn't exist on
    // remote), or the returned view ID is not equal to the local one
    // (indicating that the remote worker has a stale view of cluster), treat
    // the worker as replaced.
    if (!statuses[i].ok() ||
        responses[i].context_view_id() != context_view_id) {
      replaced_workers->emplace_back(existing_workers->at(i));
    }
  }
  return Status::OK();
}

Status CreateRemoteContexts(EagerContext* context,
                            const std::vector<string>& remote_workers,
                            uint64 context_id, uint64 context_view_id,
                            int keep_alive_secs, const ServerDef& server_def,
                            eager::EagerClientCache* remote_eager_workers,
                            bool async,
                            const eager::CreateContextRequest& base_request) {
  int num_remote_workers = remote_workers.size();
  BlockingCounter counter(num_remote_workers);
  std::vector<Status> statuses(num_remote_workers);
  for (int i = 0; i < num_remote_workers; i++) {
    const string& remote_worker = remote_workers[i];
    DeviceNameUtils::ParsedName parsed_name;
    if (!DeviceNameUtils::ParseFullName(remote_worker, &parsed_name)) {
      statuses[i] = errors::InvalidArgument("Unable to parse ", remote_worker,
                                            " as a device name");
      counter.DecrementCount();
      continue;
    }

    core::RefCountPtr<eager::EagerClient> eager_client;
    statuses[i] = remote_eager_workers->GetClient(remote_worker, &eager_client);
    if (eager_client == nullptr) {
      statuses[i] = errors::Internal(
          "Cannot find a client for the given target:", remote_worker);
    }
    if (!statuses[i].ok()) {
      counter.DecrementCount();
      continue;
    }

    eager::CreateContextRequest request;
    eager::CreateContextResponse* response = new eager::CreateContextResponse();
    request.set_context_id(context_id);
    request.set_context_view_id(context_view_id);
    *request.mutable_server_def() = server_def;
    request.mutable_server_def()->set_job_name(parsed_name.job);
    request.mutable_server_def()->set_task_index(parsed_name.task);
    request.mutable_server_def()->mutable_default_session_config()->MergeFrom(
        server_def.default_session_config());

    std::vector<bool> filtered_device_mask;
    context->FilterDevicesForRemoteWorkers(
        remote_worker, base_request.cluster_device_attributes(),
        &filtered_device_mask);
    DCHECK_EQ(filtered_device_mask.size(),
              base_request.cluster_device_attributes_size());
    for (int i = 0; i < filtered_device_mask.size(); i++) {
      if (filtered_device_mask[i]) {
        const auto& da = base_request.cluster_device_attributes(i);
        *request.add_cluster_device_attributes() = da;
      }
    }
    request.set_async(async);
    request.set_keep_alive_secs(keep_alive_secs);
    // TODO(b/134094971): deprecate lazy_copy_remote_function_inputs when server
    // doesn't try to get the value of lazy_copy_remote_function_inputs.
    request.set_lazy_copy_remote_function_inputs(true);

    eager_client->CreateContextAsync(
        &request, response,
        [i, &statuses, &counter, response](const Status& s) {
          statuses[i] = s;
          delete response;
          counter.DecrementCount();
        });
  }
  counter.Wait();
  StatusGroup sg;
  for (int i = 0; i < num_remote_workers; i++) {
    if (TF_PREDICT_FALSE(!statuses[i].ok())) {
      sg.Update(statuses[i]);
    }
  }
  return sg.as_summary_status();
}

Status UpdateRemoteContexts(EagerContext* context,
                            const std::vector<string>& remote_workers,
                            const std::vector<string>& added_workers,
                            const std::vector<string>& removed_workers,
                            uint64 context_id, uint64 context_view_id,
                            const ServerDef& server_def,
                            eager::EagerClientCache* remote_eager_workers,
                            const eager::CreateContextRequest& base_request) {
  int num_remote_workers = remote_workers.size();
  BlockingCounter counter(num_remote_workers);
  std::vector<Status> statuses(num_remote_workers);

  int cluster_device_count = base_request.cluster_device_attributes_size();
  std::unordered_set<string> added_or_removed(added_workers.begin(),
                                              added_workers.end());
  std::copy(removed_workers.begin(), removed_workers.end(),
            std::inserter(added_or_removed, added_or_removed.end()));
  // Whether each device is in the updated (added or removed) workers
  std::vector<bool> device_added_or_removed(cluster_device_count);
  for (int i = 0; i < base_request.cluster_device_attributes_size(); i++) {
    const auto& da = base_request.cluster_device_attributes().at(i);
    DeviceNameUtils::ParsedName pn;
    DeviceNameUtils::ParseFullName(da.name(), &pn);
    string task_name;
    DeviceNameUtils::GetTaskName(pn, &task_name);
    if (added_or_removed.find(task_name) != added_or_removed.end()) {
      device_added_or_removed[i] = true;
    }
  }

  for (int i = 0; i < num_remote_workers; i++) {
    const string& remote_worker = remote_workers[i];
    DeviceNameUtils::ParsedName parsed_name;
    if (!DeviceNameUtils::ParseFullName(remote_worker, &parsed_name)) {
      statuses[i] = errors::InvalidArgument("Unable to parse ", remote_worker,
                                            " as a device name");
      counter.DecrementCount();
      continue;
    }

    core::RefCountPtr<eager::EagerClient> eager_client;
    statuses[i] = remote_eager_workers->GetClient(remote_worker, &eager_client);
    if (eager_client == nullptr) {
      statuses[i] = errors::Internal(
          "Cannot find a client for the given target:", remote_worker);
    }
    if (!statuses[i].ok()) {
      counter.DecrementCount();
      continue;
    }

    std::vector<bool> filtered_device_mask;
    context->FilterDevicesForRemoteWorkers(
        remote_worker, base_request.cluster_device_attributes(),
        &filtered_device_mask);
    DCHECK_EQ(filtered_device_mask.size(), cluster_device_count);

    // If any of the devices that match the device filters are in the set of
    // added or removed workers, we must send a complete UpdateContextRequest.
    // Otherwise, only send a simple request to increment context view ID.
    std::vector<bool> added_or_removed_filtered_devices(cluster_device_count);
    std::transform(device_added_or_removed.begin(),
                   device_added_or_removed.end(), filtered_device_mask.begin(),
                   added_or_removed_filtered_devices.begin(),
                   std::logical_and<bool>());
    const bool full_update_request =
        std::accumulate(added_or_removed_filtered_devices.begin(),
                        added_or_removed_filtered_devices.end(), false,
                        std::logical_or<bool>());

    eager::UpdateContextRequest request;
    auto* response = new eager::UpdateContextResponse();
    request.set_context_id(context_id);
    request.set_context_view_id(context_view_id);
    if (full_update_request) {
      *request.mutable_server_def() = server_def;
      request.mutable_server_def()->set_job_name(parsed_name.job);
      request.mutable_server_def()->set_task_index(parsed_name.task);
      request.mutable_server_def()->mutable_default_session_config()->MergeFrom(
          server_def.default_session_config());
      for (int i = 0; i < cluster_device_count; i++) {
        if (filtered_device_mask[i]) {
          const auto& da = base_request.cluster_device_attributes(i);
          *request.add_cluster_device_attributes() = da;
        }
      }
    }

    eager_client->UpdateContextAsync(
        &request, response,
        [i, &statuses, &counter, response](const Status& s) {
          statuses[i] = s;
          delete response;
          counter.DecrementCount();
        });
  }
  counter.Wait();
  for (int i = 0; i < num_remote_workers; i++) {
    TF_RETURN_IF_ERROR(statuses[i]);
  }
  return Status::OK();
}

Status UpdateContextWithServerDef(EagerContext* context,
                                  const ServerDef& server_def,
                                  bool reset_context, int keep_alive_secs) {
  // We don't use the TF_RETURN_IF_ERROR macro directly since that destroys the
  // server object (which currently CHECK-fails) and we miss the error, instead,
  // we log the error, and then return to allow the user to see the error
  // message.
#define LOG_AND_RETURN_IF_ERROR(...)                  \
  do {                                                \
    const tensorflow::Status _status = (__VA_ARGS__); \
    if (TF_PREDICT_FALSE(!_status.ok())) {            \
      LOG(ERROR) << _status.error_message();          \
      return _status;                                 \
    }                                                 \
  } while (0);

  string worker_name =
      strings::StrCat("/job:", server_def.job_name(),
                      "/replica:0/task:", server_def.task_index());

  // List of current remote workers before updating server_def. Unused if
  // resetting the server_def.
  std::vector<string> curr_remote_workers;
  // List of updated remote workers.
  std::vector<string> remote_workers;

  // New server created for new server_def. Unused if updating server_def.
  std::unique_ptr<ServerInterface> new_server;
  ServerInterface* server;
  if (reset_context) {
    DeviceMgr* device_mgr = AreLocalDevicesCompatible(context, server_def)
                                ? context->local_device_mgr()
                                : nullptr;
    LOG_AND_RETURN_IF_ERROR(
        NewServerWithOptions(server_def, {device_mgr}, &new_server));
    server = new_server.get();
    LOG_AND_RETURN_IF_ERROR(
        ListRemoteWorkers(new_server.get(), worker_name, &remote_workers));
  } else {
    LOG_AND_RETURN_IF_ERROR(ListRemoteWorkers(context->GetServer(), worker_name,
                                              &curr_remote_workers));
    // No need to check the cast here, since `ListRemoteWorkers` already checks
    // if the server is a GRPC server or not.
    server = context->GetServer();
    LOG_AND_RETURN_IF_ERROR(server->UpdateServerDef(server_def));
    LOG_AND_RETURN_IF_ERROR(
        ListRemoteWorkers(server, worker_name, &remote_workers));
  }

  uint64 context_id = context->GetContextId();
  uint64 context_view_id = context->GetContextViewId();
  if (reset_context) {
    context_id = EagerContext::NewContextId();
    context_view_id = 0;
    // Make master eager context accessible by local eager service, which might
    // receive send tensor requests from remote workers.
    LOG_AND_RETURN_IF_ERROR(
        server->AddMasterEagerContextToEagerService(context_id, context));
  }

  std::unique_ptr<eager::EagerClientCache> remote_eager_workers;
  LOG_AND_RETURN_IF_ERROR(
      server->master_env()->worker_cache->GetEagerClientCache(
          &remote_eager_workers));

  // For cluster update, use a status group to aggregate statuses from
  //   * adding and removing remote devices
  //   * creating remote contexts on newly added workers
  //   * updating remote contexts on existing workers
  //   * updating the master context
  // Note that we should not return immediately on errors in the middle of these
  // updates to prevent cluster from having inconsistent context views.
  //
  // Unused if `reset_context` is True.
  StatusGroup sg;

  // When updating an existing context, populate the following lists with:
  // * added_workers: set(remote_workers) - set(curr_remote_workers)
  // * removed_workers: set(curr_remote_workers) - set(remote_workers)
  // * existing_workers: set(curr_remote_workers) intersect set(remote_workers)
  // * replaced_workers: workers with the same task names and potentially the
  //     same `hostname:port`s, but replaced by different processes
  std::vector<string> added_workers;
  std::vector<string> removed_workers;
  std::vector<string> existing_workers;
  std::vector<string> replaced_workers;

  // New remote device manager created for new server_def. Unused if updating
  // server_def.
  std::unique_ptr<DynamicDeviceMgr> new_remote_device_mgr;
  DynamicDeviceMgr* remote_device_mgr = nullptr;
  if (reset_context) {
    LOG_AND_RETURN_IF_ERROR(
        GetAllRemoteDevices(remote_workers, server->master_env()->worker_cache,
                            &new_remote_device_mgr));
    remote_device_mgr = new_remote_device_mgr.get();
  } else {
    // NOTE(b/143914772): Potential memory leak if rendezvous has pending
    // tensors for removed / replaced workers.
    context->ClearCachesAndDefaultExecutor();

    remote_device_mgr = context->GetOwnedRemoteDeviceMgr();
    if (remote_device_mgr == nullptr) {
      LOG_AND_RETURN_IF_ERROR(errors::InvalidArgument(
          "Updating context with an invalid set of remote devices."));
    }
    std::sort(curr_remote_workers.begin(), curr_remote_workers.end());
    std::sort(remote_workers.begin(), remote_workers.end());
    DifferentiateWorkerLists(&curr_remote_workers, &remote_workers,
                             &added_workers, &removed_workers,
                             &existing_workers);
    sg.Update(GetReplacedFromExistingWorkers(
        &existing_workers, context_id, context->GetContextViewId(), server_def,
        remote_eager_workers.get(), &replaced_workers));
    if (VLOG_IS_ON(1)) {
      VLOG(1) << "Updating cluster with following changes";
      for (const string& w : added_workers) VLOG(1) << "  Added worker " << w;
      for (const string& w : removed_workers)
        VLOG(1) << "  Removed worker " << w;
      for (const string& w : replaced_workers)
        VLOG(1) << "  Replaced worker " << w;
    }
    if (!replaced_workers.empty()) {
      // Treat replaced workers as removed then added back, so that we recreate
      // remote devices and contexts, and re-register functions on those workers
      removed_workers.insert(removed_workers.end(), replaced_workers.begin(),
                             replaced_workers.end());
      added_workers.insert(added_workers.end(), replaced_workers.begin(),
                           replaced_workers.end());
      for (const string& w : replaced_workers) {
        existing_workers.erase(
            std::remove(existing_workers.begin(), existing_workers.end(), w),
            existing_workers.end());
      }
    }
    sg.Update(RemoveRemoteDevicesFromMgr(removed_workers, remote_device_mgr));
    sg.Update(AddRemoteDevicesToMgr(
        added_workers, server->master_env()->worker_cache, remote_device_mgr));
  }

  std::vector<DeviceAttributes> cluster_device_attributes;
  remote_device_mgr->ListDeviceAttributes(&cluster_device_attributes);

  std::vector<DeviceAttributes> local_device_attributes;
  server->worker_env()->device_mgr->ListDeviceAttributes(
      &local_device_attributes);

  // This request make sure that we can create Rendezvous properly between
  // Local and Remote context.
  eager::CreateContextRequest base_request;
  for (const auto& da : cluster_device_attributes) {
    *base_request.add_cluster_device_attributes() = da;
  }
  for (const auto& da : local_device_attributes) {
    *base_request.add_cluster_device_attributes() = da;
  }

  // Initialize remote eager workers.
  if (reset_context) {
    const Status s = CreateRemoteContexts(
        context, remote_workers, context_id, context_view_id, keep_alive_secs,
        server_def, remote_eager_workers.get(), context->Executor().Async(),
        base_request);
    // NOTE: the remote tasks could fail after `GetAllRemoteDevices` and cause
    // the CreateRemoteContexts to fail. We currently only log instead of
    // directly returning the error, since returning here will cause the server
    // object to be destroyed (which currently CHECK-fails). The client will
    // see additional errors if ops are subsequently sent to the failed workers.
    if (TF_PREDICT_FALSE(!s.ok())) {
      LOG(ERROR) << "Error when creating contexts on remote targets: "
                 << s.error_message()
                 << "\nExecuting remote ops or functions on these remote "
                    "targets will fail.";
    }
  } else {
    if (sg.ok()) {
      // Create remote contexts on the newly added workers only if the master
      // has collected all device information from them (i.e., the
      // GetAllRemoteDevices call returns succussfully). Note that in rare cases
      // GetAllRemoteDevices can still fail even with RPCs configured to wait
      // until the remote workers to become alive. If the master creates remote
      // contexts on the workers whose devices are still not collected, those
      // workers will be treated as existing workers subsequently, so the master
      // will never get devices from them even with retrying UpdateServerDef.
      sg.Update(CreateRemoteContexts(
          context, added_workers, context_id, context_view_id + 1,
          keep_alive_secs, server_def, remote_eager_workers.get(),
          context->Executor().Async(), base_request));
    }
    if (!existing_workers.empty()) {
      if (VLOG_IS_ON(1)) {
        for (const string& w : existing_workers) {
          VLOG(1) << "Updating cluster with existing worker " << w;
        }
      }
      // The master's context_view_id will be incremented by one in the
      // UpdateRemoteMaster call later. We want existing workers to also have
      // the updated context_view_id, so we must set their context_view_id to
      // the master's current context_view_id + 1.
      sg.Update(UpdateRemoteContexts(context, existing_workers, added_workers,
                                     removed_workers, context_id,
                                     context_view_id + 1, server_def,
                                     remote_eager_workers.get(), base_request));
    }
  }

  auto session_name = strings::StrCat("eager_", context_id);
  if (reset_context) {
    RemoteRendezvous* r =
        server->worker_env()->rendezvous_mgr->Find(context_id);
    auto* device_mgr = server->worker_env()->device_mgr;
    std::shared_ptr<WorkerSession> worker_session;
    LOG_AND_RETURN_IF_ERROR(server->worker_env()->session_mgr->CreateSession(
        session_name, server_def, base_request.cluster_device_attributes(),
        true));
    LOG_AND_RETURN_IF_ERROR(
        server->worker_env()->session_mgr->WorkerSessionForSession(
            session_name, &worker_session));

    // Initialize remote tensor communication based on worker session.
    LOG_AND_RETURN_IF_ERROR(r->Initialize(worker_session.get()));

    DistributedFunctionLibraryRuntime* cluster_flr =
        eager::CreateClusterFLR(context_id, context, worker_session.get());
    auto remote_mgr = std::make_unique<eager::RemoteMgr>(
        /*is_master=*/true, context);

    LOG_AND_RETURN_IF_ERROR(context->InitializeRemoteMaster(
        std::move(new_server), server->worker_env(), worker_session,
        std::move(remote_eager_workers), std::move(new_remote_device_mgr),
        remote_workers, context_id, r, device_mgr, keep_alive_secs, cluster_flr,
        std::move(remote_mgr)));

    // NOTE: We start the server after all other initialization, because the
    // GrpcServer cannot be destroyed after it is started.
    LOG_AND_RETURN_IF_ERROR(server->Start());
  } else {
    sg.Update(server->worker_env()->session_mgr->UpdateSession(
        session_name, server_def, base_request.cluster_device_attributes(),
        /*isolate_session_state=*/true));
    sg.Update(context->UpdateRemoteMaster(context_id,
                                          std::move(remote_eager_workers),
                                          added_workers, removed_workers));
    LOG_AND_RETURN_IF_ERROR(sg.as_summary_status());
  }
#undef LOG_AND_RETURN_IF_ERROR

  return Status::OK();
}
}  // namespace

Status EagerContextDistributedManager::SetOrUpdateServerDef(
    const ServerDef& server_def, bool reset_context, int keep_alive_secs) {
  if (server_def.has_cluster_device_filters()) {
    if (reset_context) {
      const auto& cdf = server_def.cluster_device_filters();
      for (const auto& jdf : cdf.jobs()) {
        const string remote_prefix = "/job:" + jdf.name() + "/task:";
        for (const auto& tdf : jdf.tasks()) {
          const int32_t task_index = tdf.first;
          std::vector<string> device_filters(tdf.second.device_filters_size());
          for (int i = 0; i < tdf.second.device_filters_size(); i++) {
            device_filters[i] = tdf.second.device_filters(i);
          }
          const string remote_worker =
              strings::StrCat(remote_prefix, task_index);
          TF_RETURN_IF_ERROR(
              context_->SetRemoteDeviceFilters(remote_worker, device_filters));
        }
      }
    } else {
      LOG(WARNING) << "Device filters can only be specified when initializing "
                      "the cluster. Any changes in device filters are ignored "
                      "when updating the server def.";
    }
  }
  return UpdateContextWithServerDef(context_, server_def, reset_context,
                                    keep_alive_secs);
}

Status EagerContextDistributedManager::EnableCollectiveOps(
    const ServerDef& server_def) {
  // We don't use the TF_RETURN_IF_ERROR macro directly since that destroys the
  // server object (which currently CHECK-fails) and we miss the error, instead,
  // we log the error, and then return to allow the user to see the error
  // message.
#define LOG_AND_RETURN_IF_ERROR(...)                  \
  do {                                                \
    const tensorflow::Status _status = (__VA_ARGS__); \
    if (TF_PREDICT_FALSE(!_status.ok())) {            \
      LOG(ERROR) << _status.error_message();          \
      return _status;                                 \
    }                                                 \
  } while (0);

  ServerInterface* server = context_->GetServer();
  if (server == nullptr) {
    std::unique_ptr<ServerInterface> new_server;
    LOG_AND_RETURN_IF_ERROR(NewServer(server_def, &new_server));
    server = new_server.get();
    if (server == nullptr) {
      LOG_AND_RETURN_IF_ERROR(errors::Internal(
          "Currently, TF eager runtime only supports GrpcServer."));
    }
    auto worker_cache =
        server->worker_env()->session_mgr->LegacySession()->worker_cache();
    const auto& config = server_def.default_session_config();
    const bool enable_coordination =
        !config.experimental().coordination_service().empty();
    const bool fetch_remote_devices =
        config.experimental().fetch_remote_devices_in_multi_client();
    // TODO(haoyuzhang): Consider unify the two solutions for fetching remote
    // devices in multi-client cluster setup
    assert(!(enable_coordination && fetch_remote_devices) &&
           "Use one of the coordination service or all-gather solutions to "
           "fetch remote device attributes.");
    if (enable_coordination) {
      // For coordination leader: start the service instance
      const std::string& leader =
          config.experimental().collective_group_leader();
      DeviceNameUtils::ParsedName parsed;
      DeviceNameUtils::ParseFullName(leader, &parsed);
      if (parsed.job == server_def.job_name() &&
          parsed.task == server_def.task_index()) {
        LOG_AND_RETURN_IF_ERROR(EnableCoordinationService(
            config.experimental().coordination_service(), server->worker_env(),
            server_def, worker_cache));
      }
      LOG_AND_RETURN_IF_ERROR(server->SetCoordinationServiceAgentInstance(
          coordination_service_agent_.get()));
    }
    LOG_AND_RETURN_IF_ERROR(server->Start());
    if (fetch_remote_devices || enable_coordination) {
      auto session_name = strings::StrCat("eager_", context_->GetContextId());
      std::shared_ptr<WorkerSession> worker_session;
      LOG_AND_RETURN_IF_ERROR(server->worker_env()->session_mgr->CreateSession(
          session_name, server_def, true));
      LOG_AND_RETURN_IF_ERROR(
          server->worker_env()->session_mgr->WorkerSessionForSession(
              session_name, &worker_session));
      context_->SetWorkerEnv(server->worker_env(), worker_session);
    }

    if (enable_coordination) {
      // Coordination agent: initialize, connect, wait for all tasks
      std::unique_ptr<CoordinationClientCache> agent_cache;
      LOG_AND_RETURN_IF_ERROR(
          worker_cache->GetCoordinationClientCache(&agent_cache));
      LOG_AND_RETURN_IF_ERROR(coordination_service_agent_->Initialize(
          server->worker_env(), server_def, std::move(agent_cache),
          [this](Status s) {
            context_->GetCollectiveExecutorHandle()->get()->StartAbort(s);
          }));
      LOG_AND_RETURN_IF_ERROR(coordination_service_agent_->Connect());
      LOG_AND_RETURN_IF_ERROR(coordination_service_agent_->WaitForAllTasks());

      // Add remote devices to eager context.
      std::vector<std::unique_ptr<Device>> remote_devices;
      for (const auto& d :
           coordination_service_agent_->GetClusterDeviceAttributes()) {
        bool is_local;
        LOG_AND_RETURN_IF_ERROR(IsLocalDevice(d.name(), server_def, &is_local));
        if (!is_local) {
          remote_devices.emplace_back(NewRemoteDevice(context_->TFEnv(), d));
        }
      }
      LOG_AND_RETURN_IF_ERROR(context_->AddDevices(std::move(remote_devices)));
    }

    LOG_AND_RETURN_IF_ERROR(context_->StoreCollectiveOpsServer(
        std::move(new_server), server->worker_env()->device_mgr,
        server->worker_env()->collective_executor_mgr.get()));
    if (fetch_remote_devices) {
      std::vector<DeviceAttributes> local_devices;
      context_->ListDevices(&local_devices);
      LOG_AND_RETURN_IF_ERROR(
          UpdateDeviceManager(context_, server_def, local_devices));
    }
  } else {
    LOG_AND_RETURN_IF_ERROR(server->UpdateServerDef(server_def));
    LOG_AND_RETURN_IF_ERROR(context_->StoreCollectiveOpsServer(
        /*new_server=*/nullptr, server->worker_env()->device_mgr,
        server->worker_env()->collective_executor_mgr.get()));
  }
#undef LOG_AND_RETURN_IF_ERROR
  return Status::OK();
}

Status EagerContextDistributedManager::EnableCoordinationService(
    const std::string& service_type, const WorkerEnv* worker_env,
    const ServerDef& server_def, WorkerCacheInterface* worker_cache) {
  std::unique_ptr<CoordinationClientCache> client_cache;
  TF_RETURN_IF_ERROR(worker_cache->GetCoordinationClientCache(&client_cache));
  coordination_service_ =
      CoordinationServiceInterface::EnableCoordinationService(
          service_type, worker_env, server_def, std::move(client_cache));
  return Status::OK();
}

Status EagerContextDistributedManager::CheckRemoteAlive(
    const std::string& remote_task_name, bool* is_alive) {
  *is_alive = false;
  WorkerInterface* wi =
      context_->GetServer()->master_env()->worker_cache->GetOrCreateWorker(
          remote_task_name);
  if (wi == nullptr) {
    return errors::InvalidArgument(
        "Unable to find worker interface corresponding to task ",
        remote_task_name);
  }

  GetStatusRequest request;
  GetStatusResponse response;
  Status remote_status;
  Notification done;
  wi->GetStatusAsync(/*opts_=*/nullptr, &request, &response, /*fail_fast=*/true,
                     [&remote_status, &done](const Status& s) {
                       remote_status = s;
                       done.Notify();
                     });
  done.WaitForNotification();

  if (remote_status.ok()) {
    *is_alive = true;
  } else {
    LOG(INFO) << "Remote worker " << remote_task_name
              << " is not alive: " << remote_status.error_message();
  }
  return Status::OK();
}
#endif  // !IS_MOBILE_PLATFORM
}  // namespace tensorflow
