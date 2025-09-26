#include "bmruntime_interface.h"
#include "bmlib_runtime.h"

void empty(bm_handle_t &bm_handle, bm_device_mem_t &mem) {
  int value = 0;
  int mode = 4;
  auto data_size = bm_mem_get_device_size(mem);
  if (data_size % mode != 0) {
      mode = 1;
  }
  auto ret = bm_memset_device_ext(bm_handle, &value, mode, mem);
  assert(BM_SUCCESS == ret);
}

void empty_in_net(bm_handle_t &bm_handle, const bm_net_info_t *net,
                  int stage_idx = 0) {
  for (int i = 0; i < net->input_num; i++) {
    empty(bm_handle, net->stages[stage_idx].input_mems[i]);
  }
}

void empty_out_net(bm_handle_t &bm_handle, const bm_net_info_t *net,
                   int stage_idx = 0) {
  for (int i = 0; i < net->output_num; i++) {
    empty(bm_handle, net->stages[stage_idx].output_mems[i]);
  }
}

void empty_net(bm_handle_t &bm_handle, const bm_net_info_t *net,
               int stage_idx = 0) {
  empty_in_net(bm_handle, net, stage_idx);
  empty_out_net(bm_handle, net, stage_idx);
}

