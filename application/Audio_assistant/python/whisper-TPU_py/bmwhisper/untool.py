import ctypes 
import numpy as np 
import time 
import os

int_point = ctypes.POINTER(ctypes.c_int)
int_      = ctypes.c_int
ulonglong = ctypes.c_ulonglong
cpoint    = ctypes.c_ulong
vpoint    = ctypes.c_void_p
spoint    = ctypes.c_char_p
bool_     = ctypes.c_bool

def make2_c_int_list(my_list):
    return (ctypes.c_int * len(my_list))(*my_list)

def char_point_2_str(char_point):
    return ctypes.string_at(char_point).decode('utf-8')

def make_np2c(np_array):
    return np_array.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))

cpoint = ctypes.c_ulong


data_type = {
    np.float32:0,
    np.float16:1,
    np.int16:4,
    np.int32:6,
    np.dtype(np.float32):0,
    np.dtype(np.float16):1,
    np.dtype(np.int16):4,
    np.dtype(np.int32):6,
}

data_type_map = {
    0: np.float32,
    1: np.float16,
    4: np.int16,
    6: np.int32
}

# mylibrary = ctypes.CDLL(os.path.join(os.path.dirname(__file__), './third_party/untpu/lib/libuntpu.so'))

class Tool:
    def __init__(self, chip_mode="pcie"):
        self.mylibrary = ctypes.CDLL(os.path.join(os.path.dirname(__file__), f'./third_party/untpu/lib/libuntpu_{chip_mode}.so'))
        self.base_init()
        self.tensor_init()
        self.model_init()
        self.runtime_init()
        # self.op_init()

    def base_init(self):
        self.base = {}
        self.bmhandle       = self.mylibrary.get_bmhandle
        self.bmrt           = self.mylibrary.get_bmrt_
        self.free_bmhandle  = self.mylibrary.free_bmhandle
        self.free_bmrt      = self.mylibrary.free_bmrt_
        self.print_data_fp32= self.mylibrary.print_data_fp32
        # set argtypes and restypes
        self.bmhandle.restype       = cpoint
        self.free_bmhandle.argtypes = [cpoint]
        self.bmrt.argtypes          = [cpoint]
        self.bmrt.restype           = cpoint
        self.free_bmrt.argtypes     = [cpoint]
        self.free_bmrt.argtypes     = [cpoint]
        self.print_data_fp32.argtypes=[vpoint, int_, int_, int_, int_]
    
    def tensor_init(self):
        self.create_tensor             = self.mylibrary.create_tensor
        self.destroy_tensor            = self.mylibrary.destroy_tensor
        self.init_tensor               = self.mylibrary.init_tensor
        self.init_tensor_by_shape      = self.mylibrary.init_tensor_by_shape
        self.init_tensor_by_device_mem = self.mylibrary.init_tensor_by_device_mem
        self.copy_data_from_numpy      = self.mylibrary.copy_data_from_numpy
        self.device_to_host            = self.mylibrary.device_to_host
        self.host_to_device            = self.mylibrary.host_to_device
        self.force_host_to_device      = self.mylibrary.force_host_to_device
        self.generate_random_data      = self.mylibrary.generate_random_data
        self.print_data                = self.mylibrary.print_data
        self.print_data_limit          = self.mylibrary.print_data_limit
        self.print_bmtensor            = self.mylibrary.print_bmtensor
        self.print_untensor            = self.mylibrary.print_untensor
        self.shallow_copy_tensor_c     = self.mylibrary.shallow_copy_tensor_c
        self.host_copy_tensor_c        = self.mylibrary.host_copy_tensor_c
        # malloc 
        self.malloc_device             = self.mylibrary.malloc_device
        self.malloc_host               = self.mylibrary.malloc_host
        # print device 
        self.print_device_data         = self.mylibrary.print_device_data_fp32
        
        # type definition for tensor
        self.create_tensor.restype              = cpoint
        self.destroy_tensor.argtypes            = [cpoint,cpoint]
        # void init_tensor(tensor tensor_, int* shape_, int dims, int dtype_)
        self.init_tensor.argtypes               = [cpoint, int_point, int_, int_]
        # void init_tensor_by_shape(tensor tensor_, int* shape_, int dims)
        self.init_tensor_by_shape.argtypes      = [cpoint, int_point, int_]
        # void init_tensor_by_device_mem(tensor tensor_, unsigned long long device_mem_addr_)
        self.init_tensor_by_device_mem.argtypes = [cpoint, ulonglong]
        # void copy_data_from_numpy(tensor tensor_, void* data_, int dtype_)
        self.copy_data_from_numpy.argtypes      = [cpoint, vpoint, int_]
        # void device_to_host(tensor tensor_, bm_handle_t bm_handle)
        self.device_to_host.argtypes            = [cpoint, cpoint]
        # void host_to_device(tensor tensor_, bm_handle_t bm_handle)
        self.host_to_device.argtypes            = [cpoint, cpoint]
        # void force_host_to_device(tensor tensor_, bm_handle_t bm_handle)
        self.force_host_to_device.argtypes      = [cpoint, cpoint]
        # void generate_random_data(tensor tensor_, int dtype_)
        self.generate_random_data.argtypes      = [cpoint, int_]
        # void print_data(tensor tensor_)
        self.print_data.argtypes                = [cpoint]
        # void print_data_limit(tensor tensor_, int start, int len)
        self.print_data_limit.argtypes          = [cpoint, int_, int_]
        # void print_bmtensor(tensor tensor_)
        self.print_bmtensor.argtypes            = [cpoint]
        # void print_untensor(tensor tensor_)
        self.print_untensor.argtypes            = [cpoint]
        # void shallow_copy_tensor_c(tensor dst, tensor src)
        self.shallow_copy_tensor_c.argtypes     = [cpoint, cpoint]
        # void host_copy_tensor_c(tensor dst, tensor src)
        self.host_copy_tensor_c.argtypes        = [cpoint, cpoint]
        # void malloc_device(tensor tensor_, bm_handle_t bm_handle)
        self.malloc_device.argtypes             = [cpoint, cpoint]
        # void malloc_host(tensor tensor_)
        self.malloc_host.argtypes               = [cpoint]
        #  void print_device_data(bm_handle_t bm_handle,int start=0 ,int len=100)
        self.print_device_data.argtypes         = [cpoint, cpoint, int_, int_, bool_, vpoint]
    
    def model_init(self):
        self.create_model                    = self.mylibrary.create_model
        self.destroy_model                   = self.mylibrary.destroy_model
        self.get_model_stage_num             = self.mylibrary.get_model_stage_num
        self.get_model_input_num             = self.mylibrary.get_model_input_num
        self.get_model_output_num            = self.mylibrary.get_model_output_num
        self.get_model_input_dtype           = self.mylibrary.get_model_input_dtype
        self.get_model_output_dtype          = self.mylibrary.get_model_output_dtype
        self.get_model_input_name            = self.mylibrary.get_model_input_name
        self.get_model_output_name           = self.mylibrary.get_model_output_name
        self.get_model_input_dim_by_stage    = self.mylibrary.get_model_input_dim_by_stage
        self.get_model_output_dim_by_stage   = self.mylibrary.get_model_output_dim_by_stage
        self.get_model_input_shape_by_stage  = self.mylibrary.get_model_input_shape_by_stage
        self.get_model_output_shape_by_stage = self.mylibrary.get_model_output_shape_by_stage
        # self.get_coeff_v_start               = self.mylibrary.get_coeff_v_start
        # type definition for tensor
        # un_model* create_model(const char* bmodel, void* p_bmrt)
        self.create_model.argtypes                    = [spoint, cpoint]
        self.create_model.restype                     = cpoint
        # void destroy_model(un_model* model)
        self.destroy_model.argtypes                   = [cpoint]
        # int get_model_stage_num(un_model* model)
        self.get_model_stage_num.argtypes             = [cpoint]
        # int get_model_input_num(un_model* model)
        self.get_model_input_num.argtypes             = [cpoint]
        # int get_model_output_num(un_model* model)
        self.get_model_output_num.argtypes            = [cpoint]
        # int get_model_input_dtype(un_model* model, int stage)
        self.get_model_input_dtype.argtypes           = [cpoint, int_]
        # int get_model_output_dtype(un_model* model, int stage)
        self.get_model_output_dtype.argtypes          = [cpoint, int_]
        # const char* get_model_input_name(un_model* model, int input_id)
        self.get_model_input_name.argtypes            = [cpoint, int_]
        self.get_model_input_name.restype             = cpoint
        # const char* get_model_output_name(un_model* model, int output_id)
        self.get_model_output_name.argtypes           = [cpoint, int_]
        self.get_model_output_name.restype            = cpoint
        # int get_model_input_dim_by_stage(un_model* model, int stage_id, int input_id)
        self.get_model_input_dim_by_stage.argtypes    = [cpoint, int_, int_]
        # int get_model_output_dim_by_stage(un_model* model, int stage_id, int output_id)
        self.get_model_output_dim_by_stage.argtypes   = [cpoint, int_, int_]
        # int get_model_input_shape_by_stage(un_model* model, int stage_id, int input_id)
        self.get_model_input_shape_by_stage.argtypes  = [cpoint, int_, int_]
        self.get_model_input_shape_by_stage.restype   = int_point
        # int get_model_output_shape_by_stage(un_model* model, int stage_id, int output_id)
        self.get_model_output_shape_by_stage.argtypes = [cpoint, int_, int_]
        self.get_model_output_shape_by_stage.restype  = int_point
        # u64 get_coeff_v_start(un_model* model, int net_id, int stage_id)
        # self.get_coeff_v_start.argtypes               = [cpoint, int_, int_]
        # self.get_coeff_v_start.restype                = ulonglong
        def get_model_info(model_handle):
            stage_num = self.get_model_stage_num(model_handle)
            input_num = self.get_model_input_num(model_handle)
            output_num = self.get_model_output_num(model_handle)
            input_names   = []
            output_names  = []
            input_dtypes  = []
            output_dtypes = []
            for i in range(input_num):
                input_names.append(char_point_2_str(self.get_model_input_name(model_handle, i)))
                input_dtypes.append(self.get_model_input_dtype(model_handle, i))
            for i in range(output_num):
                output_names.append(char_point_2_str(self.get_model_output_name(model_handle, i)))
                output_dtypes.append(self.get_model_output_dtype(model_handle, i))  
            all_info = {}
            all_info["stage_num"] = stage_num
            all_info["input_num"] = input_num
            all_info["output_num"] = output_num
            all_info["input_dtypes"] = input_dtypes
            all_info["output_dtypes"] = output_dtypes
            all_info['input_names'] = input_names
            all_info['output_names'] = output_names
            for stage in range(stage_num):
                all_info[stage] = {}
                input_dims = []
                output_dims= []
                input_shapes = []
                output_shapes= []
                for i in range(input_num):
                    input_dims.append(self.get_model_input_dim_by_stage(model_handle, stage, i))
                    tempshape = self.get_model_input_shape_by_stage(model_handle, stage, i)
                    shape = []
                    for j in range(input_dims[i]):
                        shape.append(tempshape[j])
                    input_shapes.append(shape)
                for i in range(output_num):
                    output_dims.append(self.get_model_output_dim_by_stage(model_handle, stage, i))
                    tempshape = self.get_model_output_shape_by_stage(model_handle, stage, i)
                    shape = []
                    for j in range(output_dims[i]):
                        shape.append(tempshape[j])
                    output_shapes.append(shape)
                all_info[stage]['input_dims'] = input_dims
                all_info[stage]['output_dims'] = output_dims
                all_info[stage]['input_shapes'] = input_shapes
                all_info[stage]['output_shapes'] = output_shapes
            return all_info
        self.model_info = get_model_info
    
    def runtime_init(self):
        self.create_un_runtime     = self.mylibrary.create_un_runtime
        self.destroy_un_runtime    = self.mylibrary.free_un_runtime
        self.set_bmodel_info       = self.mylibrary.set_bmodel_info
        self.set_stage             = self.mylibrary.set_stage
        self.init_all_tensors      = self.mylibrary.init_all_tensors
        self.get_input_num         = self.mylibrary.get_input_num
        self.get_output_num        = self.mylibrary.get_output_num
        self.get_input_tensor      = self.mylibrary.get_input_tensor
        self.get_output_tensor     = self.mylibrary.get_output_tensor
        self.set_input_tensor      = self.mylibrary.set_input_tensor
        self.set_output_tensor     = self.mylibrary.set_output_tensor
        # self.set_tensors_shape     = self.mylibrary.set_tensors_shape
        self.malloc_device_address = self.mylibrary.malloc_device_address
        self.generate_input_data   = self.mylibrary.generate_input_data
        self.inference             = self.mylibrary.inference
        self.copy_input_data_to_device = self.mylibrary.copy_input_data_to_device
        self.copy_output_data_to_host  = self.mylibrary.copy_output_data_to_host
        self.print_output_data     = self.mylibrary.print_output_data
        self.print_input_data      = self.mylibrary.print_input_data
        # set argtypes and restype
        # un_run* create_un_runtime(bm_handle_t p_bm_handle)
        self.create_un_runtime.argtypes = [cpoint]
        self.create_un_runtime.restype  = cpoint
        # void free_un_runtime(un_run* p_un_run)
        self.destroy_un_runtime.argtypes = [cpoint]
        # void set_bmodel_info(un_run* un_runtime, un_model* p_un_model)
        self.set_bmodel_info.argtypes    = [cpoint, cpoint]
        # void set_stage(un_run* un_runtime, int stage_id)
        self.set_stage.argtypes         = [cpoint, int_]
        # void init_all_tensors(un_run* un_runtime)
        self.init_all_tensors.argtypes  = [cpoint]
        # int get_input_num(un_run* un_runtime)
        self.get_input_num.argtypes     = [cpoint]
        # int get_output_num(un_run* un_runtime)
        self.get_output_num.argtypes    = [cpoint]
        # tensor get_input_tensor(un_run* un_runtime, int input_id)
        self.get_input_tensor.argtypes  = [cpoint, int_]
        self.get_input_tensor.restype   = cpoint
        # tensor get_output_tensor(un_run* un_runtime, int output_id)
        self.get_output_tensor.argtypes = [cpoint, int_]
        self.get_output_tensor.restype  = cpoint
        # void set_input_tensor(un_run* un_runtime, int input_id, tensor p_tensor)
        self.set_input_tensor.argtypes  = [cpoint, int_, cpoint]
        # void set_output_tensor(un_run* un_runtime, int output_id, tensor p_tensor)
        self.set_output_tensor.argtypes = [cpoint, int_, cpoint]
        # void malloc_device_address(un_run* un_runtime)
        self.malloc_device_address.argtypes = [cpoint]
        # void generate_input_data(un_run* un_runtime)
        self.generate_input_data.argtypes = [cpoint]
        # void inference(un_run* un_runtime)
        self.inference.argtypes = [cpoint]
        # void copy_input_data_to_device(un_run* un_runtime)
        self.copy_input_data_to_device.argtypes = [cpoint]
        # void copy_output_data_to_host(un_run* un_runtime)
        self.copy_output_data_to_host.argtypes = [cpoint]
        # void print_output_data(un_run* un_runtime)
        self.print_output_data.argtypes = [cpoint]
        # void print_input_data(un_run* un_runtime)
        self.print_input_data.argtypes = [cpoint]

        

class Tensor:
    def __init__(self,):
        self._lib = Tool()

    
class Model:
    def __init__(self,):
        self._lib = Tool()
        pass    

class Runtime:
    def __init__(self,):
        self._lib = Tool()
        pass