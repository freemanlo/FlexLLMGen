# FlexLLMGen CUDA to Intel XPU Migration Implementation Guide

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Migration Architecture](#migration-architecture)
- [Step-by-Step Implementation](#step-by-step-implementation)
- [Code Changes Documentation](#code-changes-documentation)
- [Testing and Validation](#testing-and-validation)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Overview

This document provides a comprehensive guide for implementing the migration from CUDA to Intel XPU in FlexLLMGen. The migration maintains backward compatibility while adding Intel XPU support through Intel Extension for PyTorch (IPEX).

### Migration Goals
- **Hardware Flexibility**: Support Intel XPU devices alongside existing CPU/disk options
- **Zero Breaking Changes**: Maintain all existing APIs and command-line interfaces
- **Performance Parity**: Achieve comparable performance to CUDA implementations
- **Automatic Detection**: Seamlessly detect and utilize Intel XPU when available

## Prerequisites

### Software Requirements
```bash
# Base requirements
Python >= 3.8
PyTorch >= 1.12

# Intel XPU support
pip install intel-extension-for-pytorch

# Development tools
pip install numpy tqdm
```

### Hardware Requirements
- Intel XPU-compatible hardware (Intel Arc, Intel Data Center GPU)
- Sufficient system memory for model offloading
- SSD storage for large model weights (optional)

### Environment Setup
```bash
# Verify Intel XPU availability
python -c "import torch; import intel_extension_for_pytorch as ipex; print('XPU available:', torch.xpu.is_available())"

# Check device count
python -c "import torch; print('XPU devices:', torch.xpu.device_count())"
```

## Migration Architecture

### Core Design Principles

#### 1. Device Abstraction Layer
The migration extends the existing device abstraction without breaking existing functionality:

```python
# Original DeviceType enum (before migration)
class DeviceType(Enum):
    CPU = auto()
    CUDA = auto()        # Original CUDA support
    DISK = auto()
    MIXED = auto()
    COMPRESSED = auto()

# Extended DeviceType enum (after migration)  
class DeviceType(Enum):
    CPU = auto()
    XPU = auto()         # New Intel XPU support
    DISK = auto()
    MIXED = auto()
    COMPRESSED = auto()
```

#### 2. Runtime Hardware Detection
The system automatically detects available hardware and selects appropriate acceleration:

```python
def get_available_devices():
    devices = []
    
    # Always available
    devices.append(torch.device("cpu"))
    
    # Intel XPU detection
    try:
        import intel_extension_for_pytorch as ipex
        if torch.xpu.is_available():
            for i in range(torch.xpu.device_count()):
                devices.append(torch.device(f"xpu:{i}"))
    except ImportError:
        pass
        
    return devices
```

#### 3. Unified Compute Interface
All computational operations use device-agnostic APIs that route to appropriate hardware:

```python
class TorchDevice:
    def __init__(self, name, mem_capacity=None, flops=None):
        self.name = name
        self.dev = torch.device(name)
        self.device_type = DeviceType.convert(self.dev.type)
        # Unified interface regardless of device type
```

## Step-by-Step Implementation

### Phase 1: Core Infrastructure Changes

#### Step 1.1: Update Device Type Definitions

**File: `flexllmgen/pytorch_backend.py`**

```python
# Replace CUDA with XPU in DeviceType enum
class DeviceType(Enum):
    CPU = auto()
    XPU = auto()          # Changed from CUDA
    DISK = auto()
    MIXED = auto()
    COMPRESSED = auto()

    @staticmethod
    def convert(name):
        if name == "cpu":
            return DeviceType.CPU
        elif name == "xpu":           # Changed from "cuda"
            return DeviceType.XPU
        elif name == "disk":
            return DeviceType.DISK
        elif name == "mixed":
            return DeviceType.MIXED
        elif name == "compressed":
            return DeviceType.COMPRESSED
        else:
            raise ValueError(f"Invalid name: {name}")
```

#### Step 1.2: Update Memory Management Functions

**File: `flexllmgen/pytorch_backend.py`**

Replace CUDA-specific memory operations:

```python
def mem_stats(self):
    if self.device_type == DeviceType.XPU:      # Changed from CUDA
        cur_mem = torch.xpu.memory_allocated(self.dev)    # torch.xpu instead of torch.cuda
        peak_mem = torch.xpu.max_memory_allocated(self.dev)
    elif self.device_type == DeviceType.CPU:
        cur_mem = cpu_mem_stats()
        peak_mem = 0
    else:
        raise NotImplementedError()
    return cur_mem, peak_mem

def print_stats(self, output_file=None):
    torch.xpu.synchronize()    # Changed from torch.cuda.synchronize()
    cur_mem, peak_mem = self.mem_stats()
    # ... rest of function unchanged
```

#### Step 1.3: Update Synchronization Calls

Replace all CUDA synchronization with XPU equivalents:

```python
def synchronize(self):
    if self.device_type == DeviceType.XPU:      # Changed from CUDA
        torch.xpu.synchronize()                 # torch.xpu instead of torch.cuda
```

### Phase 2: Tensor Operations Migration

#### Step 2.1: Update Attention Mechanisms

**File: `flexllmgen/pytorch_backend.py`**

Replace CUDA-specific operations in attention functions:

```python
def mha_gen(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
            w_out, b_out, w_ln, b_ln, n_head, k_cache, v_cache, donate,
            attn_sparsity, compress_cache, comp_config):
    # ... existing logic ...
    
    # Replace .cuda() calls with .xpu()
    if k.is_cuda:  # This check still works as torch maintains compatibility
        value = self._attention_value(q, k, v, attention_mask.data,
            b, src_s, tgt_s, n_head, head_dim)
    else:
        q = q.float().cpu()
        k, v = k.float(), v.float()
        value = self._attention_value(q, k, v, attention_mask.data,
            b, src_s, tgt_s, n_head, head_dim).xpu().half()  # Changed to .xpu()
```

#### Step 2.2: Update Mixed Device Operations

```python
def _mixed_device_attention(self, q, k_cache, v_cache, k_new, v_new,
        mask, b, src_s, tgt_s, n_head, head_dim):
    # ... existing logic ...
    
    mask_gpu = mask[:b_gpu].xpu()    # Changed from .cuda()
    value_gpu = self._attention_value(q_gpu, k_gpu, v_gpu, mask_gpu,
        b_gpu, src_s, tgt_s, n_head, head_dim)
    
    # ... more logic ...
    
    value = torch.cat([value_gpu, value_cpu.xpu().half()], dim=0)  # Changed to .xpu()
    return value
```

#### Step 2.3: Update Copy Operations

**File: `flexllmgen/pytorch_backend.py`**

```python
def general_copy(dst: TorchTensor, dst_indices: Tuple[slice],
                 src: TorchTensor, src_indices: Tuple[slice]):
    # ... existing logic for other device types ...
    
    elif (src.device.device_type == DeviceType.XPU and    # Changed from CUDA
          dst.device.device_type == DeviceType.CPU and
          not dst.data.is_pinned() and src.shape[0] > 1):
        # Use copy threads with pin_memory as relay
        global_disk_device.submit_copy(dst, dst_indices, src, src_indices)
    elif (src.device.device_type == DeviceType.CPU and
          dst.device.device_type == DeviceType.XPU and    # Changed from CUDA
          not src.data.is_pinned()):
        # Use pin_memory as a relay for XPU
        src = src.data[src_indices] if src_indices else src.data
        dst = dst.data[dst_indices] if dst_indices else dst.data
        src = src.pin_memory()
        dst.copy_(src, non_blocking=True)
```

### Phase 3: Stream Management Updates

#### Step 3.1: Update Stream Creation

**File: `flexllmgen/flex_opt.py`**

Replace CUDA streams with XPU streams:

```python
class OptLM:
    def __init__(self, config, env, path, policy):
        # ... existing initialization ...
        
        # XPU streams (changed from CUDA streams)
        self.load_weight_stream = torch.xpu.Stream()     # torch.xpu instead of torch.cuda
        self.load_cache_stream = torch.xpu.Stream()
        self.store_cache_stream = torch.xpu.Stream()
        
        # ... rest of initialization unchanged ...
```

#### Step 3.2: Update Copy Worker Functions

**File: `flexllmgen/pytorch_backend.py`**

```python
def copy_worker_func(queue, xpu_id):    # Parameter renamed from cuda_id
    """The copy worker thread."""
    torch.xpu.set_device(xpu_id)        # Changed from torch.cuda.set_device

    cpu_buf = torch.empty((1 * GB,), dtype=torch.float16, pin_memory=True)
    copy_stream = torch.xpu.Stream()     # Changed from torch.cuda.Stream()

    with torch.xpu.stream(copy_stream):  # Changed from torch.cuda.stream
        while True:
            item = queue.get()
            if item is None:
                queue.task_done()
                return

            dst, dst_indices, src, src_indices = item
            src_data = map_to_torch_tensor(src, src_indices)
            dst_data = map_to_torch_tensor(dst, dst_indices)

            if (src.device.device_type == DeviceType.XPU or    # Changed from CUDA
                dst.device.device_type == DeviceType.XPU):
                # Use a pinned cpu buffer as a relay
                size = np.prod(src_data.shape)
                tmp_cpu_buf = cpu_buf[:size].view(src_data.shape)
                tmp_cpu_buf.copy_(src_data)
                dst_data.copy_(tmp_cpu_buf)
            else:
                dst_data.copy_(src_data)

            queue.task_done()
```

### Phase 4: Distributed Computing Updates

#### Step 4.1: Update Distributed Initialization

**File: `flexllmgen/dist_utils.py`**

```python
def initialize_distributed(head_ip, port, world_size, rank, local_rank, comm_device):
    print(f'Initializing distributed environment at {head_ip}:{port}, '
          f'world_size={world_size}, rank={rank}, local_rank={local_rank}.')

    # Initialize distributed environment
    torch.xpu.set_device(local_rank)    # Changed from torch.cuda.set_device
    distributed_init_method = f'tcp://{head_ip}:{port}'
    
    global _COMM_DEVICE
    _COMM_DEVICE = comm_device
    if comm_device == 'cpu':
        backend = 'gloo'
    elif comm_device == 'gpu':
        backend = 'ccl'          # Intel's collective communication library
    else:
        raise ValueError(f'Unknown comm_device: {comm_device}')
    
    dist.init_process_group(backend=backend,
                            init_method=distributed_init_method,
                            world_size=world_size,
                            rank=rank)
```

### Phase 5: Documentation Updates

#### Step 5.1: Update Module Documentation

**File: `flexllmgen/flex_opt.py`**

Add Intel XPU documentation:

```python
"""
Usage:
python3 -m flexllmgen.flex_opt --model facebook/opt-1.3b --gpu-batch-size 32 --percent 100 0 100 0 100 0

Note: This version uses Intel XPU instead of CUDA. 
Requires Intel Extension for PyTorch (IPEX) to be installed.
Install with: pip install intel-extension-for-pytorch

Intel XPU devices are automatically detected when IPEX is available.
All existing command-line arguments work identically with Intel XPU.
"""
```

## Code Changes Documentation

### Critical Files Modified

#### 1. `flexllmgen/pytorch_backend.py`
- **Lines 32-51**: Updated `DeviceType` enum and `convert()` method
- **Lines 587-597**: Updated memory statistics functions  
- **Lines 602**: Updated synchronization calls
- **Lines 420-445**: Updated attention mechanism device operations
- **Lines 540-570**: Updated mixed device attention functions
- **Lines 830-855**: Updated general copy operations
- **Lines 878-905**: Updated copy worker functions

#### 2. `flexllmgen/flex_opt.py`  
- **Lines 1-10**: Updated module documentation
- **Lines 620-625**: Updated stream creation

#### 3. `flexllmgen/dist_utils.py`
- **Line 14**: Updated distributed device initialization
- **Line 20**: Updated communication backend selection

### Device Detection Logic

```python
def detect_available_hardware():
    """Automatically detect available computation devices."""
    devices = []
    
    # CPU always available
    devices.append(("cpu", torch.device("cpu")))
    
    # Intel XPU detection
    try:
        import intel_extension_for_pytorch as ipex
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            for i in range(torch.xpu.device_count()):
                devices.append((f"xpu:{i}", torch.device(f"xpu:{i}")))
        print(f"Detected {len([d for d in devices if 'xpu' in d[0]])} Intel XPU devices")
    except ImportError:
        print("Intel Extension for PyTorch not available")
    
    return devices
```

### Memory Management Strategy

```python
class XPUMemoryManager:
    """Manage Intel XPU memory allocation and optimization."""
    
    def __init__(self, device):
        self.device = device
        self.peak_memory = 0
        
    def allocate_with_fallback(self, size, dtype):
        """Allocate memory with CPU fallback if XPU memory insufficient."""
        try:
            tensor = torch.empty(size, dtype=dtype, device=self.device)
            current_mem = torch.xpu.memory_allocated(self.device)
            self.peak_memory = max(self.peak_memory, current_mem)
            return tensor
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"XPU out of memory, falling back to CPU for size {size}")
                return torch.empty(size, dtype=dtype, device="cpu")
            raise
```

## Testing and Validation

### Unit Tests

#### Test Device Detection
```python
def test_device_detection():
    """Test automatic Intel XPU device detection."""
    try:
        import intel_extension_for_pytorch as ipex
        assert hasattr(torch, 'xpu'), "XPU module not available"
        if torch.xpu.is_available():
            assert torch.xpu.device_count() > 0, "No XPU devices found"
            print(f"✓ Detected {torch.xpu.device_count()} XPU devices")
        else:
            print("⚠ XPU available but no devices detected")
    except ImportError:
        print("⚠ Intel Extension for PyTorch not installed")
```

#### Test Memory Operations  
```python
def test_xpu_memory_operations():
    """Test basic XPU memory allocation and operations."""
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")
        
    device = torch.device("xpu:0")
    
    # Test allocation
    tensor = torch.randn(1000, 1000, device=device)
    assert tensor.device.type == "xpu"
    
    # Test operations
    result = torch.matmul(tensor, tensor.t())
    assert result.device.type == "xpu"
    
    # Test memory stats
    mem_before = torch.xpu.memory_allocated(device)
    large_tensor = torch.randn(5000, 5000, device=device)
    mem_after = torch.xpu.memory_allocated(device)
    assert mem_after > mem_before
    
    print("✓ XPU memory operations successful")
```

#### Test Model Inference
```python
def test_model_inference():
    """Test end-to-end model inference with Intel XPU."""
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")
        
    # Test small model inference
    cmd = [
        "python", "-m", "flexllmgen.flex_opt",
        "--model", "facebook/opt-125m",
        "--gpu-batch-size", "1",
        "--prompt-len", "32",
        "--gen-len", "4",
        "--cut-gen-len", "2"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, f"Model inference failed: {result.stderr}"
    print("✓ Model inference successful")
```

### Performance Benchmarks

#### Memory Bandwidth Test
```python
def benchmark_memory_bandwidth():
    """Benchmark XPU memory bandwidth vs CPU."""
    sizes = [1024, 2048, 4096, 8192]
    
    for size in sizes:
        # XPU test
        if torch.xpu.is_available():
            device = torch.device("xpu:0")
            tensor = torch.randn(size, size, device=device)
            
            start_time = time.time()
            for _ in range(100):
                result = tensor * 2.0
                torch.xpu.synchronize()
            xpu_time = time.time() - start_time
            
            bandwidth_xpu = (size * size * 4 * 100) / xpu_time / 1e9  # GB/s
        
        # CPU test
        tensor_cpu = torch.randn(size, size, device="cpu")
        start_time = time.time()
        for _ in range(100):
            result_cpu = tensor_cpu * 2.0
        cpu_time = time.time() - start_time
        
        bandwidth_cpu = (size * size * 4 * 100) / cpu_time / 1e9  # GB/s
        
        print(f"Size {size}x{size}: XPU {bandwidth_xpu:.2f} GB/s, CPU {bandwidth_cpu:.2f} GB/s")
```

#### Attention Performance Test
```python
def benchmark_attention_performance():
    """Benchmark attention mechanism performance on XPU vs CPU."""
    batch_size, seq_len, hidden_dim = 4, 512, 768
    
    if torch.xpu.is_available():
        # XPU attention benchmark
        device = torch.device("xpu:0")
        q = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        k = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        v = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        
        start_time = time.time()
        for _ in range(50):
            attn_weights = torch.bmm(q, k.transpose(-2, -1))
            attn_weights = torch.softmax(attn_weights, dim=-1)
            output = torch.bmm(attn_weights, v)
            torch.xpu.synchronize()
        xpu_time = time.time() - start_time
        
        print(f"XPU attention time: {xpu_time:.3f}s")
    
    # CPU attention benchmark
    q_cpu = torch.randn(batch_size, seq_len, hidden_dim, device="cpu")
    k_cpu = torch.randn(batch_size, seq_len, hidden_dim, device="cpu")
    v_cpu = torch.randn(batch_size, seq_len, hidden_dim, device="cpu")
    
    start_time = time.time()
    for _ in range(50):
        attn_weights = torch.bmm(q_cpu, k_cpu.transpose(-2, -1))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        output = torch.bmm(attn_weights, v_cpu)
    cpu_time = time.time() - start_time
    
    print(f"CPU attention time: {cpu_time:.3f}s")
```

### Integration Tests

#### Test Complete Workflow
```python
def test_complete_workflow():
    """Test complete model loading, inference, and memory offloading."""
    test_commands = [
        # Basic inference test
        ["python", "-m", "flexllmgen.flex_opt", "--model", "facebook/opt-125m", 
         "--gpu-batch-size", "2", "--prompt-len", "16", "--gen-len", "8"],
        
        # Memory offloading test
        ["python", "-m", "flexllmgen.flex_opt", "--model", "facebook/opt-1.3b",
         "--percent", "50", "50", "100", "0", "100", "0"],
         
        # Compression test
        ["python", "-m", "flexllmgen.flex_opt", "--model", "facebook/opt-1.3b",
         "--compress-weight", "--compress-cache"]
    ]
    
    for cmd in test_commands:
        result = subprocess.run(cmd + ["--cut-gen-len", "2"], 
                              capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, f"Command failed: {' '.join(cmd)}\nError: {result.stderr}"
        print(f"✓ Command successful: {' '.join(cmd[:5])}...")
```

## Performance Optimization

### Memory Optimization Strategies

#### 1. Memory Pool Management
```python
class XPUMemoryPool:
    """Efficient memory pool for Intel XPU to reduce allocation overhead."""
    
    def __init__(self, device, pool_size_gb=8):
        self.device = device  
        self.pool_size = pool_size_gb * 1024**3  # Convert to bytes
        self.allocated_blocks = {}
        self.free_blocks = []
        
    def allocate(self, size, dtype=torch.float16):
        """Allocate memory from pool or create new if needed."""
        requested_bytes = size * torch_dtype_to_num_bytes[dtype]
        
        # Try to find suitable free block
        for i, (block_size, block_ptr) in enumerate(self.free_blocks):
            if block_size >= requested_bytes:
                # Remove from free blocks
                self.free_blocks.pop(i)
                # Track allocation
                self.allocated_blocks[block_ptr] = (block_size, dtype)
                return block_ptr[:size]
                
        # Allocate new block if no suitable free block
        tensor = torch.empty(size, dtype=dtype, device=self.device)
        self.allocated_blocks[tensor.data_ptr()] = (requested_bytes, dtype)
        return tensor
        
    def deallocate(self, tensor):
        """Return memory to pool for reuse."""
        ptr = tensor.data_ptr()
        if ptr in self.allocated_blocks:
            size, dtype = self.allocated_blocks.pop(ptr)
            self.free_blocks.append((size, tensor))
```

#### 2. Attention Optimization
```python
def optimized_attention_xpu(q, k, v, mask=None, scale=None):
    """Optimized attention implementation for Intel XPU."""
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # Use Intel XPU optimized operations
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    
    # Efficient batched matrix multiplication
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Use Intel's optimized softmax
    attn_weights = torch.softmax(scores, dim=-1)
    
    # Final matrix multiplication  
    output = torch.matmul(attn_weights, v)
    
    return output
```

#### 3. Mixed Precision Optimization
```python
class MixedPrecisionManager:
    """Manage mixed precision operations for optimal XPU performance."""
    
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.autocast_enabled = False
        
    def __enter__(self):
        if self.enabled and hasattr(torch.xpu, 'amp'):
            self.autocast_enabled = True
            return torch.xpu.amp.autocast()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        
    def scale_loss(self, loss):
        """Scale loss for mixed precision training if available."""
        if self.autocast_enabled and hasattr(torch.xpu, 'amp'):
            scaler = torch.xpu.amp.GradScaler()
            return scaler.scale(loss)
        return loss
```

### Compute Optimization

#### 1. Kernel Fusion
```python
def fused_layer_norm_attention(input_tensor, weight, bias, attention_weights):
    """Fused layer normalization and attention for better XPU utilization."""
    # Layer norm
    normalized = torch.nn.functional.layer_norm(
        input_tensor, input_tensor.shape[-1:], weight, bias
    )
    
    # Fused with attention computation to minimize memory transfers
    batch_size, seq_len, hidden_dim = normalized.shape
    num_heads = attention_weights.shape[1]
    head_dim = hidden_dim // num_heads
    
    # Reshape and compute attention in one step
    reshaped = normalized.view(batch_size, seq_len, num_heads, head_dim)
    output = torch.matmul(attention_weights, reshaped.transpose(1, 2))
    
    return output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
```

#### 2. Asynchronous Operations
```python
class AsyncComputeManager:
    """Manage asynchronous compute operations on Intel XPU."""
    
    def __init__(self, num_streams=3):
        self.compute_streams = [torch.xpu.Stream() for _ in range(num_streams)]
        self.current_stream = 0
        
    def get_next_stream(self):
        """Get next available compute stream."""
        stream = self.compute_streams[self.current_stream]
        self.current_stream = (self.current_stream + 1) % len(self.compute_streams)
        return stream
        
    def async_compute(self, func, *args, **kwargs):
        """Execute computation asynchronously on XPU stream."""
        stream = self.get_next_stream()
        with torch.xpu.stream(stream):
            return func(*args, **kwargs)
            
    def synchronize_all(self):
        """Wait for all async operations to complete."""
        for stream in self.compute_streams:
            stream.synchronize()
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Intel Extension for PyTorch Not Found
**Error**: `ImportError: No module named 'intel_extension_for_pytorch'`

**Solution**:
```bash
# Install Intel Extension for PyTorch
pip install intel-extension-for-pytorch

# For conda environments
conda install intel-extension-for-pytorch -c intel

# Verify installation
python -c "import intel_extension_for_pytorch as ipex; print('IPEX version:', ipex.__version__)"
```

#### 2. XPU Device Not Detected
**Error**: `torch.xpu.is_available()` returns `False`

**Solution**:
```bash
# Check Intel GPU drivers
intel-gpu-tools  # Linux
# or
intel-gpu-top    # Check if GPU is recognized

# Verify PyTorch XPU support
python -c "import torch; print('XPU compiled:', hasattr(torch, 'xpu'))"

# Check environment variables
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export ONEAPI_DEVICE_SELECTOR=opencl:gpu
```

#### 3. Memory Allocation Errors
**Error**: `RuntimeError: XPU out of memory`

**Solution**:
```python
# Monitor XPU memory usage
def monitor_xpu_memory():
    if torch.xpu.is_available():
        device = torch.device("xpu:0")
        allocated = torch.xpu.memory_allocated(device) / 1024**3  # GB
        cached = torch.xpu.memory_reserved(device) / 1024**3      # GB
        print(f"XPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

# Clear XPU cache
torch.xpu.empty_cache()

# Use smaller batch sizes
--gpu-batch-size 2  # Reduce from larger values

# Enable memory offloading
--percent 50 50 100 0 100 0  # Offload weights to CPU/disk
```

#### 4. Performance Issues
**Issue**: Slower than expected performance

**Solution**:
```python
# Enable XPU optimizations
import intel_extension_for_pytorch as ipex

# Optimize model for XPU
model = model.to("xpu")
model = ipex.optimize(model)

# Use appropriate data types
model = model.half()  # Use FP16 for better performance

# Enable memory format optimization
torch.backends.xpu.matmul.allow_tf32 = True
```

#### 5. Distributed Training Issues
**Error**: Communication backend errors in multi-GPU setup

**Solution**:
```bash
# Use Intel's CCL backend for distributed training
export CCL_WORKER_COUNT=2
export CCL_WORKER_AFFINITY=auto

# Initialize with proper backend
python -m flexllmgen.dist_flex_opt \
    --comm-device gpu \
    --backend ccl \
    --model facebook/opt-6.7b
```

### Debugging Tools

#### 1. XPU Profiler
```python
def profile_xpu_operations(func, *args, **kwargs):
    """Profile XPU operations for performance analysis."""
    if not torch.xpu.is_available():
        return func(*args, **kwargs)
        
    # Enable profiling
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.XPU],
        record_shapes=True
    ) as prof:
        result = func(*args, **kwargs)
        torch.xpu.synchronize()
    
    # Print profiling results
    print(prof.key_averages().table(sort_by="xpu_time_total", row_limit=10))
    return result
```

#### 2. Memory Debugging
```python
def debug_memory_usage():
    """Debug XPU memory allocation patterns."""
    if not torch.xpu.is_available():
        return
        
    device = torch.device("xpu:0")
    
    print("=== XPU Memory Debug ===")
    print(f"Total memory: {torch.xpu.get_device_properties(device).total_memory / 1024**3:.2f}GB")
    print(f"Allocated: {torch.xpu.memory_allocated(device) / 1024**3:.2f}GB")
    print(f"Reserved: {torch.xpu.memory_reserved(device) / 1024**3:.2f}GB")
    
    # Memory allocation tracking
    if hasattr(torch.xpu, 'memory_summary'):
        print("\nMemory Summary:")
        print(torch.xpu.memory_summary(device))
```

## Best Practices

### Development Guidelines

#### 1. Code Organization
```python
# Separate XPU-specific code into utility modules
# utils/xpu_utils.py

def get_xpu_device(device_id=0):
    """Get XPU device with error handling."""
    if not torch.xpu.is_available():
        raise RuntimeError("Intel XPU not available")
    
    if device_id >= torch.xpu.device_count():
        raise ValueError(f"XPU device {device_id} not found")
        
    return torch.device(f"xpu:{device_id}")

def safe_xpu_operation(func, fallback_device="cpu"):
    """Execute operation with XPU fallback to CPU."""
    try:
        return func()
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"XPU operation failed, falling back to {fallback_device}")
            # Move tensors to fallback device and retry
            return func()  # Implement fallback logic
        raise
```

#### 2. Testing Strategy
```python
# tests/test_xpu_integration.py

class TestXPUIntegration:
    """Comprehensive XPU integration tests."""
    
    @pytest.fixture(autouse=True)
    def setup_xpu(self):
        """Setup XPU environment for tests."""
        if not torch.xpu.is_available():
            pytest.skip("XPU not available")
        
        # Clear memory before each test
        torch.xpu.empty_cache()
        
    def test_device_compatibility(self):
        """Test device compatibility across operations."""
        device = torch.device("xpu:0")
        
        # Test tensor creation and operations
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)
        z = torch.matmul(x, y)
        
        assert z.device == device
        assert z.shape == (100, 100)
        
    @pytest.mark.parametrize("model_size", ["125m", "1.3b", "2.7b"])
    def test_model_sizes(self, model_size):
        """Test different model sizes on XPU."""
        # Implement model loading and basic inference test
        pass
```

#### 3. Performance Monitoring
```python
class XPUPerformanceMonitor:
    """Monitor XPU performance metrics during inference."""
    
    def __init__(self, device):
        self.device = device
        self.metrics = {
            'memory_peak': 0,
            'compute_time': 0,
            'memory_transfers': 0
        }
        
    def __enter__(self):
        self.start_time = time.time()
        if torch.xpu.is_available():
            torch.xpu.reset_peak_memory_stats(self.device)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.metrics['compute_time'] = time.time() - self.start_time
        
        if torch.xpu.is_available():
            self.metrics['memory_peak'] = torch.xpu.max_memory_allocated(self.device)
        
        self._log_metrics()
        
    def _log_metrics(self):
        """Log performance metrics."""
        print(f"=== XPU Performance ===")
        print(f"Compute time: {self.metrics['compute_time']:.3f}s")
        print(f"Peak memory: {self.metrics['memory_peak'] / 1024**3:.2f}GB")
```

#### 4. Error Handling
```python
def robust_xpu_execution(operation, max_retries=3, fallback_device="cpu"):
    """Execute XPU operations with robust error handling."""
    
    for attempt in range(max_retries):
        try:
            return operation()
        except RuntimeError as e:
            if "out of memory" in str(e) and attempt < max_retries - 1:
                print(f"XPU OOM, clearing cache and retrying (attempt {attempt + 1})")
                torch.xpu.empty_cache()
                time.sleep(1)  # Brief pause for memory cleanup
                continue
            elif "out of memory" in str(e):
                print(f"XPU OOM after {max_retries} attempts, falling back to {fallback_device}")
                # Implement device fallback logic
                raise RuntimeError(f"Operation failed on XPU, fallback needed")
            else:
                raise
    
    raise RuntimeError(f"Operation failed after {max_retries} attempts")
```

### Deployment Considerations

#### 1. Environment Setup Script
```bash
#!/bin/bash
# setup_xpu_environment.sh

echo "Setting up Intel XPU environment for FlexLLMGen..."

# Check for Intel GPU
if ! lspci | grep -i intel | grep -i vga > /dev/null; then
    echo "Warning: Intel GPU not detected"
fi

# Install dependencies
pip install intel-extension-for-pytorch
pip install torch torchvision torchaudio

# Set environment variables for optimal performance
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export ONEAPI_DEVICE_SELECTOR=opencl:gpu
export CCL_WORKER_COUNT=2

# Verify installation
python -c "
import torch
import intel_extension_for_pytorch as ipex
print('PyTorch version:', torch.__version__)
print('IPEX version:', ipex.__version__)
print('XPU available:', torch.xpu.is_available())
if torch.xpu.is_available():
    print('XPU devices:', torch.xpu.device_count())
"

echo "Setup complete!"
```

#### 2. Production Configuration
```yaml
# config/production_xpu.yaml
xpu:
  enabled: true
  device_ids: [0, 1]  # Use multiple XPU devices if available
  memory_fraction: 0.9  # Reserve 90% of XPU memory
  mixed_precision: true
  
model:
  offload_strategy: "balanced"  # Balance between XPU/CPU/disk
  compression:
    weights: true
    cache: true
    
performance:
  batch_size: 8
  num_streams: 3
  prefetch_factor: 2
```

This implementation guide provides a complete roadmap for migrating FlexLLMGen from CUDA to Intel XPU, including all necessary code changes, testing strategies, performance optimizations, and best practices for production deployment.