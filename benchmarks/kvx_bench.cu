#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <vector>

#include "../kernels/kvx_paged_kv.h"
#include "../kvx_abi.h"

typedef struct BenchConfig {
  int32_t seq_count;
  int32_t seq_len;
  int32_t tokens_per_seq;
  int32_t block_size;
  int32_t num_heads;
  int32_t head_dim;
  int32_t warmup;
  int32_t iters;
  kvx_dtype_t dtype;
  size_t elem_size;
} BenchConfig;

static void set_default_config(BenchConfig *cfg) {
  cfg->seq_count = 8;
  cfg->seq_len = 128;
  cfg->tokens_per_seq = 1;
  cfg->block_size = 16;
  cfg->num_heads = 8;
  cfg->head_dim = 128;
  cfg->warmup = 10;
  cfg->iters = 50;
  cfg->dtype = KVX_DTYPE_F32;
  cfg->elem_size = sizeof(float);
}

static int parse_arg_int(const char *arg, const char *name, int32_t *out,
                         int *idx, int argc, char **argv) {
  size_t len = strlen(name);
  if (strncmp(arg, name, len) != 0) {
    return 0;
  }
  if (arg[len] == '=') {
    *out = atoi(arg + len + 1);
    return 1;
  }
  if (arg[len] == '\0' && *idx + 1 < argc) {
    *out = atoi(argv[*idx + 1]);
    (*idx)++;
    return 1;
  }
  return 0;
}

static int parse_arg_dtype(const char *arg, BenchConfig *cfg, int *idx,
                           int argc, char **argv) {
  const char *name = "--dtype";
  size_t len = strlen(name);
  if (strncmp(arg, name, len) != 0) {
    return 0;
  }
  const char *value = NULL;
  if (arg[len] == '=') {
    value = arg + len + 1;
  } else if (arg[len] == '\0' && *idx + 1 < argc) {
    value = argv[*idx + 1];
    (*idx)++;
  } else {
    return 0;
  }
  if (strcmp(value, "float32") == 0) {
    cfg->dtype = KVX_DTYPE_F32;
    cfg->elem_size = sizeof(float);
    return 1;
  }
  if (strcmp(value, "float16") == 0) {
    cfg->dtype = KVX_DTYPE_F16;
    cfg->elem_size = sizeof(__half);
    return 1;
  }
  if (strcmp(value, "bfloat16") == 0) {
    cfg->dtype = KVX_DTYPE_BF16;
    cfg->elem_size = sizeof(__nv_bfloat16);
    return 1;
  }
  fprintf(stderr, "Unsupported --dtype=%s\n", value);
  exit(1);
}

static void parse_args(int argc, char **argv, BenchConfig *cfg) {
  for (int i = 1; i < argc; ++i) {
    if (parse_arg_int(argv[i], "--seq-count", &cfg->seq_count, &i, argc,
                      argv)) {
      continue;
    }
    if (parse_arg_int(argv[i], "--seq-len", &cfg->seq_len, &i, argc, argv)) {
      continue;
    }
    if (parse_arg_int(argv[i], "--tokens-per-seq", &cfg->tokens_per_seq, &i,
                      argc, argv)) {
      continue;
    }
    if (parse_arg_int(argv[i], "--block-size", &cfg->block_size, &i, argc,
                      argv)) {
      continue;
    }
    if (parse_arg_int(argv[i], "--num-heads", &cfg->num_heads, &i, argc,
                      argv)) {
      continue;
    }
    if (parse_arg_int(argv[i], "--head-dim", &cfg->head_dim, &i, argc, argv)) {
      continue;
    }
    if (parse_arg_int(argv[i], "--warmup", &cfg->warmup, &i, argc, argv)) {
      continue;
    }
    if (parse_arg_int(argv[i], "--iters", &cfg->iters, &i, argc, argv)) {
      continue;
    }
    if (parse_arg_dtype(argv[i], cfg, &i, argc, argv)) {
      continue;
    }
  }
}

static void set_tensor_desc_nhd(kvx_tensor_desc_t *tensor, uint32_t num_blocks,
                                uint32_t block_size, uint32_t num_heads,
                                uint32_t head_dim, kvx_dtype_t dtype,
                                void *data) {
  memset(tensor, 0, sizeof(*tensor));
  tensor->size = sizeof(*tensor);
  tensor->dtype = dtype;
  tensor->layout = KVX_LAYOUT_BLOCK_NHD;
  tensor->memory = KVX_MEMORY_DEVICE;
  tensor->ndim = 4;
  tensor->shape[0] = num_blocks;
  tensor->shape[1] = block_size;
  tensor->shape[2] = num_heads;
  tensor->shape[3] = head_dim;
  tensor->stride[3] = 1;
  tensor->stride[2] = head_dim;
  tensor->stride[1] = (int64_t)num_heads * head_dim;
  tensor->stride[0] = (int64_t)block_size * num_heads * head_dim;
  tensor->data = data;
}

static void set_tensor_desc_tokens(kvx_tensor_desc_t *tensor, uint32_t num_tokens,
                                   uint32_t num_heads, uint32_t head_dim,
                                   kvx_dtype_t dtype, void *data,
                                   kvx_memory_type_t memory) {
  memset(tensor, 0, sizeof(*tensor));
  tensor->size = sizeof(*tensor);
  tensor->dtype = dtype;
  tensor->layout = KVX_LAYOUT_BLOCK_CUSTOM;
  tensor->memory = memory;
  tensor->ndim = 3;
  tensor->shape[0] = num_tokens;
  tensor->shape[1] = num_heads;
  tensor->shape[2] = head_dim;
  tensor->stride[2] = 1;
  tensor->stride[1] = head_dim;
  tensor->stride[0] = (int64_t)num_heads * head_dim;
  tensor->data = data;
}

static void set_tensor_desc_seq(kvx_tensor_desc_t *tensor, uint32_t seq_count,
                                uint32_t max_seq_len, uint32_t num_heads,
                                uint32_t head_dim, kvx_dtype_t dtype,
                                void *data) {
  memset(tensor, 0, sizeof(*tensor));
  tensor->size = sizeof(*tensor);
  tensor->dtype = dtype;
  tensor->layout = KVX_LAYOUT_BLOCK_CUSTOM;
  tensor->memory = KVX_MEMORY_DEVICE;
  tensor->ndim = 4;
  tensor->shape[0] = seq_count;
  tensor->shape[1] = max_seq_len;
  tensor->shape[2] = num_heads;
  tensor->shape[3] = head_dim;
  tensor->stride[3] = 1;
  tensor->stride[2] = head_dim;
  tensor->stride[1] = (int64_t)num_heads * head_dim;
  tensor->stride[0] = (int64_t)max_seq_len * num_heads * head_dim;
  tensor->data = data;
}

static void check_cuda(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
    exit(1);
  }
}

static void fill_host_pattern(std::vector<float> *out, float base) {
  for (size_t i = 0; i < out->size(); ++i) {
    (*out)[i] = base + 0.01f * (float)i;
  }
}

static std::vector<__half> to_half(const std::vector<float> &src) {
  std::vector<__half> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    out[i] = __float2half_rn(src[i]);
  }
  return out;
}

static std::vector<__nv_bfloat16> to_bfloat16(const std::vector<float> &src) {
  std::vector<__nv_bfloat16> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    out[i] = __float2bfloat16(src[i]);
  }
  return out;
}

int main(int argc, char **argv) {
  BenchConfig cfg;
  set_default_config(&cfg);
  parse_args(argc, argv, &cfg);

  if (cfg.seq_count <= 0 || cfg.seq_len <= 0 || cfg.tokens_per_seq <= 0 ||
      cfg.block_size <= 0 ||
      cfg.num_heads <= 0 || cfg.head_dim <= 0 || cfg.iters <= 0) {
    fprintf(stderr, "Invalid config values.\n");
    return 1;
  }
  if (cfg.tokens_per_seq > cfg.seq_len) {
    fprintf(stderr, "tokens_per_seq cannot exceed seq_len.\n");
    return 1;
  }

  int32_t max_blocks_per_seq =
      (cfg.seq_len + cfg.block_size - 1) / cfg.block_size;
  int32_t num_blocks = cfg.seq_count * max_blocks_per_seq;

  int32_t num_tokens = cfg.seq_count * cfg.tokens_per_seq;
  size_t token_elems =
      (size_t)num_tokens * cfg.num_heads * cfg.head_dim;
  size_t cache_elems =
      (size_t)num_blocks * cfg.block_size * cfg.num_heads * cfg.head_dim;
  size_t gather_elems =
      (size_t)cfg.seq_count * cfg.seq_len * cfg.num_heads * cfg.head_dim;

  std::vector<float> h_key_f32(token_elems);
  std::vector<float> h_value_f32(token_elems);
  fill_host_pattern(&h_key_f32, 0.1f);
  fill_host_pattern(&h_value_f32, 0.2f);

  std::vector<int64_t> h_slots(num_tokens, -1);
  int32_t start_pos = cfg.seq_len - cfg.tokens_per_seq;
  for (int32_t s = 0; s < cfg.seq_count; ++s) {
    for (int32_t t = 0; t < cfg.tokens_per_seq; ++t) {
      int32_t token_pos = start_pos + t;
      int32_t logical_block = token_pos / cfg.block_size;
      int32_t block_offset = token_pos - logical_block * cfg.block_size;
      int32_t block_id = s * max_blocks_per_seq + logical_block;
      int32_t token_idx = s * cfg.tokens_per_seq + t;
      h_slots[token_idx] =
          (int64_t)block_id * cfg.block_size + (int64_t)block_offset;
    }
  }

  std::vector<int32_t> h_block_table(
      (size_t)cfg.seq_count * max_blocks_per_seq, 0);
  for (int32_t s = 0; s < cfg.seq_count; ++s) {
    for (int32_t b = 0; b < max_blocks_per_seq; ++b) {
      h_block_table[s * max_blocks_per_seq + b] = s * max_blocks_per_seq + b;
    }
  }

  std::vector<int32_t> h_seq_lens(cfg.seq_count, cfg.seq_len);

  void *d_key = NULL;
  void *d_value = NULL;
  void *d_key_cache = NULL;
  void *d_value_cache = NULL;
  void *d_key_out = NULL;
  void *d_value_out = NULL;
  int64_t *d_slots = NULL;
  int32_t *d_block_table = NULL;
  int32_t *d_seq_lens = NULL;

  check_cuda(cudaMalloc(&d_key, token_elems * cfg.elem_size),
             "cudaMalloc key");
  check_cuda(cudaMalloc(&d_value, token_elems * cfg.elem_size),
             "cudaMalloc value");
  check_cuda(cudaMalloc(&d_key_cache, cache_elems * cfg.elem_size),
             "cudaMalloc key_cache");
  check_cuda(cudaMalloc(&d_value_cache, cache_elems * cfg.elem_size),
             "cudaMalloc value_cache");
  check_cuda(cudaMalloc(&d_key_out, gather_elems * cfg.elem_size),
             "cudaMalloc key_out");
  check_cuda(cudaMalloc(&d_value_out, gather_elems * cfg.elem_size),
             "cudaMalloc value_out");
  check_cuda(cudaMalloc(&d_slots, num_tokens * sizeof(int64_t)),
             "cudaMalloc slots");
  check_cuda(
      cudaMalloc(&d_block_table,
                 (size_t)cfg.seq_count * max_blocks_per_seq *
                     sizeof(int32_t)),
      "cudaMalloc block_table");
  check_cuda(cudaMalloc(&d_seq_lens, cfg.seq_count * sizeof(int32_t)),
             "cudaMalloc seq_lens");

  if (cfg.dtype == KVX_DTYPE_F32) {
    check_cuda(cudaMemcpy(d_key, h_key_f32.data(),
                          token_elems * sizeof(float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy key");
    check_cuda(cudaMemcpy(d_value, h_value_f32.data(),
                          token_elems * sizeof(float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy value");
  } else if (cfg.dtype == KVX_DTYPE_F16) {
    std::vector<__half> h_key = to_half(h_key_f32);
    std::vector<__half> h_value = to_half(h_value_f32);
    check_cuda(cudaMemcpy(d_key, h_key.data(),
                          token_elems * sizeof(__half),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy key");
    check_cuda(cudaMemcpy(d_value, h_value.data(),
                          token_elems * sizeof(__half),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy value");
  } else if (cfg.dtype == KVX_DTYPE_BF16) {
    std::vector<__nv_bfloat16> h_key = to_bfloat16(h_key_f32);
    std::vector<__nv_bfloat16> h_value = to_bfloat16(h_value_f32);
    check_cuda(cudaMemcpy(d_key, h_key.data(),
                          token_elems * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy key");
    check_cuda(cudaMemcpy(d_value, h_value.data(),
                          token_elems * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy value");
  } else {
    fprintf(stderr, "Unsupported dtype.\n");
    return 1;
  }
  check_cuda(cudaMemcpy(d_slots, h_slots.data(),
                        num_tokens * sizeof(int64_t),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy slots");
  check_cuda(
      cudaMemcpy(d_block_table, h_block_table.data(),
                 (size_t)cfg.seq_count * max_blocks_per_seq *
                     sizeof(int32_t),
                 cudaMemcpyHostToDevice),
      "cudaMemcpy block_table");
  check_cuda(cudaMemcpy(d_seq_lens, h_seq_lens.data(),
                        cfg.seq_count * sizeof(int32_t),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy seq_lens");

  kvx_cache_desc_t cache;
  memset(&cache, 0, sizeof(cache));
  cache.size = sizeof(cache);
  cache.num_blocks = num_blocks;
  cache.block_size = cfg.block_size;
  cache.num_kv_heads = cfg.num_heads;
  cache.head_dim = cfg.head_dim;
  set_tensor_desc_nhd(&cache.k, num_blocks, cfg.block_size, cfg.num_heads,
                      cfg.head_dim, cfg.dtype, d_key_cache);
  set_tensor_desc_nhd(&cache.v, num_blocks, cfg.block_size, cfg.num_heads,
                      cfg.head_dim, cfg.dtype, d_value_cache);

  kvx_write_desc_t write_desc;
  memset(&write_desc, 0, sizeof(write_desc));
  write_desc.size = sizeof(write_desc);
  write_desc.io.size = sizeof(write_desc.io);
  write_desc.io.num_tokens = num_tokens;
  write_desc.io.num_kv_heads = cfg.num_heads;
  write_desc.io.head_dim = cfg.head_dim;
  set_tensor_desc_tokens(&write_desc.io.key, num_tokens, cfg.num_heads,
                         cfg.head_dim, cfg.dtype, d_key, KVX_MEMORY_DEVICE);
  set_tensor_desc_tokens(&write_desc.io.value, num_tokens, cfg.num_heads,
                         cfg.head_dim, cfg.dtype, d_value, KVX_MEMORY_DEVICE);
  write_desc.slots.size = sizeof(write_desc.slots);
  write_desc.slots.dtype = KVX_DTYPE_S64;
  write_desc.slots.token_count = num_tokens;
  write_desc.slots.invalid_slot = -1;
  write_desc.slots.slots = d_slots;

  kvx_gather_desc_t gather_desc;
  memset(&gather_desc, 0, sizeof(gather_desc));
  gather_desc.size = sizeof(gather_desc);
  gather_desc.io.size = sizeof(gather_desc.io);
  gather_desc.io.num_tokens = cfg.seq_count * cfg.seq_len;
  gather_desc.io.num_kv_heads = cfg.num_heads;
  gather_desc.io.head_dim = cfg.head_dim;
  set_tensor_desc_seq(&gather_desc.io.key, cfg.seq_count, cfg.seq_len,
                      cfg.num_heads, cfg.head_dim, cfg.dtype, d_key_out);
  set_tensor_desc_seq(&gather_desc.io.value, cfg.seq_count, cfg.seq_len,
                      cfg.num_heads, cfg.head_dim, cfg.dtype, d_value_out);
  gather_desc.block_table.size = sizeof(gather_desc.block_table);
  gather_desc.block_table.format = KVX_BLOCK_TABLE_PACKED;
  gather_desc.block_table.index_dtype = KVX_DTYPE_S32;
  gather_desc.block_table.seq_count = cfg.seq_count;
  gather_desc.block_table.beam_width = 1;
  gather_desc.block_table.max_blocks_per_seq = max_blocks_per_seq;
  gather_desc.block_table.indices = d_block_table;
  gather_desc.block_table.indices_count =
      cfg.seq_count * max_blocks_per_seq;
  gather_desc.block_table.indptr = NULL;
  gather_desc.block_table.indptr_count = 0;
  gather_desc.block_table.flags = 0;
  gather_desc.seq_lens.size = sizeof(gather_desc.seq_lens);
  gather_desc.seq_lens.dtype = KVX_DTYPE_S32;
  gather_desc.seq_lens.seq_count = cfg.seq_count;
  gather_desc.seq_lens.lengths = d_seq_lens;
  gather_desc.max_seq_len = cfg.seq_len;

  cudaDeviceProp prop;
  check_cuda(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
  int driver_version = 0;
  int runtime_version = 0;
  check_cuda(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
  check_cuda(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
  printf("GPU: %s, SM %d.%d, driver %d, runtime %d\n", prop.name,
         prop.major, prop.minor, driver_version, runtime_version);

  bool use_prefill = (cfg.tokens_per_seq > 1);
  for (int i = 0; i < cfg.warmup; ++i) {
    kvx_status_t st = KVX_STATUS_UNSUPPORTED;
    if (cfg.dtype == KVX_DTYPE_F32) {
      st = use_prefill
               ? kvx_launch_write_kv_prefill_f32(&cache, &write_desc,
                                                 cfg.tokens_per_seq, NULL)
               : kvx_launch_write_kv_f32(&cache, &write_desc, NULL);
    } else if (cfg.dtype == KVX_DTYPE_F16) {
      st = use_prefill
               ? kvx_launch_write_kv_prefill_f16(&cache, &write_desc,
                                                 cfg.tokens_per_seq, NULL)
               : kvx_launch_write_kv_f16(&cache, &write_desc, NULL);
    } else if (cfg.dtype == KVX_DTYPE_BF16) {
      st = use_prefill
               ? kvx_launch_write_kv_prefill_bf16(&cache, &write_desc,
                                                  cfg.tokens_per_seq, NULL)
               : kvx_launch_write_kv_bf16(&cache, &write_desc, NULL);
    }
    if (st != KVX_STATUS_OK) {
      fprintf(stderr, "write warmup failed: %d\n", st);
      return 1;
    }
  }
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize write warmup");

  cudaEvent_t write_start;
  cudaEvent_t write_stop;
  check_cuda(cudaEventCreate(&write_start), "cudaEventCreate write_start");
  check_cuda(cudaEventCreate(&write_stop), "cudaEventCreate write_stop");
  check_cuda(cudaEventRecord(write_start, 0), "cudaEventRecord write_start");
  for (int i = 0; i < cfg.iters; ++i) {
    kvx_status_t st = KVX_STATUS_UNSUPPORTED;
    if (cfg.dtype == KVX_DTYPE_F32) {
      st = use_prefill
               ? kvx_launch_write_kv_prefill_f32(&cache, &write_desc,
                                                 cfg.tokens_per_seq, NULL)
               : kvx_launch_write_kv_f32(&cache, &write_desc, NULL);
    } else if (cfg.dtype == KVX_DTYPE_F16) {
      st = use_prefill
               ? kvx_launch_write_kv_prefill_f16(&cache, &write_desc,
                                                 cfg.tokens_per_seq, NULL)
               : kvx_launch_write_kv_f16(&cache, &write_desc, NULL);
    } else if (cfg.dtype == KVX_DTYPE_BF16) {
      st = use_prefill
               ? kvx_launch_write_kv_prefill_bf16(&cache, &write_desc,
                                                  cfg.tokens_per_seq, NULL)
               : kvx_launch_write_kv_bf16(&cache, &write_desc, NULL);
    }
    if (st != KVX_STATUS_OK) {
      fprintf(stderr, "write iter failed: %d\n", st);
      return 1;
    }
  }
  check_cuda(cudaEventRecord(write_stop, 0), "cudaEventRecord write_stop");
  check_cuda(cudaEventSynchronize(write_stop),
             "cudaEventSynchronize write_stop");
  float write_ms = 0.0f;
  check_cuda(cudaEventElapsedTime(&write_ms, write_start, write_stop),
             "cudaEventElapsedTime write");

  for (int i = 0; i < cfg.warmup; ++i) {
    kvx_status_t st = KVX_STATUS_UNSUPPORTED;
    if (cfg.dtype == KVX_DTYPE_F32) {
      st = kvx_launch_gather_kv_f32(&cache, &gather_desc, NULL);
    } else if (cfg.dtype == KVX_DTYPE_F16) {
      st = kvx_launch_gather_kv_f16(&cache, &gather_desc, NULL);
    } else if (cfg.dtype == KVX_DTYPE_BF16) {
      st = kvx_launch_gather_kv_bf16(&cache, &gather_desc, NULL);
    }
    if (st != KVX_STATUS_OK) {
      fprintf(stderr, "gather warmup failed: %d\n", st);
      return 1;
    }
  }
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize gather warmup");

  cudaEvent_t gather_start;
  cudaEvent_t gather_stop;
  check_cuda(cudaEventCreate(&gather_start), "cudaEventCreate gather_start");
  check_cuda(cudaEventCreate(&gather_stop), "cudaEventCreate gather_stop");
  check_cuda(cudaEventRecord(gather_start, 0),
             "cudaEventRecord gather_start");
  for (int i = 0; i < cfg.iters; ++i) {
    kvx_status_t st = KVX_STATUS_UNSUPPORTED;
    if (cfg.dtype == KVX_DTYPE_F32) {
      st = kvx_launch_gather_kv_f32(&cache, &gather_desc, NULL);
    } else if (cfg.dtype == KVX_DTYPE_F16) {
      st = kvx_launch_gather_kv_f16(&cache, &gather_desc, NULL);
    } else if (cfg.dtype == KVX_DTYPE_BF16) {
      st = kvx_launch_gather_kv_bf16(&cache, &gather_desc, NULL);
    }
    if (st != KVX_STATUS_OK) {
      fprintf(stderr, "gather iter failed: %d\n", st);
      return 1;
    }
  }
  check_cuda(cudaEventRecord(gather_stop, 0),
             "cudaEventRecord gather_stop");
  check_cuda(cudaEventSynchronize(gather_stop),
             "cudaEventSynchronize gather_stop");
  float gather_ms = 0.0f;
  check_cuda(cudaEventElapsedTime(&gather_ms, gather_start, gather_stop),
             "cudaEventElapsedTime gather");

  double write_ms_per = write_ms / cfg.iters;
  double gather_ms_per = gather_ms / cfg.iters;

  double write_tokens_per_s =
      (double)num_tokens / (write_ms_per / 1e3);
  double gather_tokens_per_s =
      (double)(cfg.seq_count * cfg.seq_len) / (gather_ms_per / 1e3);

  size_t write_bytes = token_elems * cfg.elem_size * 4;
  size_t gather_bytes = gather_elems * cfg.elem_size * 4;
  double write_gb_s =
      (double)write_bytes / (write_ms_per / 1e3) / (1024.0 * 1024.0 * 1024.0);
  double gather_gb_s =
      (double)gather_bytes / (gather_ms_per / 1e3) / (1024.0 * 1024.0 * 1024.0);

  printf("write: %.3f ms, %.2f tokens/s, %.2f GB/s\n", write_ms_per,
         write_tokens_per_s, write_gb_s);
  printf("gather: %.3f ms, %.2f tokens/s, %.2f GB/s\n", gather_ms_per,
         gather_tokens_per_s, gather_gb_s);

  printf("kernel,seq_count,seq_len,block_size,num_heads,head_dim,num_blocks,"
         "iters,warmup,ms_per,tokens_per_s,gb_per_s,tokens_per_seq\n");
  printf("write,%d,%d,%d,%d,%d,%d,%d,%d,%.6f,%.2f,%.2f,%d\n", cfg.seq_count,
         cfg.seq_len, cfg.block_size, cfg.num_heads, cfg.head_dim, num_blocks,
         cfg.iters, cfg.warmup, write_ms_per, write_tokens_per_s, write_gb_s,
         cfg.tokens_per_seq);
  printf("gather,%d,%d,%d,%d,%d,%d,%d,%d,%.6f,%.2f,%.2f,%d\n", cfg.seq_count,
         cfg.seq_len, cfg.block_size, cfg.num_heads, cfg.head_dim, num_blocks,
         cfg.iters, cfg.warmup, gather_ms_per, gather_tokens_per_s, gather_gb_s,
         cfg.tokens_per_seq);

  cudaEventDestroy(write_start);
  cudaEventDestroy(write_stop);
  cudaEventDestroy(gather_start);
  cudaEventDestroy(gather_stop);

  cudaFree(d_key);
  cudaFree(d_value);
  cudaFree(d_key_cache);
  cudaFree(d_value_cache);
  cudaFree(d_key_out);
  cudaFree(d_value_out);
  cudaFree(d_slots);
  cudaFree(d_block_table);
  cudaFree(d_seq_lens);

  return 0;
}
