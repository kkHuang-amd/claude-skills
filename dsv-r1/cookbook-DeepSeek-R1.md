Source URL: https://lmsysorg.mintlify.app/cookbook/autoregressive/DeepSeek/DeepSeek-R1
Title: Hardware Platform

> ## Documentation Index
> Fetch the complete documentation index at: https://docs.sglang.io/llms.txt
> Use this file to discover all available pages before exploring further.

# DeepSeek-R1

export const DeepSeekR1AdvancedDeployment = () => {
  const lookupData = {
    "model": "deepseek-r1",
    "version": "v0.5.6",
    "ui_options": {
      "hardware": [{
        "id": "b200",
        "label": "B200",
        "default": true
      }, {
        "id": "b300",
        "label": "B300",
        "default": false
      }, {
        "id": "h200",
        "label": "H200",
        "default": false
      }, {
        "id": "mi300x",
        "label": "MI300X",
        "default": false
      }, {
        "id": "mi325x",
        "label": "MI325X",
        "default": false
      }, {
        "id": "mi355x",
        "label": "MI355X",
        "default": false
      }],
      "quantization": [{
        "id": "fp8",
        "label": "FP8",
        "default": true
      }, {
        "id": "fp4",
        "label": "FP4",
        "default": false
      }],
      "scenario": [{
        "id": "low-latency",
        "label": "Low Latency",
        "subtitle": "Concurrency 4-8",
        "default": true
      }, {
        "id": "high-throughput",
        "label": "High Throughput",
        "subtitle": "Concurrency 16-128",
        "default": false
      }],
      "gpu_count": [{
        "id": 4,
        "label": "4 GPUs",
        "default": false
      }, {
        "id": 8,
        "label": "8 GPUs",
        "default": true
      }]
    },
    "configs": [{
      "hardware": "b200",
      "quantization": "fp4",
      "gpu_count": 4,
      "scenario": "low-latency",
      "parameters": {
        "model_path": "nvidia/DeepSeek-R1-0528-FP4-v2",
        "tensor_parallel_size": 4,
        "cuda_graph_max_bs_decode": 256,
        "max_running_requests": 256,
        "mem_fraction_static": 0.85,
        "ep_size": 4,
        "scheduler_recv_interval": 10,
        "enable_symm_mem": true,
        "stream_interval": 10
      }
    }, {
      "hardware": "b200",
      "quantization": "fp4",
      "gpu_count": 4,
      "scenario": "high-throughput",
      "parameters": {
        "model_path": "nvidia/DeepSeek-R1-0528-FP4-v2",
        "tensor_parallel_size": 4,
        "cuda_graph_max_bs_decode": 256,
        "max_running_requests": 256,
        "mem_fraction_static": 0.85,
        "ep_size": 4,
        "scheduler_recv_interval": 30,
        "enable_symm_mem": true,
        "stream_interval": 10
      }
    }, {
      "hardware": "b200",
      "quantization": "fp4",
      "gpu_count": 8,
      "scenario": "low-latency",
      "parameters": {
        "model_path": "nvidia/DeepSeek-R1-0528-FP4-v2",
        "tensor_parallel_size": 8,
        "cuda_graph_max_bs_decode": 256,
        "max_running_requests": 256,
        "mem_fraction_static": 0.85,
        "kv_cache_dtype": "fp8_e4m3",
        "chunked_prefill_size": 16384,
        "ep_size": 8,
        "scheduler_recv_interval": 10,
        "enable_symm_mem": true,
        "stream_interval": 10
      }
    }, {
      "hardware": "b200",
      "quantization": "fp4",
      "gpu_count": 8,
      "scenario": "high-throughput",
      "parameters": {
        "model_path": "nvidia/DeepSeek-R1-0528-FP4-v2",
        "tensor_parallel_size": 8,
        "cuda_graph_max_bs_decode": 256,
        "max_running_requests": 256,
        "mem_fraction_static": 0.85,
        "kv_cache_dtype": "fp8_e4m3",
        "chunked_prefill_size": 16384,
        "ep_size": 8,
        "scheduler_recv_interval": 30,
        "enable_symm_mem": true,
        "stream_interval": 10
      }
    }, {
      "hardware": "b200",
      "quantization": "fp8",
      "gpu_count": 8,
      "scenario": "low-latency",
      "parameters": {
        "env_vars": "SGLANG_ENABLE_JIT_DEEPGEMM=false",
        "model_path": "deepseek-ai/DeepSeek-R1-0528",
        "tensor_parallel_size": 8,
        "cuda_graph_max_bs_decode": 128,
        "max_running_requests": 128,
        "mem_fraction_static": 0.82,
        "kv_cache_dtype": "fp8_e4m3",
        "chunked_prefill_size": 32768,
        "max_prefill_tokens": 32768,
        "scheduler_recv_interval": 10,
        "stream_interval": 30,
        "fp8_gemm_backend": "flashinfer_trtllm"
      }
    }, {
      "hardware": "b200",
      "quantization": "fp8",
      "gpu_count": 8,
      "scenario": "high-throughput",
      "parameters": {
        "env_vars": "SGLANG_ENABLE_JIT_DEEPGEMM=false",
        "model_path": "deepseek-ai/DeepSeek-R1-0528",
        "tensor_parallel_size": 8,
        "cuda_graph_max_bs_decode": 128,
        "max_running_requests": 128,
        "mem_fraction_static": 0.82,
        "kv_cache_dtype": "fp8_e4m3",
        "chunked_prefill_size": 32768,
        "max_prefill_tokens": 32768,
        "scheduler_recv_interval": 30,
        "stream_interval": 30,
        "fp8_gemm_backend": "flashinfer_trtllm"
      }
    }, {
      "hardware": "b300",
      "quantization": "fp8",
      "gpu_count": 8,
      "scenario": "low-latency",
      "parameters": {
        "model_path": "deepseek-ai/DeepSeek-R1-0528",
        "tensor_parallel_size": 8,
        "kv_cache_dtype": "fp8_e4m3",
        "attention_backend": "flashinfer",
        "enforce_disable_flashinfer_allreduce_fusion": true,
        "enable_symm_mem": true
      }
    }, {
      "hardware": "b300",
      "quantization": "fp8",
      "gpu_count": 8,
      "scenario": "high-throughput",
      "parameters": {
        "model_path": "deepseek-ai/DeepSeek-R1-0528",
        "tensor_parallel_size": 8,
        "kv_cache_dtype": "fp8_e4m3",
        "attention_backend": "flashinfer",
        "enforce_disable_flashinfer_allreduce_fusion": true,
        "enable_symm_mem": true
      }
    }, {
      "hardware": "b300",
      "quantization": "fp4",
      "gpu_count": 8,
      "scenario": "low-latency",
      "parameters": {
        "model_path": "nvidia/DeepSeek-R1-0528-FP4-v2",
        "tensor_parallel_size": 8,
        "kv_cache_dtype": "fp8_e4m3",
        "attention_backend": "flashinfer",
        "enforce_disable_flashinfer_allreduce_fusion": true,
        "moe_runner_backend": "flashinfer_cutlass",
        "mem_fraction_static": 0.85,
        "enable_symm_mem": true
      }
    }, {
      "hardware": "b300",
      "quantization": "fp4",
      "gpu_count": 8,
      "scenario": "high-throughput",
      "parameters": {
        "model_path": "nvidia/DeepSeek-R1-0528-FP4-v2",
        "tensor_parallel_size": 8,
        "kv_cache_dtype": "fp8_e4m3",
        "attention_backend": "flashinfer",
        "enforce_disable_flashinfer_allreduce_fusion": true,
        "moe_runner_backend": "flashinfer_cutlass",
        "mem_fraction_static": 0.85,
        "enable_symm_mem": true
      }
    }, {
      "hardware": "h200",
      "quantization": "fp8",
      "gpu_count": 8,
      "scenario": "low-latency",
      "parameters": {
        "model_path": "deepseek-ai/DeepSeek-R1-0528",
        "trust_remote_code": true,
        "tensor_parallel_size": 8,
        "disable_radix_cache": true,
        "max_running_requests": 256,
        "cuda_graph_max_bs_decode": 256,
        "chunked_prefill_size": 32768,
        "max_prefill_tokens": 32768,
        "mem_fraction_static": 0.82,
        "attention_backend": "flashinfer",
        "stream_interval": 10,
        "decode_log_interval": 1
      }
    }, {
      "hardware": "h200",
      "quantization": "fp8",
      "gpu_count": 8,
      "scenario": "high-throughput",
      "parameters": {
        "model_path": "deepseek-ai/DeepSeek-R1-0528",
        "trust_remote_code": true,
        "tensor_parallel_size": 8,
        "disable_radix_cache": true,
        "max_running_requests": 512,
        "cuda_graph_max_bs_decode": 512,
        "chunked_prefill_size": 32768,
        "max_prefill_tokens": 32768,
        "mem_fraction_static": 0.82,
        "attention_backend": "flashinfer",
        "stream_interval": 10,
        "decode_log_interval": 1
      }
    }, {
      "hardware": "mi300x",
      "quantization": "fp8",
      "gpu_count": 8,
      "scenario": "low-latency",
      "parameters": {
        "env_vars": "SGLANG_USE_AITER=1 SGLANG_AITER_MLA_PERSIST=1",
        "model_path": "deepseek-ai/DeepSeek-R1-0528",
        "trust_remote_code": true,
        "tensor_parallel_size": 8,
        "mem_fraction_static": 0.8,
        "cuda_graph_max_bs_decode": 128,
        "chunked_prefill_size": 131072,
        "num_continuous_decode_steps": 4,
        "max_prefill_tokens": 131072,
        "kv_cache_dtype": "fp8_e4m3",
        "attention_backend": "aiter",
        "disable_radix_cache": true
      }
    }, {
      "hardware": "mi300x",
      "quantization": "fp8",
      "gpu_count": 8,
      "scenario": "high-throughput",
      "parameters": {
        "env_vars": "SGLANG_USE_AITER=1 SGLANG_AITER_MLA_PERSIST=1",
        "model_path": "deepseek-ai/DeepSeek-R1-0528",
        "trust_remote_code": true,
        "tensor_parallel_size": 8,
        "mem_fraction_static": 0.8,
        "cuda_graph_max_bs_decode": 512,
        "chunked_prefill_size": 131072,
        "num_continuous_decode_steps": 4,
        "max_prefill_tokens": 131072,
        "kv_cache_dtype": "fp8_e4m3",
        "attention_backend": "aiter",
        "disable_radix_cache": true
      }
    }, {
      "hardware": "mi325x",
      "quantization": "fp8",
      "gpu_count": 8,
      "scenario": "low-latency",
      "parameters": {
        "env_vars": "SGLANG_USE_AITER=1 SGLANG_AITER_MLA_PERSIST=1",
        "model_path": "deepseek-ai/DeepSeek-R1-0528",
        "trust_remote_code": true,
        "tensor_parallel_size": 8,
        "mem_fraction_static": 0.8,
        "cuda_graph_max_bs_decode": 128,
        "chunked_prefill_size": 131072,
        "num_continuous_decode_steps": 4,
        "max_prefill_tokens": 131072,
        "kv_cache_dtype": "fp8_e4m3",
        "attention_backend": "aiter",
        "disable_radix_cache": true
      }
    }, {
      "hardware": "mi325x",
      "quantization": "fp8",
      "gpu_count": 8,
      "scenario": "high-throughput",
      "parameters": {
        "env_vars": "SGLANG_USE_AITER=1 SGLANG_AITER_MLA_PERSIST=1",
        "model_path": "deepseek-ai/DeepSeek-R1-0528",
        "trust_remote_code": true,
        "tensor_parallel_size": 8,
        "mem_fraction_static": 0.8,
        "cuda_graph_max_bs_decode": 512,
        "chunked_prefill_size": 131072,
        "num_continuous_decode_steps": 4,
        "max_prefill_tokens": 131072,
        "kv_cache_dtype": "fp8_e4m3",
        "attention_backend": "aiter",
        "disable_radix_cache": true
      }
    }, {
      "hardware": "mi355x",
      "quantization": "fp8",
      "gpu_count": 8,
      "scenario": "low-latency",
      "parameters": {
        "env_vars": "SGLANG_USE_AITER=1 RCCL_MSCCL_ENABLE=0 ROCM_QUICK_REDUCE_QUANTIZATION=INT4",
        "model_path": "deepseek-ai/DeepSeek-R1-0528",
        "trust_remote_code": true,
        "tensor_parallel_size": 8,
        "mem_fraction_static": 0.8,
        "disable_radix_cache": true,
        "chunked_prefill_size": 196608,
        "num_continuous_decode_steps": 4,
        "max_prefill_tokens": 196608,
        "cuda_graph_max_bs_decode": 128,
        "attention_backend": "aiter",
        "kv_cache_dtype": "fp8_e4m3"
      }
    }, {
      "hardware": "mi355x",
      "quantization": "fp8",
      "gpu_count": 8,
      "scenario": "high-throughput",
      "parameters": {
        "env_vars": "SGLANG_USE_AITER=1 RCCL_MSCCL_ENABLE=0 ROCM_QUICK_REDUCE_QUANTIZATION=INT4",
        "model_path": "deepseek-ai/DeepSeek-R1-0528",
        "trust_remote_code": true,
        "tensor_parallel_size": 8,
        "mem_fraction_static": 0.8,
        "disable_radix_cache": true,
        "chunked_prefill_size": 196608,
        "num_continuous_decode_steps": 4,
        "max_prefill_tokens": 196608,
        "cuda_graph_max_bs_decode": 512,
        "attention_backend": "aiter",
        "kv_cache_dtype": "fp8_e4m3"
      }
    }, {
      "hardware": "mi355x",
      "quantization": "fp4",
      "gpu_count": 8,
      "scenario": "low-latency",
      "parameters": {
        "env_vars": "SGLANG_USE_AITER=1 ROCM_QUICK_REDUCE_QUANTIZATION=INT4",
        "model_path": "deepseek-ai/DeepSeek-R1-0528",
        "trust_remote_code": true,
        "tensor_parallel_size": 8,
        "mem_fraction_static": 0.8,
        "disable_radix_cache": true,
        "chunked_prefill_size": 196608,
        "num_continuous_decode_steps": 4,
        "max_prefill_tokens": 196608,
        "cuda_graph_max_bs_decode": 128,
        "attention_backend": "aiter",
        "kv_cache_dtype": "fp8_e4m3"
      }
    }, {
      "hardware": "mi355x",
      "quantization": "fp4",
      "gpu_count": 8,
      "scenario": "high-throughput",
      "parameters": {
        "env_vars": "SGLANG_USE_AITER=1 ROCM_QUICK_REDUCE_QUANTIZATION=INT4",
        "model_path": "deepseek-ai/DeepSeek-R1-0528",
        "trust_remote_code": true,
        "tensor_parallel_size": 8,
        "mem_fraction_static": 0.8,
        "disable_radix_cache": true,
        "chunked_prefill_size": 196608,
        "num_continuous_decode_steps": 4,
        "max_prefill_tokens": 196608,
        "cuda_graph_max_bs_decode": 512,
        "attention_backend": "aiter",
        "kv_cache_dtype": "fp8_e4m3"
      }
    }],
    "validation": [{
      "hardware": "h200",
      "quantization": "fp4",
      "error": "FP4 is only available for B200/B300 hardware. Please select FP8 quantization."
    }]
  };
  const fieldToFlag = {
    model_path: 'model-path',
    trust_remote_code: 'trust-remote-code',
    tensor_parallel_size: 'tp',
    data_parallel_size: 'dp',
    ep_size: 'ep-size',
    cuda_graph_max_bs_decode: 'cuda-graph-max-bs-decode',
    max_running_requests: 'max-running-requests',
    mem_fraction_static: 'mem-fraction-static',
    kv_cache_dtype: 'kv-cache-dtype',
    chunked_prefill_size: 'chunked-prefill-size',
    max_prefill_tokens: 'max-prefill-tokens',
    enable_flashinfer_allreduce_fusion: 'enable-flashinfer-allreduce-fusion',
    scheduler_recv_interval: 'scheduler-recv-interval',
    enable_symm_mem: 'enable-symm-mem',
    enforce_disable_flashinfer_allreduce_fusion: 'enforce-disable-flashinfer-allreduce-fusion',
    disable_radix_cache: 'disable-radix-cache',
    attention_backend: 'attention-backend',
    moe_runner_backend: 'moe-runner-backend',
    stream_interval: 'stream-interval',
    quantization: 'quantization',
    decode_log_interval: 'decode-log-interval',
    fp8_gemm_backend: 'fp8-gemm-backend',
    num_continuous_decode_steps: 'num-continuous-decode-steps'
  };
  const findConfig = (hardware, quantization, gpuCount, scenario) => {
    const match = lookupData.configs.find(entry => {
      const hardwareMatch = entry.hardware === hardware;
      const quantizationMatch = entry.quantization === quantization;
      const gpuCountMatch = !entry.gpu_count || entry.gpu_count === Number.parseInt(gpuCount, 10);
      const scenarioMatch = entry.scenario === scenario;
      return hardwareMatch && quantizationMatch && gpuCountMatch && scenarioMatch;
    });
    return match ? match.parameters : null;
  };
  const getAvailableGpuCounts = (hardware, quantization) => {
    const entries = lookupData.configs.filter(entry => entry.hardware === hardware && entry.quantization === quantization);
    const gpuCounts = [...new Set(entries.map(entry => entry.gpu_count))].filter(Boolean);
    return gpuCounts.length > 0 ? gpuCounts.sort((a, b) => a - b) : [8];
  };
  const generateCommandFromConfig = config => {
    if (!config) {
      return '# Error: Configuration not found';
    }
    let command = '';
    if (config.env_vars) {
      command = `${config.env_vars} `;
    }
    command += 'python3 -m sglang.launch_server \\\n';
    command += `  --model-path ${config.model_path}`;
    for (const [key, value] of Object.entries(config)) {
      if (key === 'model_path' || key === 'env_vars') {
        continue;
      }
      const flagName = fieldToFlag[key];
      if (!flagName) {
        continue;
      }
      if (typeof value === 'boolean') {
        if (value) {
          command += ` \\\n  --${flagName}`;
        }
        continue;
      }
      command += ` \\\n  --${flagName} ${value}`;
    }
    return command;
  };
  const validateSelection = (hardware, quantization) => {
    for (const rule of lookupData.validation || []) {
      const hardwareMatch = Array.isArray(rule.hardware) ? rule.hardware.includes(hardware) : rule.hardware === hardware;
      const quantizationMatch = Array.isArray(rule.quantization) ? rule.quantization.includes(quantization) : rule.quantization === quantization;
      if (hardwareMatch && quantizationMatch) {
        return rule.error;
      }
    }
    return null;
  };
  const resolveItems = (option, values) => typeof option.getDynamicItems === 'function' ? option.getDynamicItems(values) : option.items;
  const uiOptions = lookupData.ui_options;
  const options = {
    hardware: {
      name: 'hardware',
      title: 'Hardware Platform',
      items: uiOptions.hardware.filter(option => ['b200', 'b300', 'h200', 'mi300x', 'mi325x', 'mi355x'].includes(option.id)).map(option => ({
        id: option.id,
        label: option.label,
        default: option.id === 'b200'
      }))
    },
    quantization: {
      name: 'quantization',
      title: 'Quantization',
      getDynamicItems: values => uiOptions.quantization.map(option => {
        const fp4Disabled = ['h200', 'mi300x', 'mi325x'].includes(values.hardware) && option.id === 'fp4';
        return {
          id: option.id,
          label: option.label,
          default: ['h200', 'mi300x', 'mi325x'].includes(values.hardware) ? option.id === 'fp8' : option.default,
          disabled: fp4Disabled,
          disabledReason: fp4Disabled ? 'FP4 not supported on H200, MI300X, MI325X' : ''
        };
      })
    },
    gpuCount: {
      name: 'gpuCount',
      title: 'GPU Count',
      getDynamicItems: values => {
        const availableGpuCounts = getAvailableGpuCounts(values.hardware, values.quantization);
        const allGpuCounts = uiOptions.gpu_count.map(option => typeof option.id === 'number' ? option.id : Number.parseInt(option.id, 10));
        const defaultGpuCount = Math.max(...availableGpuCounts);
        return allGpuCounts.map(count => ({
          id: String(count),
          label: `${count} GPUs`,
          default: count === defaultGpuCount,
          disabled: !availableGpuCounts.includes(count),
          disabledReason: availableGpuCounts.includes(count) ? '' : `${count} GPUs not available for ${values.hardware.toUpperCase()} ${values.quantization.toUpperCase()}`
        }));
      }
    },
    scenario: {
      name: 'scenario',
      title: 'Scenario',
      items: uiOptions.scenario.map(option => ({
        id: option.id,
        label: option.label,
        subtitle: option.subtitle,
        default: option.default
      }))
    }
  };
  const getInitialState = () => {
    const initialState = {};
    for (const [key, option] of Object.entries(options)) {
      const items = resolveItems(option, initialState) || [];
      const fallback = items.find(item => item.default && !item.disabled) || items.find(item => !item.disabled) || items[0];
      initialState[key] = fallback ? fallback.id : '';
    }
    return initialState;
  };
  const [values, setValues] = useState(getInitialState);
  const [isDark, setIsDark] = useState(false);
  useEffect(() => {
    const checkDarkMode = () => {
      const html = document.documentElement;
      const isDarkMode = html.classList.contains('dark') || html.getAttribute('data-theme') === 'dark' || html.style.colorScheme === 'dark';
      setIsDark(isDarkMode);
    };
    checkDarkMode();
    const observer = new MutationObserver(checkDarkMode);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class', 'data-theme', 'style']
    });
    return () => observer.disconnect();
  }, []);
  const handleRadioChange = (optionName, value) => {
    setValues(prev => {
      const next = {
        ...prev,
        [optionName]: value
      };
      for (const [key, option] of Object.entries(options)) {
        if (typeof option.getDynamicItems !== 'function') {
          continue;
        }
        const items = option.getDynamicItems(next);
        const current = items.find(item => item.id === next[key]);
        if (!current || current.disabled) {
          const fallback = items.find(item => item.default && !item.disabled) || items.find(item => !item.disabled);
          if (fallback) {
            next[key] = fallback.id;
          }
        }
      }
      return next;
    });
  };
  const handleCheckboxChange = (optionName, itemId, isChecked) => {
    setValues(prev => {
      const currentValues = prev[optionName] || [];
      if (isChecked) {
        return {
          ...prev,
          [optionName]: [...currentValues, itemId]
        };
      }
      return {
        ...prev,
        [optionName]: currentValues.filter(id => id !== itemId)
      };
    });
  };
  const handleTextChange = (optionName, value) => {
    setValues(prev => ({
      ...prev,
      [optionName]: value
    }));
  };
  const generateCommand = vals => {
    const validationError = validateSelection(vals.hardware, vals.quantization);
    if (validationError) {
      return `# Error: ${validationError}`;
    }
    const config = findConfig(vals.hardware, vals.quantization, vals.gpuCount || '8', vals.scenario);
    if (!config) {
      return `# Error: No configuration found for:
# Hardware: ${vals.hardware}
# Quantization: ${vals.quantization}
# GPU Count: ${vals.gpuCount}
# Scenario: ${vals.scenario}
# This combination is not yet supported.`;
    }
    return generateCommandFromConfig(config);
  };
  const command = generateCommand(values);
  const containerStyle = {
    maxWidth: '900px',
    margin: '0 auto',
    display: 'flex',
    flexDirection: 'column',
    gap: '4px'
  };
  const cardStyle = {
    padding: '8px 12px',
    border: `1px solid ${isDark ? '#374151' : '#e5e7eb'}`,
    borderLeft: `3px solid ${isDark ? '#E85D4D' : '#D45D44'}`,
    borderRadius: '4px',
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    background: isDark ? '#1f2937' : '#fff'
  };
  const titleStyle = {
    fontSize: '13px',
    fontWeight: '600',
    minWidth: '140px',
    flexShrink: 0,
    color: isDark ? '#e5e7eb' : 'inherit'
  };
  const itemsStyle = {
    display: 'flex',
    rowGap: '2px',
    columnGap: '6px',
    flexWrap: 'wrap',
    alignItems: 'center',
    flex: 1
  };
  const labelBaseStyle = {
    padding: '4px 10px',
    border: `1px solid ${isDark ? '#9ca3af' : '#d1d5db'}`,
    borderRadius: '3px',
    cursor: 'pointer',
    display: 'inline-flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    fontWeight: '500',
    fontSize: '13px',
    transition: 'all 0.2s',
    userSelect: 'none',
    minWidth: '45px',
    textAlign: 'center',
    flex: 1,
    background: isDark ? '#374151' : '#fff',
    color: isDark ? '#e5e7eb' : 'inherit'
  };
  const checkedStyle = {
    background: '#D45D44',
    color: 'white',
    borderColor: '#D45D44'
  };
  const disabledStyle = {
    cursor: 'not-allowed',
    opacity: 0.5
  };
  const subtitleStyle = {
    display: 'block',
    fontSize: '9px',
    marginTop: '1px',
    lineHeight: '1.1',
    opacity: 0.7
  };
  const textInputStyle = {
    flex: 1,
    padding: '8px 10px',
    borderRadius: '4px',
    border: `1px solid ${isDark ? '#4b5563' : '#d1d5db'}`,
    background: isDark ? '#111827' : '#fff',
    color: isDark ? '#e5e7eb' : '#111827',
    fontSize: '13px'
  };
  const commandDisplayStyle = {
    flex: 1,
    padding: '12px 16px',
    background: isDark ? '#111827' : '#f5f5f5',
    borderRadius: '6px',
    fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace",
    fontSize: '12px',
    lineHeight: '1.5',
    color: isDark ? '#e5e7eb' : '#374151',
    whiteSpace: 'pre-wrap',
    overflowX: 'auto',
    margin: 0,
    border: `1px solid ${isDark ? '#374151' : '#e5e7eb'}`
  };
  return <div style={containerStyle} className="not-prose">
      {Object.entries(options).map(([key, option]) => {
    if (option.condition && !option.condition(values)) {
      return null;
    }
    const items = option.getDynamicItems ? option.getDynamicItems(values) : option.items || [];
    return <div key={key} style={cardStyle}>
            <div style={titleStyle}>{option.title}</div>
            <div style={itemsStyle}>
              {option.type === 'text' ? <input type="text" value={values[option.name] || ''} placeholder={option.placeholder || ''} onChange={event => handleTextChange(option.name, event.target.value)} style={textInputStyle} /> : option.type === 'checkbox' ? (option.items || []).map(item => {
      const isChecked = (values[option.name] || []).includes(item.id);
      const isDisabled = item.required || typeof item.disabledWhen === 'function' && item.disabledWhen(values);
      return <label key={item.id} title={item.disabledReason || ''} style={{
        ...labelBaseStyle,
        ...isChecked ? checkedStyle : {},
        ...isDisabled ? disabledStyle : {}
      }}>
                      <input type="checkbox" checked={isChecked} disabled={isDisabled} onChange={event => handleCheckboxChange(option.name, item.id, event.target.checked)} style={{
        display: 'none'
      }} />
                      {item.label}
                      {item.subtitle && <small style={{
        ...subtitleStyle,
        color: isChecked ? 'rgba(255,255,255,0.85)' : 'inherit'
      }}>
                          {item.subtitle}
                        </small>}
                    </label>;
    }) : items.map(item => {
      const isChecked = values[option.name] === item.id;
      const isDisabled = Boolean(item.disabled);
      return <label key={item.id} title={item.disabledReason || ''} style={{
        ...labelBaseStyle,
        ...isChecked ? checkedStyle : {},
        ...isDisabled ? disabledStyle : {}
      }}>
                      <input type="radio" name={option.name} value={item.id} checked={isChecked} disabled={isDisabled} onChange={() => !isDisabled && handleRadioChange(option.name, item.id)} style={{
        display: 'none'
      }} />
                      {item.label}
                      {item.subtitle && <small style={{
        ...subtitleStyle,
        color: isChecked ? 'rgba(255,255,255,0.85)' : 'inherit'
      }}>
                          {item.subtitle}
                        </small>}
                    </label>;
    })}
            </div>
          </div>;
  })}

      <div style={cardStyle}>
        <div style={titleStyle}>Run this Command:</div>
        <pre style={commandDisplayStyle}>{command}</pre>
      </div>
    </div>;
};

export const DeepSeekR1BasicDeployment = () => {
  const options = {
    hardware: {
      name: 'hardware',
      title: 'Hardware Platform',
      items: [{
        id: 'h100',
        label: 'H100',
        default: false
      }, {
        id: 'h200',
        label: 'H200',
        default: false
      }, {
        id: 'b200',
        label: 'B200',
        default: true
      }, {
        id: 'b300',
        label: 'B300',
        default: false
      }, {
        id: 'mi300x',
        label: 'MI300X',
        default: false
      }, {
        id: 'mi325x',
        label: 'MI325X',
        default: false
      }, {
        id: 'mi355x',
        label: 'MI355X',
        default: false
      }, {
        id: 'xeon',
        label: 'XEON',
        default: false
      }]
    },
    quantization: {
      name: 'quantization',
      title: 'Quantization',
      getDynamicItems: values => {
        const isXeon = values.hardware === 'xeon';
        const fp4Disabled = values.hardware === 'h100' || values.hardware === 'mi300x' || isXeon;
        return [{
          id: 'fp8',
          label: 'FP8',
          default: true
        }, {
          id: 'fp4',
          label: 'FP4',
          default: false,
          disabled: fp4Disabled,
          disabledReason: isXeon ? 'Intel Xeon CPUs do not support FP4 quantization' : 'H100 and MI300X only support FP8 quantization'
        }, {
          id: 'int8',
          label: 'INT8',
          default: false,
          disabled: !isXeon,
          disabledReason: 'INT8 is only available when XEON hardware is selected'
        }];
      }
    },
    strategy: {
      name: 'strategy',
      title: 'Deployment Strategy',
      type: 'checkbox',
      items: [{
        id: 'tp',
        label: 'TP',
        subtitle: 'Tensor Parallel',
        default: true,
        required: true
      }, {
        id: 'dp',
        label: 'DP',
        subtitle: 'Data Parallel',
        default: false,
        disabledWhen: v => v.hardware === 'xeon',
        disabledReason: 'Intel Xeon CPUs only support Tensor Parallel (TP)'
      }, {
        id: 'ep',
        label: 'EP',
        subtitle: 'Expert Parallel',
        default: false,
        disabledWhen: v => v.hardware === 'xeon',
        disabledReason: 'Intel Xeon CPUs only support Tensor Parallel (TP)'
      }, {
        id: 'mtp',
        label: 'MTP',
        subtitle: 'Multi-token Prediction',
        default: false,
        disabledWhen: v => v.hardware === 'xeon',
        disabledReason: 'Intel Xeon CPUs do not support Multi-token Prediction'
      }]
    },
    thinking: {
      name: 'thinking',
      title: 'Reasoning Parser',
      items: [{
        id: 'disabled',
        label: 'Disabled',
        default: true
      }, {
        id: 'enabled',
        label: 'Enabled',
        default: false
      }]
    },
    toolcall: {
      name: 'toolcall',
      title: 'Tool Call Parser',
      items: [{
        id: 'disabled',
        label: 'Disabled',
        default: true
      }, {
        id: 'enabled',
        label: 'Enabled',
        default: false
      }]
    }
  };
  const resolveItems = (option, values) => typeof option.getDynamicItems === 'function' ? option.getDynamicItems(values) : option.items;
  const getInitialState = () => {
    const initialState = {};
    for (const [key, option] of Object.entries(options)) {
      if (option.type === 'checkbox') {
        initialState[key] = (option.items || []).filter(item => item.default).map(item => item.id);
        continue;
      }
      const items = resolveItems(option, initialState) || [];
      const fallback = items.find(item => item.default && !item.disabled) || items.find(item => !item.disabled) || items[0];
      initialState[key] = fallback ? fallback.id : '';
    }
    return initialState;
  };
  const generateCommand = values => {
    const {hardware, quantization, strategy, thinking, toolcall} = values;
    const strategyValues = Array.isArray(strategy) ? strategy : [];
    if ((hardware === 'h100' || hardware === 'mi300x') && quantization === 'fp4') {
      return '# Error: H100 and MI300X only support FP8 quantization';
    }
    const isXeon = hardware === 'xeon';
    const modelPath = quantization === 'fp4' ? 'nvidia/DeepSeek-R1-0528-FP4-v2' : quantization === 'int8' ? 'Conexis/DeepSeek-R1-0528-Channel-INT8' : 'deepseek-ai/DeepSeek-R1-0528';
    let command = 'python3 -m sglang.launch_server \\\n';
    command += `  --model-path ${modelPath}`;
    if (strategyValues.includes('tp')) {
      command += isXeon ? ' \\\n  --tp 6' : ' \\\n  --tp 8';
    }
    if (strategyValues.includes('dp')) {
      command += ' \\\n  --attn-dp-size 8';
    }
    if (strategyValues.includes('ep')) {
      command += ' \\\n  --ep 8';
    }
    if (strategyValues.includes('mtp')) {
      command += ' \\\n  --speculative-algorithm EAGLE' + ' \\\n  --speculative-num-steps 3' + ' \\\n  --speculative-eagle-topk 1' + ' \\\n  --speculative-num-draft-tokens 4';
    }
    if (!isXeon) {
      command += ' \\\n  --enable-symm-mem # Optional: improves performance, but may be unstable';
    }
    if (hardware === 'b200' || hardware === 'mi355x' && quantization === 'fp8') {
      command += ' \\\n  --kv-cache-dtype fp8_e4m3 # Optional: enables fp8 kv cache and fp8 attention kernels to improve performance';
    }
    if (hardware === 'b300') {
      command += ' \\\n  --kv-cache-dtype fp8_e4m3';
      command += ' \\\n  --attention-backend flashinfer';
      command += ' \\\n  --enforce-disable-flashinfer-allreduce-fusion';
      if (quantization === 'fp4') {
        command += ' \\\n  --moe-runner-backend flashinfer_cutlass';
      }
      if (quantization === 'fp4' || strategyValues.includes('mtp')) {
        command += ' \\\n  --mem-fraction-static 0.85';
      }
    }
    if (isXeon) {
      command += ' \\\n  --device cpu \\\n  --disable-overlap-schedule';
      if (quantization === 'int8') {
        command += ' \\\n  --quantization w8a8_int8';
      }
    }
    if (thinking === 'enabled') {
      command += ' \\\n  --reasoning-parser deepseek-r1';
    }
    if (toolcall === 'enabled') {
      command += ' \\\n  --tool-call-parser deepseekv3' + ' \\\n  --chat-template examples/chat_template/tool_chat_template_deepseekr1.jinja';
    }
    return command;
  };
  const [values, setValues] = useState(getInitialState);
  const [isDark, setIsDark] = useState(false);
  useEffect(() => {
    const checkDarkMode = () => {
      const html = document.documentElement;
      const isDarkMode = html.classList.contains('dark') || html.getAttribute('data-theme') === 'dark' || html.style.colorScheme === 'dark';
      setIsDark(isDarkMode);
    };
    checkDarkMode();
    const observer = new MutationObserver(checkDarkMode);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class', 'data-theme', 'style']
    });
    return () => observer.disconnect();
  }, []);
  const handleRadioChange = (optionName, value) => {
    setValues(prev => {
      const next = {
        ...prev,
        [optionName]: value
      };
      if (optionName === 'hardware') {
        const quantizationItems = resolveItems(options.quantization, next);
        const current = quantizationItems.find(item => item.id === next.quantization);
        if (!current || current.disabled) {
          const fallback = quantizationItems.find(item => item.default && !item.disabled) || quantizationItems.find(item => !item.disabled);
          if (fallback) {
            next.quantization = fallback.id;
          }
        }
      }
      if (optionName === 'hardware') {
        if (next.hardware === 'xeon') {
          next.quantization = 'int8';
        } else if (next.quantization === 'int8') {
          next.quantization = 'fp8';
        }
      }
      const strategyItems = options.strategy.items || [];
      const currentStrategy = Array.isArray(next.strategy) ? next.strategy : [];
      next.strategy = currentStrategy.filter(id => {
        const item = strategyItems.find(s => s.id === id);
        if (!item) {
          return false;
        }
        if (typeof item.disabledWhen === 'function' && item.disabledWhen(next)) {
          return false;
        }
        return true;
      });
      return next;
    });
  };
  const handleCheckboxChange = (optionName, itemId, isChecked) => {
    setValues(prev => {
      const currentValues = prev[optionName] || [];
      if (isChecked) {
        return {
          ...prev,
          [optionName]: [...currentValues, itemId]
        };
      }
      return {
        ...prev,
        [optionName]: currentValues.filter(id => id !== itemId)
      };
    });
  };
  const handleTextChange = (optionName, value) => {
    setValues(prev => ({
      ...prev,
      [optionName]: value
    }));
  };
  const command = generateCommand(values);
  const containerStyle = {
    maxWidth: '900px',
    margin: '0 auto',
    display: 'flex',
    flexDirection: 'column',
    gap: '4px'
  };
  const cardStyle = {
    padding: '8px 12px',
    border: `1px solid ${isDark ? '#374151' : '#e5e7eb'}`,
    borderLeft: `3px solid ${isDark ? '#E85D4D' : '#D45D44'}`,
    borderRadius: '4px',
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    background: isDark ? '#1f2937' : '#fff'
  };
  const titleStyle = {
    fontSize: '13px',
    fontWeight: '600',
    minWidth: '140px',
    flexShrink: 0,
    color: isDark ? '#e5e7eb' : 'inherit'
  };
  const itemsStyle = {
    display: 'flex',
    rowGap: '2px',
    columnGap: '6px',
    flexWrap: 'wrap',
    alignItems: 'center',
    flex: 1
  };
  const labelBaseStyle = {
    padding: '4px 10px',
    border: `1px solid ${isDark ? '#9ca3af' : '#d1d5db'}`,
    borderRadius: '3px',
    cursor: 'pointer',
    display: 'inline-flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    fontWeight: '500',
    fontSize: '13px',
    transition: 'all 0.2s',
    userSelect: 'none',
    minWidth: '45px',
    textAlign: 'center',
    flex: 1,
    background: isDark ? '#374151' : '#fff',
    color: isDark ? '#e5e7eb' : 'inherit'
  };
  const checkedStyle = {
    background: '#D45D44',
    color: 'white',
    borderColor: '#D45D44'
  };
  const disabledStyle = {
    cursor: 'not-allowed',
    opacity: 0.5
  };
  const subtitleStyle = {
    display: 'block',
    fontSize: '9px',
    marginTop: '1px',
    lineHeight: '1.1',
    opacity: 0.7
  };
  const textInputStyle = {
    flex: 1,
    padding: '8px 10px',
    borderRadius: '4px',
    border: `1px solid ${isDark ? '#4b5563' : '#d1d5db'}`,
    background: isDark ? '#111827' : '#fff',
    color: isDark ? '#e5e7eb' : '#111827',
    fontSize: '13px'
  };
  const commandDisplayStyle = {
    flex: 1,
    padding: '12px 16px',
    background: isDark ? '#111827' : '#f5f5f5',
    borderRadius: '6px',
    fontFamily: "'Menlo', 'Monaco', 'Courier New', monospace",
    fontSize: '12px',
    lineHeight: '1.5',
    color: isDark ? '#e5e7eb' : '#374151',
    whiteSpace: 'pre-wrap',
    overflowX: 'auto',
    margin: 0,
    border: `1px solid ${isDark ? '#374151' : '#e5e7eb'}`
  };
  return <div style={containerStyle} className="not-prose">
      {Object.entries(options).map(([key, option]) => {
    if (option.condition && !option.condition(values)) {
      return null;
    }
    const items = option.getDynamicItems ? option.getDynamicItems(values) : option.items || [];
    return <div key={key} style={cardStyle}>
            <div style={titleStyle}>{option.title}</div>
            <div style={itemsStyle}>
              {option.type === 'text' ? <input type="text" value={values[option.name] || ''} placeholder={option.placeholder || ''} onChange={event => handleTextChange(option.name, event.target.value)} style={textInputStyle} /> : option.type === 'checkbox' ? (option.items || []).map(item => {
      const isChecked = (values[option.name] || []).includes(item.id);
      const isDisabled = item.required || typeof item.disabledWhen === 'function' && item.disabledWhen(values);
      return <label key={item.id} title={item.disabledReason || ''} style={{
        ...labelBaseStyle,
        ...isChecked ? checkedStyle : {},
        ...isDisabled ? disabledStyle : {}
      }}>
                      <input type="checkbox" checked={isChecked} disabled={isDisabled} onChange={event => handleCheckboxChange(option.name, item.id, event.target.checked)} style={{
        display: 'none'
      }} />
                      {item.label}
                      {item.subtitle && <small style={{
        ...subtitleStyle,
        color: isChecked ? 'rgba(255,255,255,0.85)' : 'inherit'
      }}>
                          {item.subtitle}
                        </small>}
                    </label>;
    }) : items.map(item => {
      const isChecked = values[option.name] === item.id;
      const isDisabled = Boolean(item.disabled);
      return <label key={item.id} title={item.disabledReason || ''} style={{
        ...labelBaseStyle,
        ...isChecked ? checkedStyle : {},
        ...isDisabled ? disabledStyle : {}
      }}>
                      <input type="radio" name={option.name} value={item.id} checked={isChecked} disabled={isDisabled} onChange={() => !isDisabled && handleRadioChange(option.name, item.id)} style={{
        display: 'none'
      }} />
                      {item.label}
                      {item.subtitle && <small style={{
        ...subtitleStyle,
        color: isChecked ? 'rgba(255,255,255,0.85)' : 'inherit'
      }}>
                          {item.subtitle}
                        </small>}
                    </label>;
    })}
            </div>
          </div>;
  })}

      <div style={cardStyle}>
        <div style={titleStyle}>Run this Command:</div>
        <pre style={commandDisplayStyle}>{command}</pre>
      </div>
    </div>;
};

## 1. Model Introduction

[DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) is DeepSeek's advanced reasoning model that combines powerful language understanding with step-by-step reasoning capabilities. The model is available in multiple quantization formats optimized for different hardware platforms.

**Key Features:**

* **Advanced Reasoning**: Built-in reasoning capabilities for complex problem-solving
* **Multiple Quantizations**: FP8 and FP4 variants for different performance/memory trade-offs
* **Hardware Optimization**: Specifically tuned for NVIDIA B200 (Blackwell) and H200 (Hopper) GPUs, AMD MI300X, MI325X and MI355X GPUs, as well as Intel Xeon CPUs
* **High Performance**: Optimized for both throughput and latency scenarios

**Available Models:**

* **FP8 (8-bit quantized)**: [deepseek-ai/DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) - Recommended for H200 and MI300X
* **FP4 (4-bit quantized)**: [nvidia/DeepSeek-R1-0528-FP4-v2](https://huggingface.co/nvidia/DeepSeek-R1-0528-FP4-v2) - Recommended for B200 and MI355X
* **BF16 (upcast from FP8)**: [unsloth/DeepSeek-R1-0528-BF16](https://huggingface.co/unsloth/DeepSeek-R1-0528-BF16)
* **INT8 (channel-wise)**: [meituan/DeepSeek-R1-Channel-INT8](https://huggingface.co/meituan/DeepSeek-R1-Channel-INT8)
* **W4A8**: [novita/Deepseek-R1-0528-W4AFP8](https://huggingface.co/novita/Deepseek-R1-0528-W4AFP8)
* **AWQ (4-bit)**: [QuixiAI/DeepSeek-R1-0528-AWQ](https://huggingface.co/QuixiAI/DeepSeek-R1-0528-AWQ)
* **MXFP4**: [amd/DeepSeek-R1-MXFP4](https://huggingface.co/amd/DeepSeek-R1-MXFP4)

**License:**
To use DeepSeek-R1, you must agree to DeepSeek's Community License. See [LICENSE](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528/blob/main/LICENSE) for details.

For more details, please refer to the [official DeepSeek-R1 repository](https://github.com/deepseek-ai/DeepSeek-R1).

## 2. SGLang Installation

Please refer to the [official SGLang installation guide](../../../docs/get-started/install) for installation instructions.

For SGLang CPU installation, please refer to the [CPU version installation guide](../../../docs/hardware-platforms/cpu_server#installation).

## 3. Model Deployment

This section provides deployment configurations optimized for different hardware platforms and use cases.

### 3.1 Basic Configuration

**Interactive Command Generator**: Use the configuration selector below to automatically generate a basic deployment command for your hardware platform, quantization method, and deployment strategy.

<DeepSeekR1BasicDeployment />

### 3.2 Optimal Configurations

Pareto-optimal configurations for B200, H200, MI300X, MI325X, and MI355X hardware.

<DeepSeekR1AdvancedDeployment />

### 3.3 Configuration Tips

DeepSeek-R1 shares the same MoE architecture as DeepSeek-V3, so the same hardware and optimization recommendations apply.

**Recommended GPU configurations by weight type:**

<table style={{width: "100%", borderCollapse: "collapse"}}>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, backgroundColor: "rgba(255,255,255,0.02)"}}>Weight Type</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, backgroundColor: "rgba(255,255,255,0.05)"}}>Supported Hardware</th>
    </tr>
  </thead>

  <tbody>
    <tr>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}><strong>FP8</strong> (recommended)</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>8× H200, 8× B200, 8× MI300X, 2×8× H100/H800/H20</td>
    </tr>

    <tr>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}><strong>BF16</strong> (upcast from FP8)</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>2×8× H200, 2×8× MI300X, 4×8× H100/H800, 4×8× A100/A800</td>
    </tr>

    <tr>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}><strong>INT8</strong></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>16× A100/A800, 32× L40S, Xeon 6980P CPU, 4× Ascend A3 Series</td>
    </tr>

    <tr>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}><strong>W4A8 / AWQ / MXFP4 / NVFP4</strong></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>8× H20/H100, 4× H200; 8× H100/A100; 8/4× MI355X/MI350X; 8/4× B200</td>
    </tr>
  </tbody>
</table>

> The official DeepSeek-R1 checkpoint is already in FP8 format — do **not** add `--quantization fp8` when serving it.

**DeepGEMM precompilation (NVIDIA Hopper / Blackwell):** Precompile GEMM kernels to avoid JIT overhead (\~10 min):

```bash theme={null}
python3 -m sglang.compile_deep_gemm --model deepseek-ai/DeepSeek-R1 --tp 8 --trust-remote-code
```

**Data Parallelism Attention (`--attn-dp-size`):** Recommended for high-throughput scenarios. Use `--tp 8 --attn-dp-size 8` on a single 8-GPU node.

**NCCL timeout:** If model loading is slow, increase: `--dist-timeout 3600`.

**Xeon CPU service configuration:** Please refer to the `Notes` part in the serving engine launching section in [the SGLang CPU server document](../../../docs/hardware-platforms/cpu_server#launch-of-the-serving-engine) to better understand how to configure the arguments, especially for TP (tensor parallel) and NUMA binding settings.

## 4. Model Invocation

### 4.1 Basic Usage

For basic API usage and request examples, please refer to:

* [SGLang Basic Usage Guide](../../../docs/basic_usage/send_request)

### 4.2 Advanced Usage

#### 4.2.1 Reasoning Parser

DeepSeek-R1 supports advanced reasoning capabilities with built-in thinking process. Enable the reasoning parser during deployment to separate the thinking and content sections:

```shell Command theme={null}
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-R1-0528 \
  --reasoning-parser deepseek-r1 \
  --tp 8
```

**Streaming with Thinking Process:**

```python Example theme={null}
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="EMPTY"
)

# Enable streaming to see the thinking process in real-time
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-0528",
    messages=[
        {"role": "user", "content": "Solve this problem step by step: What is 15% of 240?"}
    ],
    temperature=0.7,
    max_tokens=2048,
    stream=True
)

# Process the stream
has_thinking = False
has_answer = False
thinking_started = False

for chunk in response:
    if chunk.choices and len(chunk.choices) > 0:
        delta = chunk.choices[0].delta

        # Print thinking process
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
            if not thinking_started:
                print("=============== Thinking =================", flush=True)
                thinking_started = True
            has_thinking = True
            print(delta.reasoning_content, end="", flush=True)

        # Print answer content
        if delta.content:
            # Close thinking section and add content header
            if has_thinking and not has_answer:
                print("\n=============== Content =================", flush=True)
                has_answer = True
            print(delta.content, end="", flush=True)

print()
```

**Output Example:**

```text Output theme={null}
=============== Thinking =================
To solve this problem, I need to calculate 15% of 240.
Step 1: Convert 15% to decimal: 15% = 0.15
Step 2: Multiply 240 by 0.15
Step 3: 240 × 0.15 = 36
=============== Content =================

The answer is 36. To find 15% of 240, we multiply 240 by 0.15, which equals 36.
```

**Note:** The reasoning parser captures the model's step-by-step thinking process, allowing you to see how the model arrives at its conclusions.

#### 4.2.2 Tool Calling

DeepSeek-R1 supports tool calling capabilities. Enable the tool call parser:

```shell Command theme={null}
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-R1-0528 \
  --reasoning-parser deepseek-r1 \
  --tool-call-parser deepseekv3 \
  --chat-template examples/chat_template/tool_chat_template_deepseekr1.jinja \
  --tp 8
```

**Python Example (with Thinking Process):**

```python Example theme={null}
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="EMPTY"
)

# Define available tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# Make request with streaming to see thinking process
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-0528",
    messages=[
        {"role": "user", "content": "What's the weather in Beijing?"}
    ],
    tools=tools,
    temperature=0.7,
    stream=True
)

# Process streaming response
thinking_started = False
has_thinking = False

for chunk in response:
    if chunk.choices and len(chunk.choices) > 0:
        delta = chunk.choices[0].delta

        # Print thinking process
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
            if not thinking_started:
                print("=============== Thinking =================", flush=True)
                thinking_started = True
            has_thinking = True
            print(delta.reasoning_content, end="", flush=True)

        # Print tool calls
        if hasattr(delta, 'tool_calls') and delta.tool_calls:
            # Close thinking section if needed
            if has_thinking and thinking_started:
                print("\n=============== Content =================", flush=True)
                thinking_started = False

            for tool_call in delta.tool_calls:
                if tool_call.function:
                    print(f"🔧 Tool Call: {tool_call.function.name}")
                    print(f"   Arguments: {tool_call.function.arguments}")

        # Print content
        if delta.content:
            print(delta.content, end="", flush=True)

print()
```

**Output Example:**

```text Output theme={null}
=============== Thinking =================
The user is asking about the weather in Beijing. I need to use the get_weather function to retrieve this information.
I should call the function with location="Beijing".
=============== Content =================

🔧 Tool Call: get_weather
   Arguments:
🔧 Tool Call: None
   Arguments: {"location": "Beijing"}
```

**Note:**

* The reasoning parser shows how the model decides to use a tool
* Tool calls are clearly marked with the function name and arguments
* You can then execute the function and send the result back to continue the conversation

**Handling Tool Call Results:**

```python Example theme={null}
# After getting the tool call, execute the function
def get_weather(location, unit="celsius"):
    # Your actual weather API call here
    return f"The weather in {location} is 22°{unit[0].upper()} and sunny."

# Send tool result back to the model
messages = [
    {"role": "user", "content": "What's the weather in Beijing?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "Beijing", "unit": "celsius"}'
            }
        }]
    },
    {
        "role": "tool",
        "tool_call_id": "call_123",
        "content": get_weather("Beijing", "celsius")
    }
]

final_response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-0528",
    messages=messages,
    temperature=0.7
)

print(final_response.choices[0].message.content)
# Output: "The weather in Beijing is currently 22°C and sunny."
```

#### 4.2.3 Multi-Token Prediction (EAGLE Speculative Decoding)

DeepSeek-R1 supports EAGLE-based Multi-Token Prediction (MTP), the same mechanism as DeepSeek-V3. Refer to [DeepSeek-V3 §4.2.3](/cookbook/autoregressive/DeepSeek/DeepSeek-V3#4-2-3-multi-token-prediction-eagle-speculative-decoding) for the complete launch command, flag reference, tuning guidance (`--speculative-num-steps`, `--speculative-eagle-topk`, `--max-running-requests`), and `bench_speculative.py` link. R1's speed benchmark commands that include `--speculative-*` flags use this mechanism.

#### 4.2.4 Thinking Budget

Limit the model's thinking token budget using `CustomLogitProcessor`. Launch with `--enable-custom-logit-processor`:

```shell Command theme={null}
python3 -m sglang.launch_server \
  --model deepseek-ai/DeepSeek-R1 \
  --tp 8 \
  --port 30000 \
  --reasoning-parser deepseek-r1 \
  --enable-custom-logit-processor
```

```python Example theme={null}
import openai
from sglang.srt.sampling.custom_logit_processor import DeepSeekR1ThinkingBudgetLogitProcessor

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="*")
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1",
    messages=[{"role": "user", "content": "Is Paris the Capital of France?"}],
    max_tokens=1024,
    extra_body={
        "custom_logit_processor": DeepSeekR1ThinkingBudgetLogitProcessor().to_str(),
        "custom_params": {"thinking_budget": 512},
    },
)
print(response)
```

## 5. Benchmark

This section uses **industry-standard configurations** for comparable benchmark results.

### 5.1 Speed Benchmark

**Test Environment:**

* Hardware: B200 GPU (8x)
* Model: DeepSeek-R1-0528
* Tensor Parallelism: 8
* SGLang Version: 0.5.6.post1

**Benchmark Methodology:**

We use industry-standard benchmark configurations to ensure results are comparable across frameworks and hardware platforms.

#### 5.1.1 Standard Test Scenarios

Three core scenarios reflect real-world usage patterns:

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "25%"}} />

    <col style={{width: "25%"}} />

    <col style={{width: "25%"}} />

    <col style={{width: "25%"}} />
  </colgroup>

  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.02)"}}>Scenario</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.05)"}}>Input Length</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.02)"}}>Output Length</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.05)"}}>Use Case</th>
    </tr>
  </thead>

  <tbody>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>**Chat**</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>1K</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>1K</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Most common conversational AI workload</td>
    </tr>

    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>**Reasoning**</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>1K</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>8K</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Long-form generation, complex reasoning tasks</td>
    </tr>

    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>**Summarization**</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>8K</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>1K</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Document summarization, RAG retrieval</td>
    </tr>
  </tbody>
</table>

#### 5.1.2 Concurrency Levels

Test each scenario at different concurrency levels to capture the throughput vs. latency trade-off:

* **Low Concurrency**: `--max-concurrency 1` (Latency-optimized)
* **Medium Concurrency**: `--max-concurrency 16` (Balanced)
* **High Concurrency**: `--max-concurrency 100` (Throughput-optimized)

#### 5.1.3 Number of Prompts

For each concurrency level, configure `num_prompts` to simulate realistic user loads:

* **Quick Test**: `num_prompts = concurrency × 1` (minimal test)
* **Recommended**: `num_prompts = concurrency × 5` (standard benchmark)
* **Stable Measurements**: `num_prompts = concurrency × 10` (production-grade)

***

#### 5.1.4 Benchmark Commands

**Scenario 1: Chat (1K/1K) - Most Important**

* **Model Deployment**

```bash Command theme={null}
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-R1-0528 \
  --tp 8
```

* Low Concurrency (Latency-Optimized)

```bash Command theme={null}
python -m sglang.bench_serving \
  --backend sglang \
  --model deepseek-ai/DeepSeek-R1-0528 \
  --dataset-name random \
  --random-input-len 1000 \
  --random-output-len 1000 \
  --num-prompts 10 \
  --max-concurrency 1 \
  --request-rate inf
```

```text Output theme={null}
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 1
Successful requests:                     10
Benchmark duration (s):                  40.00
Total input tokens:                      6101
Total input text tokens:                 6101
Total input vision tokens:               0
Total generated tokens:                  4210
Total generated tokens (retokenized):    4205
Request throughput (req/s):              0.25
Input token throughput (tok/s):          152.52
Output token throughput (tok/s):         105.24
Peak output token throughput (tok/s):    110.00
Peak concurrent requests:                2
Total token throughput (tok/s):          257.76
Concurrency:                             1.00
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   3998.40
Median E2E Latency (ms):                 3207.53
---------------Time to First Token----------------
Mean TTFT (ms):                          153.00
Median TTFT (ms):                        140.76
P99 TTFT (ms):                           214.66
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          9.16
Median TPOT (ms):                        9.15
P99 TPOT (ms):                           9.21
---------------Inter-Token Latency----------------
Mean ITL (ms):                           9.16
Median ITL (ms):                         9.15
P95 ITL (ms):                            9.47
P99 ITL (ms):                            9.63
Max ITL (ms):                            15.45
==================================================
```

* Medium Concurrency (Balanced)

```bash Command theme={null}
python -m sglang.bench_serving \
  --backend sglang \
  --model deepseek-ai/DeepSeek-R1-0528 \
  --dataset-name random \
  --random-input-len 1000 \
  --random-output-len 1000 \
  --num-prompts 80 \
  --max-concurrency 16 \
  --request-rate inf
```

```text Output theme={null}
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 16
Successful requests:                     80
Benchmark duration (s):                  51.21
Total input tokens:                      39668
Total input text tokens:                 39668
Total input vision tokens:               0
Total generated tokens:                  40725
Total generated tokens (retokenized):    40458
Request throughput (req/s):              1.56
Input token throughput (tok/s):          774.66
Output token throughput (tok/s):         795.30
Peak output token throughput (tok/s):    1088.00
Peak concurrent requests:                21
Total token throughput (tok/s):          1569.96
Concurrency:                             13.93
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   8918.33
Median E2E Latency (ms):                 9466.16
---------------Time to First Token----------------
Mean TTFT (ms):                          273.51
Median TTFT (ms):                        131.71
P99 TTFT (ms):                           839.57
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          17.56
Median TPOT (ms):                        17.46
P99 TPOT (ms):                           28.68
---------------Inter-Token Latency----------------
Mean ITL (ms):                           17.02
Median ITL (ms):                         14.70
P95 ITL (ms):                            16.41
P99 ITL (ms):                            112.38
Max ITL (ms):                            461.90
==================================================
```

* High Concurrency (Throughput-Optimized)

```bash Command theme={null}
python -m sglang.bench_serving \
  --backend sglang \
  --model deepseek-ai/DeepSeek-R1-0528 \
  --dataset-name random \
  --random-input-len 1000 \
  --random-output-len 1000 \
  --num-prompts 500 \
  --max-concurrency 100 \
  --request-rate inf
```

```text Output theme={null}
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 100
Successful requests:                     500
Benchmark duration (s):                  110.46
Total input tokens:                      249831
Total input text tokens:                 249831
Total input vision tokens:               0
Total generated tokens:                  252162
Total generated tokens (retokenized):    251441
Request throughput (req/s):              4.53
Input token throughput (tok/s):          2261.80
Output token throughput (tok/s):         2282.90
Peak output token throughput (tok/s):    3900.00
Peak concurrent requests:                109
Total token throughput (tok/s):          4544.71
Concurrency:                             92.26
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   20380.71
Median E2E Latency (ms):                 19391.65
---------------Time to First Token----------------
Mean TTFT (ms):                          563.14
Median TTFT (ms):                        147.62
P99 TTFT (ms):                           2632.11
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          40.11
Median TPOT (ms):                        41.98
P99 TPOT (ms):                           50.10
---------------Inter-Token Latency----------------
Mean ITL (ms):                           39.37
Median ITL (ms):                         26.36
P95 ITL (ms):                            98.16
P99 ITL (ms):                            150.08
Max ITL (ms):                            2052.85
==================================================
```

**Scenario 2: Reasoning (1K/8K)**

* Low Concurrency

```bash Command theme={null}
python -m sglang.bench_serving \
  --backend sglang \
  --model deepseek-ai/DeepSeek-R1-0528 \
  --dataset-name random \
  --random-input-len 1000 \
  --random-output-len 8000 \
  --num-prompts 10 \
  --max-concurrency 1 \
  --request-rate inf
```

```text Output theme={null}
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 1
Successful requests:                     10
Benchmark duration (s):                  411.34
Total input tokens:                      6101
Total input text tokens:                 6101
Total input vision tokens:               0
Total generated tokens:                  44452
Total generated tokens (retokenized):    44390
Request throughput (req/s):              0.02
Input token throughput (tok/s):          14.83
Output token throughput (tok/s):         108.07
Peak output token throughput (tok/s):    110.00
Peak concurrent requests:                2
Total token throughput (tok/s):          122.90
Concurrency:                             1.00
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   41132.04
Median E2E Latency (ms):                 44288.71
---------------Time to First Token----------------
Mean TTFT (ms):                          125.76
Median TTFT (ms):                        126.19
P99 TTFT (ms):                           137.69
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          9.21
Median TPOT (ms):                        9.20
P99 TPOT (ms):                           9.27
---------------Inter-Token Latency----------------
Mean ITL (ms):                           9.23
Median ITL (ms):                         9.22
P95 ITL (ms):                            9.64
P99 ITL (ms):                            9.86
Max ITL (ms):                            15.18
==================================================
```

* Medium Concurrency

```bash Command theme={null}
python -m sglang.bench_serving \
  --backend sglang \
  --model deepseek-ai/DeepSeek-R1-0528 \
  --dataset-name random \
  --random-input-len 1000 \
  --random-output-len 8000 \
  --num-prompts 80 \
  --max-concurrency 16 \
  --request-rate inf
```

```text Output theme={null}
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 16
Successful requests:                     80
Benchmark duration (s):                  348.93
Total input tokens:                      39668
Total input text tokens:                 39668
Total input vision tokens:               0
Total generated tokens:                  318226
Total generated tokens (retokenized):    317630
Request throughput (req/s):              0.23
Input token throughput (tok/s):          113.69
Output token throughput (tok/s):         912.02
Peak output token throughput (tok/s):    1088.00
Peak concurrent requests:                19
Total token throughput (tok/s):          1025.70
Concurrency:                             14.07
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   61360.70
Median E2E Latency (ms):                 62071.20
---------------Time to First Token----------------
Mean TTFT (ms):                          176.02
Median TTFT (ms):                        153.75
P99 TTFT (ms):                           268.44
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          15.42
Median TPOT (ms):                        15.59
P99 TPOT (ms):                           16.07
---------------Inter-Token Latency----------------
Mean ITL (ms):                           15.39
Median ITL (ms):                         15.17
P95 ITL (ms):                            16.62
P99 ITL (ms):                            18.13
Max ITL (ms):                            226.59
==================================================
```

* High Concurrency

```bash Command theme={null}
python -m sglang.bench_serving \
  --backend sglang \
  --model deepseek-ai/DeepSeek-R1-0528 \
  --dataset-name random \
  --random-input-len 1000 \
  --random-output-len 8000 \
  --num-prompts 320 \
  --max-concurrency 64 \
  --request-rate inf
```

```text Output theme={null}
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 64
Successful requests:                     320
Benchmark duration (s):                  589.31
Total input tokens:                      158939
Total input text tokens:                 158939
Total input vision tokens:               0
Total generated tokens:                  1300705
Total generated tokens (retokenized):    1297658
Request throughput (req/s):              0.54
Input token throughput (tok/s):          269.70
Output token throughput (tok/s):         2207.16
Peak output token throughput (tok/s):    2944.00
Peak concurrent requests:                68
Total token throughput (tok/s):          2476.86
Concurrency:                             57.03
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   105032.36
Median E2E Latency (ms):                 108229.09
---------------Time to First Token----------------
Mean TTFT (ms):                          223.91
Median TTFT (ms):                        158.15
P99 TTFT (ms):                           474.86
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          25.94
Median TPOT (ms):                        26.72
P99 TPOT (ms):                           27.99
---------------Inter-Token Latency----------------
Mean ITL (ms):                           25.79
Median ITL (ms):                         25.37
P95 ITL (ms):                            26.70
P99 ITL (ms):                            105.49
Max ITL (ms):                            237.91
==================================================
```

**Scenario 3: Summarization (8K/1K)**

* Low Concurrency

```bash Command theme={null}
python -m sglang.bench_serving \
  --backend sglang \
  --model deepseek-ai/DeepSeek-R1-0528 \
  --dataset-name random \
  --random-input-len 8000 \
  --random-output-len 1000 \
  --num-prompts 10 \
  --max-concurrency 1 \
  --request-rate inf
```

```text Output theme={null}
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 1
Successful requests:                     10
Benchmark duration (s):                  40.65
Total input tokens:                      41941
Total input text tokens:                 41941
Total input vision tokens:               0
Total generated tokens:                  4210
Total generated tokens (retokenized):    4195
Request throughput (req/s):              0.25
Input token throughput (tok/s):          1031.65
Output token throughput (tok/s):         103.56
Peak output token throughput (tok/s):    110.00
Peak concurrent requests:                2
Total token throughput (tok/s):          1135.20
Concurrency:                             1.00
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   4063.62
Median E2E Latency (ms):                 3296.13
---------------Time to First Token----------------
Mean TTFT (ms):                          165.91
Median TTFT (ms):                        154.96
P99 TTFT (ms):                           240.92
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          9.26
Median TPOT (ms):                        9.27
P99 TPOT (ms):                           9.42
---------------Inter-Token Latency----------------
Mean ITL (ms):                           9.28
Median ITL (ms):                         9.28
P95 ITL (ms):                            9.66
P99 ITL (ms):                            9.83
Max ITL (ms):                            14.06
==================================================
```

* Medium Concurrency

```bash Command theme={null}
python -m sglang.bench_serving \
  --backend sglang \
  --model deepseek-ai/DeepSeek-R1-0528 \
  --dataset-name random \
  --random-input-len 8000 \
  --random-output-len 1000 \
  --num-prompts 80 \
  --max-concurrency 16 \
  --request-rate inf
```

```text Output theme={null}
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 16
Successful requests:                     80
Benchmark duration (s):                  56.71
Total input tokens:                      300020
Total input text tokens:                 300020
Total input vision tokens:               0
Total generated tokens:                  41589
Total generated tokens (retokenized):    41490
Request throughput (req/s):              1.41
Input token throughput (tok/s):          5290.75
Output token throughput (tok/s):         733.41
Peak output token throughput (tok/s):    1024.00
Peak concurrent requests:                20
Total token throughput (tok/s):          6024.16
Concurrency:                             14.25
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   10098.99
Median E2E Latency (ms):                 10623.46
---------------Time to First Token----------------
Mean TTFT (ms):                          486.80
Median TTFT (ms):                        189.59
P99 TTFT (ms):                           2138.73
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          19.06
Median TPOT (ms):                        19.23
P99 TPOT (ms):                           30.69
---------------Inter-Token Latency----------------
Mean ITL (ms):                           18.53
Median ITL (ms):                         15.63
P95 ITL (ms):                            16.64
P99 ITL (ms):                            109.71
Max ITL (ms):                            1471.36
==================================================
```

* High Concurrency

```bash Command theme={null}
python -m sglang.bench_serving \
  --backend sglang \
  --model deepseek-ai/DeepSeek-R1-0528 \
  --dataset-name random \
  --random-input-len 8000 \
  --random-output-len 1000 \
  --num-prompts 320 \
  --max-concurrency 64 \
  --request-rate inf
```

```text Output theme={null}
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 64
Successful requests:                     320
Benchmark duration (s):                  115.55
Total input tokens:                      1273893
Total input text tokens:                 1273893
Total input vision tokens:               0
Total generated tokens:                  169680
Total generated tokens (retokenized):    169275
Request throughput (req/s):              2.77
Input token throughput (tok/s):          11024.93
Output token throughput (tok/s):         1468.50
Peak output token throughput (tok/s):    2254.00
Peak concurrent requests:                70
Total token throughput (tok/s):          12493.43
Concurrency:                             59.45
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   21465.98
Median E2E Latency (ms):                 20686.26
---------------Time to First Token----------------
Mean TTFT (ms):                          913.93
Median TTFT (ms):                        224.92
P99 TTFT (ms):                           6257.83
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          39.93
Median TPOT (ms):                        40.99
P99 TPOT (ms):                           60.91
---------------Inter-Token Latency----------------
Mean ITL (ms):                           38.83
Median ITL (ms):                         26.29
P95 ITL (ms):                            113.81
P99 ITL (ms):                            176.94
Max ITL (ms):                            5521.53
==================================================
```

#### 5.1.5 Understanding the Results

**Key Metrics:**

* **Request Throughput (req/s)**: Number of requests processed per second
* **Output Token Throughput (tok/s)**: Total tokens generated per second
* **Mean TTFT (ms)**: Time to First Token - measures responsiveness
* **Mean TPOT (ms)**: Time Per Output Token - measures generation speed
* **Mean ITL (ms)**: Inter-Token Latency - measures streaming consistency

**Why These Configurations Matter:**

* **1K/1K (Chat)**: Represents the most common conversational AI workload. This is the highest priority scenario for most deployments.
* **1K/8K (Reasoning)**: Tests long-form generation capabilities crucial for complex reasoning, code generation, and detailed explanations.
* **8K/1K (Summarization)**: Evaluates performance with large context inputs, essential for RAG systems, document Q\&A, and summarization tasks.
* **Variable Concurrency**: Captures the Pareto frontier - the optimal trade-off between throughput and latency at different load levels. Low concurrency shows best-case latency, high concurrency shows maximum throughput.

**Interpreting Results:**

* Compare your results against baseline numbers for your hardware
* Higher throughput at same latency = better performance
* Lower TTFT = more responsive user experience
* Lower TPOT = faster generation speed

### 5.2 Accuracy Benchmark

Document model accuracy on standard benchmarks:

#### 5.2.1 GSM8K Benchmark

* Benchmark Command

```bash Command theme={null}
sgl-eval run gsm8k \
  --base-url http://127.0.0.1:30000/v1 \
  --num-examples 1319 \
  --num-threads 1316
```

Historical results below use the removed harness and are not measurements of the sgl-eval command above.

**Test Results:**

```text Output theme={null}
Accuracy: 0.959
Invalid: 0.000
Latency: 29.185 s
Output throughput: 4854.672 token/s
```


This documentation is built and hosted on [Mintlify](https://mintlify.com), a developer documentation platform.