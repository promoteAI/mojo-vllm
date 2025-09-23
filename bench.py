import os
import time
from random import randint, seed

def bench_vllm():
    from vllm import LLM, SamplingParams
    print("===== vLLM Benchmark =====")
    seed(0)
    num_seqs = 8
    max_input_len = 1024
    max_ouput_len = 1024

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    # 强制使用float16，避免bfloat16导致的兼容性问题
    # 注意：Qwen3ForCausalLM 目前不被 vLLM 支持，避免报错
    try:
        llm = LLM(path,enforce_eager=True, max_model_len=4096,trust_remote_code=True, tensor_parallel_size=1,gpu_memory_utilization=0.7)
    except Exception as e:
        print("vLLM 初始化失败，可能模型架构不被支持。")
        raise

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # vllm 需要如下格式
    prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    try:
        llm.generate(["Benchmark: "], SamplingParams())
        t = time.time()
        llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
        t = (time.time() - t)
        total_tokens = sum(sp.max_tokens for sp in sampling_params)
        throughput = total_tokens / t
        print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s\n")
    except Exception as e:
        # 明确提示架构不支持
        if "not supported" in str(e):
            print("vLLM 测试失败: 当前模型架构不被支持。请更换为支持的架构，例如 LlamaForCausalLM、QWenLMHeadModel 等。")
        else:
            print(f"vLLM 测试失败: {e}")
        raise

def bench_mojovllm():
    from mojovllm import LLM, SamplingParams
    import torch.distributed as dist
    print("===== Mojo-vLLM Benchmark =====")
    seed(0)
    num_seqs = 8
    max_input_len = 1024
    max_ouput_len = 1024

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    # 尝试销毁已存在的进程组，避免重复初始化报错
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception as e:
            print(f"销毁已有进程组时出错: {e}")

    try:
        llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
    except RuntimeError as e:
        if "initialize the default process group twice" in str(e):
            print("Mojo-vLLM 测试失败: 检测到进程组已初始化，尝试销毁后重试。")
            try:
                dist.destroy_process_group()
                llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
            except Exception as e2:
                print(f"Mojo-vLLM 进程组销毁后重试仍失败: {e2}")
                raise
        else:
            print(f"Mojo-vLLM 初始化失败: {e}")
            raise

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]

    try:
        llm.generate(["Benchmark: "], SamplingParams())
        t = time.time()
        llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
        t = (time.time() - t)
        total_tokens = sum(sp.max_tokens for sp in sampling_params)
        throughput = total_tokens / t
        print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s\n")
    except Exception as e:
        if "initialize the default process group twice" in str(e):
            print("Mojo-vLLM 测试失败: 检测到进程组已初始化。请确保每次运行前销毁进程组，或重启 Python 进程。")
        else:
            print(f"Mojo-vLLM 测试失败: {e}")
        raise

if __name__ == "__main__":
    # 两种都跑一遍
    try:
        bench_vllm()
    except Exception as e:
        print(f"vLLM 测试失败: {e}")
    try:
        bench_mojovllm()
    except Exception as e:
        print(f"Mojo-vLLM 测试失败: {e}")
