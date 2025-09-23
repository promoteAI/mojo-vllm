import os
from mojovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256,ignore_eos=True)
    prompts = [
        "鞠婧祎是谁？"
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    # 非流式
    # outputs = llm.generate(prompts, sampling_params)

    # for prompt, output in zip(prompts, outputs):
    #     print("\n")
    #     print(f"Prompt: {prompt!r}")
    #     print(f"Completion: {output['text']!r}")
    
    # 流式使用
    for ev in llm.generate(prompts, sampling_params, stream=True,use_tqdm=False):
        # ev = {
        #   "index": seq_id,
        #   "delta_text": 本次新增文本,
        #   "text": 当前完整文本,
        #   "token_id": int,
        #   "finished": bool
        # }
        print(ev["delta_text"], end="", flush=True)


if __name__ == "__main__":
    main()
