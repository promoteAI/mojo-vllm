import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from mojovllm.config import Config
from mojovllm.sampling_params import SamplingParams
from mojovllm.engine.sequence import Sequence
from mojovllm.engine.scheduler import Scheduler
from mojovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self, stream: bool = False):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        finished_flags = self.scheduler.postprocess(seqs, token_ids)
        if stream:
            # In stream mode, return per-step events for all sequences in this batch
            events = []
            for seq, token_id, finished in zip(seqs, token_ids, finished_flags):
                events.append({
                    "seq_id": seq.seq_id,
                    "token_id": token_id,
                    "is_finished": finished,
                    "is_prefill": is_prefill,
                })
            num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
            return events, num_tokens
        else:
            outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
            num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
            return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
        stream: bool = False,
    ) -> list[str]:
        if stream:
            def _generator():
                if use_tqdm:
                    pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
                nonlocal sampling_params
                if not isinstance(sampling_params, list):
                    sampling_params = [sampling_params] * len(prompts)
                for prompt, sp in zip(prompts, sampling_params):
                    self.add_request(prompt, sp)
                prefill_throughput = decode_throughput = 0.
                finished = set()
                buffers: dict[int, list[int]] = {}
                while len(finished) < len(prompts):
                    t = perf_counter()
                    events, num_tokens = self.step(stream=True)
                    if use_tqdm:
                        if num_tokens > 0:
                            prefill_throughput = num_tokens / (perf_counter() - t)
                        else:
                            decode_throughput = -num_tokens / (perf_counter() - t)
                        pbar.set_postfix({
                            "Prefill": f"{int(prefill_throughput)}tok/s",
                            "Decode": f"{int(decode_throughput)}tok/s",
                        })
                    for e in events:
                        seq_id = e["seq_id"]
                        token_id = e["token_id"]
                        is_finished = e["is_finished"]
                        buf = buffers.setdefault(seq_id, [])
                        buf.append(token_id)
                        delta_text = self.tokenizer.decode([token_id])
                        text = self.tokenizer.decode(buf)
                        yield {
                            "index": seq_id,
                            "delta_text": delta_text,
                            "text": text,
                            "token_id": token_id,
                            "finished": is_finished,
                        }
                        if is_finished and seq_id not in finished:
                            finished.add(seq_id)
                            if use_tqdm:
                                pbar.update(1)
                if use_tqdm:
                    pbar.close()
            return _generator()
        else:
            if use_tqdm:
                pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
            if not isinstance(sampling_params, list):
                sampling_params = [sampling_params] * len(prompts)
            for prompt, sp in zip(prompts, sampling_params):
                self.add_request(prompt, sp)
            outputs = {}
            prefill_throughput = decode_throughput = 0.
            while not self.is_finished():
                t = perf_counter()
                output, num_tokens = self.step()
                if use_tqdm:
                    if num_tokens > 0:
                        prefill_throughput = num_tokens / (perf_counter() - t)
                    else:
                        decode_throughput = -num_tokens / (perf_counter() - t)
                    pbar.set_postfix({
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    })
                for seq_id, token_ids in output:
                    outputs[seq_id] = token_ids
                    if use_tqdm:
                        pbar.update(1)
            outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
            outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
            if use_tqdm:
                pbar.close()
            return outputs
