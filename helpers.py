# lep/helpers.p
from transformers import TrainerCallback
from dataclasses import dataclass
import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
from transformers import TrainerCallback, AutoTokenizer
import os
from tokenizer.src.tokenizer_arabic.normalizer import ArabicNormalizerConfig, build_qwen_like_normalizer
from tokenizer.src.tokenizer_arabic.pretokenizer import ArabicPreTokenizerConfig, build_qwen_like_pretokenizer


class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_evaluate = True


@dataclass
class EvalConfig:
    model_path: str
    tokenizer_path: str
    local_ds_path: str
    eval_column: str


class TorchProfilerCallback(TrainerCallback):
    """
    Profiles a small window of training steps and records CUDA memory + op breakdown.
    Open results in TensorBoard.
    """
    def __init__(self, logdir="tb_prof", wait=0, warmup=1, active=4, repeat=1):
        self.logdir = logdir
        os.makedirs(logdir, exist_ok=True)
        self.prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
            on_trace_ready=tensorboard_trace_handler(logdir),
            record_shapes=True,
            profile_memory=True,   # <= key
            with_stack=True,
        )

    def on_train_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            self.prof.__enter__()

    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            self.prof.step()

    def on_train_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            self.prof.__exit__(None, None, None)
            print(f"[prof] traces saved to: {self.logdir}", flush=True)


class MemProbeCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0:
            torch.cuda.synchronize()
            print("train peak GiB:", torch.cuda.max_memory_allocated()/1024**3, flush=True)

    def on_evaluate(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        print("after eval peak GiB:", torch.cuda.max_memory_allocated()/1024**3, flush=True)
        # optional: defragment pressure relief
        gc.collect()
        torch.cuda.empty_cache()


# u2b = {u: b for b, u in bytes_to_unicode().items()}

# def decode_qwen_visual_token(s: str) -> str:
#     # Map each displayed char back to its original byte, then decode as UTF-8 text
#     b = bytes(u2b[ch] for ch in s)
#     return b.decode("utf-8", errors="replace")

# def decode_qwen_arabic_array(tokens: list[str]) -> str:
#     return [decode_qwen_visual_token(t) for t in tokens]


def load_custom_tokenizer(tokenizer_path, normalizer_config = None, pretokenizer_config = None):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if normalizer_config is None:
        normalizer_config = ArabicNormalizerConfig(
            unicode_normalization="NFKC",
            remove_diacritics=True,
            lowercase_latin=False,
            preserve_alif_variants=True,
        )

    if pretokenizer_config is None:
        pretokenizer_config = ArabicPreTokenizerConfig(
            with_prefix_space=False,
            add_prefix_space=False,
            fuse_numeric_sequences=False
        )

    tokenizer.backend_tokenizer.normalizer = build_qwen_like_normalizer(normalizer_config)
    tokenizer.backend_tokenizer.pre_tokenizer = build_qwen_like_pretokenizer(pretokenizer_config)

    return tokenizer