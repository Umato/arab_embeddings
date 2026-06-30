from __future__ import annotations

import copy
from types import SimpleNamespace

import torch

from token_distillation.staged import (
    build_stagewise_distill_stages,
    collect_stage_token_snippets_from_texts,
    tokenize_with_teacher_tokenizer_to_final_ids,
)
from token_distillation.tokdist import compute_safe_resize_token_embeddings_size, extend_pretrained_with_tokens_and_embeddings
from token_distillation.train_loop import _mask_embedding_gradient, train_embeddings


class FakeTokenizer:
    def __init__(self, vocab, *, pad_token_id=None, eos_token_id=None, text_to_ids=None):
        self._vocab = dict(vocab)
        self._text_to_ids = copy.deepcopy(text_to_ids or {})
        self.all_special_tokens = []
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.eos_token = "<eos>" if eos_token_id is not None else None
        self.unk_token_id = None

    def get_vocab(self):
        return dict(self._vocab)

    def add_tokens(self, tokens):
        for token in tokens:
            if token not in self._vocab:
                self._vocab[token] = len(self._vocab)

    def convert_tokens_to_ids(self, token):
        return self._vocab[token]

    def convert_ids_to_tokens(self, token_ids):
        inv_vocab = {token_id: token for token, token_id in self._vocab.items()}
        if isinstance(token_ids, int):
            return inv_vocab[token_ids]
        return [inv_vocab[token_id] for token_id in token_ids]

    def __len__(self):
        return len(self._vocab)

    def __deepcopy__(self, memo):
        return FakeTokenizer(
            copy.deepcopy(self._vocab, memo),
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            text_to_ids=copy.deepcopy(self._text_to_ids, memo),
        )

    def __call__(self, text, add_special_tokens=False):
        if text not in self._text_to_ids:
            raise KeyError(f"Missing fake tokenization for {text!r}")
        return {"input_ids": list(self._text_to_ids[text])}


class TinyTiedModel(torch.nn.Module):
    def __init__(self, vocab_size=6, hidden_size=4):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=True)
        self.embeddings = torch.nn.Embedding(vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embeddings.weight

    @property
    def device(self):
        return self.embeddings.weight.device

    def get_input_embeddings(self):
        return self.embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def resize_token_embeddings(self, new_size):
        old_weight = self.embeddings.weight.detach().clone()
        hidden_size = old_weight.shape[1]
        new_embeddings = torch.nn.Embedding(new_size, hidden_size)
        new_embeddings.weight.data.zero_()
        rows_to_copy = min(new_size, old_weight.shape[0])
        new_embeddings.weight.data[:rows_to_copy] = old_weight[:rows_to_copy]
        self.embeddings = new_embeddings
        self.lm_head = torch.nn.Linear(hidden_size, new_size, bias=False)
        self.lm_head.weight = self.embeddings.weight
        return self.embeddings

    def tie_weights(self):
        self.lm_head.weight = self.embeddings.weight

    def forward(self, input_ids, output_hidden_states=False):
        hidden = self.embeddings(input_ids)
        logits = hidden @ self.embeddings.weight.transpose(0, 1)
        hidden_states = [hidden, hidden]
        return {"logits": logits, "hidden_states": hidden_states}


def test_safe_resize_keeps_existing_embedding_rows_when_tokenizer_is_shorter():
    model = TinyTiedModel(vocab_size=8, hidden_size=3)
    tokenizer = FakeTokenizer({"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}, pad_token_id=4, eos_token_id=4)

    assert compute_safe_resize_token_embeddings_size(model, len(tokenizer.get_vocab()) + 1) == 8

    model, updated_tokenizer = extend_pretrained_with_tokens_and_embeddings(
        out_path="unused",
        model=model,
        new_tokens_to_input_embs={"fresh": torch.ones(3)},
        source_tokenizer=tokenizer,
        save=False,
    )

    assert model.get_input_embeddings().weight.shape[0] == 8
    assert updated_tokenizer.convert_tokens_to_ids("fresh") == 5
    assert torch.equal(model.get_input_embeddings().weight.data[5], torch.ones(3))


def test_mask_embedding_gradient_keeps_only_explicit_trainable_ids():
    gradient = torch.ones(6, 2)
    _mask_embedding_gradient(gradient, [4], num_rows=6)

    assert torch.equal(gradient[4], torch.ones(2))
    assert torch.equal(gradient[:4], torch.zeros(4, 2))
    assert torch.equal(gradient[5:], torch.zeros(1, 2))


def test_build_stagewise_distill_stages_bootstraps_teacher_availability():
    base_tokenizer = FakeTokenizer({"a": 0, "b": 1, "c": 2, "d": 3})

    def tokenize_token(token, teacher_tokenizer, _model_path):
        vocab = teacher_tokenizer.get_vocab()
        if token in vocab:
            return [vocab[token]]
        if token == "abc":
            pieces = ["ab", "c"] if "ab" in vocab else ["a", "b", "c"]
        elif token == "abcd":
            if "abc" not in vocab:
                return [9999]
            pieces = ["abc", "d"]
        else:
            pieces = [token]
        return [vocab[piece] for piece in pieces]

    stages = build_stagewise_distill_stages(
        base_tokenizer=base_tokenizer,
        ordered_added_tokens=["ab", "abc", "abcd"],
        support_tokens=["ab"],
        distill_tokens=["abc", "abcd"],
        model_path="unused",
        tokenize_token_fn=tokenize_token,
    )

    assert [stage["eligible_tokens"] for stage in stages] == [["abc"], ["abcd"]]


def test_teacher_tokenization_is_remapped_to_final_ids_not_teacher_local_ids():
    final_tokenizer = FakeTokenizer({"base0": 0, "base1": 1, "A": 2, "B": 3, "C": 4})
    teacher_tokenizer = FakeTokenizer({"base0": 0, "base1": 1, "B": 2, "C": 3})

    remapped_ids = tokenize_with_teacher_tokenizer_to_final_ids(
        "BC",
        teacher_tokenizer=teacher_tokenizer,
        final_tokenizer=final_tokenizer,
        model_path="unused",
        tokenize_text_fn=lambda _text, _teacher_tokenizer, _model_path: [2, 3],
    )

    assert remapped_ids == [3, 4]


def test_stage_corpus_snippets_only_contain_final_tokenizer_ids():
    final_tokenizer = FakeTokenizer(
        {"pad": 0, "A": 1, "B": 7, "C": 8, "tail": 9},
        text_to_ids={
            "BC": [7, 8],
            "doc-one": [1, 7, 8, 9],
            "doc-two": [7, 8],
        },
    )
    teacher_tokenizer = FakeTokenizer(
        {"pad": 0, "B": 2, "C": 3},
        text_to_ids={"BC": [2, 3]},
    )

    collected = collect_stage_token_snippets_from_texts(
        texts=["doc-one", "doc-two"],
        todo_tokens=[("BC", 99)],
        teacher_tokenizer=teacher_tokenizer,
        final_tokenizer=final_tokenizer,
        model_path="unused",
        context_length=3,
        snippets_per_token=1,
    )

    assert collected.docs_scanned == 1
    assert collected.token_to_pattern_ids["BC"] == [7, 8]
    assert collected.skipped == []
    assert collected.token_to_snippets["BC"] == [[1, 7, 8]]
    assert 2 not in collected.token_to_pattern_ids["BC"]
    assert all(token_id in final_tokenizer.get_vocab().values() for token_id in collected.token_to_snippets["BC"][0])


def test_train_embeddings_respects_explicit_trainable_ids_with_tied_embeddings():
    model = TinyTiedModel(vocab_size=6, hidden_size=4)
    tokenizer = FakeTokenizer(
        {"tok0": 0, "tok1": 1, "tok2": 2, "tok3": 3, "new": 4, "<pad>": 5},
        pad_token_id=5,
        eos_token_id=5,
    )
    original_weight = model.get_input_embeddings().weight.detach().clone()

    trained_model = train_embeddings(
        model,
        tokenized_texts=[[[0, 1, 2, 3]]],
        new_phrase_to_new_id={(1, 2): 4},
        assigned_new_phrases=[(1, 2)],
        tokenizer=tokenizer,
        epochs=1,
        batch_size=1,
        learning_rate=0.5,
        loss_methods=["MSE-on-hiddens"],
        preserve_original_embeddings=True,
        original_token_ids=[0, 1, 2, 3, 5],
        trainable_token_ids=[4],
        logit_token_ids=[0, 1, 2, 3],
        target_layers=[-1],
        mixed_precision=False,
        learn_output_with_ce=False,
    )

    updated_weight = trained_model.get_input_embeddings().weight.detach()
    assert torch.equal(updated_weight[[0, 1, 2, 3, 5]], original_weight[[0, 1, 2, 3, 5]])
    assert not torch.equal(updated_weight[4], original_weight[4])
    assert trained_model.get_input_embeddings().weight.data_ptr() == trained_model.get_output_embeddings().weight.data_ptr()
