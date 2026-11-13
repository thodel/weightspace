"""TrocrSubject on a tiny random VisionEncoderDecoderModel built from a config (no download, no hub)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")
PIL = pytest.importorskip("PIL.Image")

from wsp.trocr_subject import TrocrSubject, canonical_name, component  # noqa: E402


def _tokenizer():
    from tokenizers import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast

    t = ByteLevelBPETokenizer()
    t.train_from_iterator(["zu trennen mich nicht entschliessen konnte . Runen und runenartig ."] * 20, vocab_size=300,
                          special_tokens=["<s>", "<pad>", "</s>", "<unk>"])
    return PreTrainedTokenizerFast(tokenizer_object=t._tokenizer, bos_token="<s>", pad_token="<pad>", eos_token="</s>",
                                   unk_token="<unk>")


@pytest.fixture(scope="module")
def parts():
    from transformers import (TrOCRConfig, ViTConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel,
                              ViTImageProcessor)
    tok = _tokenizer()
    torch.manual_seed(0)
    enc = ViTConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64, image_size=32,
                    patch_size=16, qkv_bias=False)
    dec = TrOCRConfig(vocab_size=len(tok), d_model=32, decoder_layers=2, decoder_attention_heads=2, decoder_ffn_dim=64,
                      max_position_embeddings=64, cross_attention_hidden_size=32, use_learned_position_embeddings=True,
                      layernorm_embedding=True, tie_word_embeddings=False,
                      bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                      decoder_start_token_id=tok.eos_token_id)
    cfg = VisionEncoderDecoderConfig.from_encoder_decoder_configs(enc, dec)
    cfg.decoder_start_token_id = tok.eos_token_id
    cfg.pad_token_id = tok.pad_token_id
    cfg.eos_token_id = tok.eos_token_id
    model = VisionEncoderDecoderModel(cfg)
    model.generation_config.eos_token_id = tok.eos_token_id
    model.generation_config.pad_token_id = tok.pad_token_id
    model.generation_config.decoder_start_token_id = tok.eos_token_id
    proc = ViTImageProcessor(size={"height": 32, "width": 32}, image_mean=[0.5] * 3, image_std=[0.5] * 3)
    return model, tok, proc


@pytest.fixture
def subject(parts):
    import copy
    model, tok, proc = parts
    return TrocrSubject(model=copy.deepcopy(model), tokenizer=tok, image_processor=proc, batch=2,
                        generation={"max_new_tokens": 12, "num_beams": 2})


def _items(n=3, text="zu trennen mich"):
    rng = np.random.default_rng(1)
    from PIL import Image
    return [{"key": f"k{i}", "text": text,
             "image": Image.fromarray(rng.integers(0, 255, (20, 90 + 7 * i, 3), dtype=np.uint8))} for i in range(n)]


def test_canonical_names_are_a_bijection_and_cover_everything(subject):
    names = list(subject.tensors())
    assert len(names) == len(set(names)) == len(subject.model.state_dict()) - 2    # without the pooler
    assert not [n for n in names if n.startswith("other.")]
    assert not [n for n in names if n.startswith("enc.pooler")] and subject.unused == ["enc.pooler.bias", "enc.pooler.weight"]
    for must in ("enc.00.attn.q.weight", "enc.01.ffn.out.bias", "enc.embed.patch.weight", "enc.norm_final.weight",
                 "dec.00.self.o.weight", "dec.01.cross.k.weight", "dec.00.ffn.in.weight", "dec.01.norm_cross.bias",
                 "dec.embed.tokens", "dec.embed.pos", "dec.embed.norm.weight", "dec.out.weight"):
        assert must in names, must
    assert canonical_name("encoder.encoder.layer.11.attention.attention.query.weight") == "enc.11.attn.q.weight"
    assert canonical_name("something.new") == "other.something.new"
    assert component("dec.03.cross.v.weight") == "dec.cross" and component("enc.embed.cls") == "enc.embed"
    assert all(t.dtype == np.float32 for t in subject.tensors().values())


def test_describe_counts_parameters_per_component(subject):
    d = subject.describe()
    total = sum(p.numel() for p in subject.model.parameters())
    assert d["parameters"] == total == sum(d["by_component"].values())
    assert d["encoder_parameters"] + d["decoder_parameters"] + d["bridge_and_other"] + d["unused_by_forward"] == total
    assert d["unused_by_forward"] == d["by_component"]["enc.pooler"] == 32 * 32 + 32
    assert d["by_component"]["dec.out"] == subject.tensors()["dec.out.weight"].size
    assert d["aliases"] == {} and d["encoder_layers"] == 2 and d["decoder_layers"] == 2


def test_set_tensor_changes_the_model_and_refuses_shape_change(subject):
    t = subject.tensors()["dec.out.weight"].copy()
    subject.set_tensor("dec.out.weight", t * 2)
    assert np.allclose(subject.model.state_dict()["decoder.output_projection.weight"].numpy(), 2 * t)
    assert np.allclose(subject.tensors()["dec.out.weight"], 2 * t)
    with pytest.raises(ValueError):
        subject.set_tensor("dec.out.weight", t[:-1])
    with pytest.raises(KeyError):
        subject.set_tensor("dec.nope", t)


def test_digest_moves_with_an_edit_and_returns(subject):
    d0 = subject.digest()
    t = subject.tensors()["enc.00.attn.q.weight"].copy()
    subject.set_tensor("enc.00.attn.q.weight", t + 1e-3)
    assert subject.digest() != d0
    subject.set_tensor("enc.00.attn.q.weight", t)
    assert subject.digest() == d0


def test_forward_is_deterministic_and_reports_extras(subject):
    items = _items()
    a = subject.forward(items)
    b = subject.forward(items)
    assert [p.key for p in a] == ["k0", "k1", "k2"]
    assert [p.text for p in a] == [p.text for p in b]
    assert [p.extra["score"] for p in a] == [p.extra["score"] for p in b]
    assert all(set(p.extra) == {"tokens", "at_cap", "score"} for p in a)
    assert all(p.extra["tokens"] <= 12 for p in a)


def test_an_edit_changes_what_forward_reads(subject):
    items = _items(2)
    before = subject.forward(items)
    subject.set_tensor("dec.out.weight", -subject.tensors()["dec.out.weight"])
    after = subject.forward(items)
    assert [p.text for p in before] != [p.text for p in after] or \
        [p.extra["score"] for p in before] != [p.extra["score"] for p in after]


def test_teacher_forced_loss_and_margin(subject):
    items = _items(3)
    r = subject.teacher_forced(items)
    toks = subject._label_ids(items[0]["text"])
    assert r[0]["tokens"] == len(toks) and len(r[0]["nll"]) == len(r[0]["margin"]) == len(toks)
    assert all(x >= 0 for x in r[0]["nll"])
    assert np.isclose(r[0]["loss"], np.mean(r[0]["nll"]))
    # batching does not change a line's numbers (equal text, pad-free here); one line alone gives the same loss
    one = subject.teacher_forced(items[:1])[0]
    assert np.allclose(one["nll"], r[0]["nll"], atol=1e-4)
    # nll is the log-softmax of logits() at the reference token
    lg = subject.logits(items[0])
    lp = lg - np.log(np.exp(lg - lg.max(1, keepdims=True)).sum(1, keepdims=True)) - lg.max(1, keepdims=True)
    assert np.allclose(-lp[np.arange(len(toks)), toks], r[0]["nll"], atol=1e-4)
    # margin is positive exactly where the reference token is the arg-max
    assert [m > 0 for m in r[0]["margin"]] == [int(lg[i].argmax()) == toks[i] for i in range(len(toks))]
    # raising the reference token's output row lowers the loss (the readout moves the right way)
    row = toks[1]
    w = subject.tensors()["dec.out.weight"].copy()
    w[row] *= 3.0
    subject.set_tensor("dec.out.weight", w)
    assert subject.teacher_forced(items[:1])[0]["nll"][1] != one["nll"][1]


def test_save_roundtrip_hub_format(subject, tmp_path):
    from safetensors.numpy import load_file
    subject.set_tensor("dec.00.norm_ffn.weight", subject.tensors()["dec.00.norm_ffn.weight"] * 1.5)
    out = subject.save(tmp_path / "out")
    for f in ("config.json", "model.safetensors", "tokenizer.json"):
        assert (out / f).exists(), f
    saved = load_file(str(out / "model.safetensors"))
    assert {canonical_name(k) for k in saved} == set(subject.tensors()) | set(subject.unused)
    back = TrocrSubject(out, batch=2, generation={"max_new_tokens": 12, "num_beams": 2})
    assert back.digest() == subject.digest()
    items = _items(2)
    assert [p.text for p in back.forward(items)] == [p.text for p in subject.forward(items)]
    # saving twice over the same folder works
    subject.save(out)


def test_tied_embedding_is_one_tensor_with_an_alias(parts):
    """The hub files of the Bern family hold one embedding tensor and tie the output projection to it."""
    from transformers import TrOCRConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel
    model, tok, proc = parts
    dec = TrOCRConfig(**{**model.config.decoder.to_dict(), "tie_word_embeddings": True})
    cfg = VisionEncoderDecoderConfig.from_encoder_decoder_configs(model.config.encoder, dec)
    cfg.decoder_start_token_id = tok.eos_token_id; cfg.pad_token_id = tok.pad_token_id
    m = VisionEncoderDecoderModel(cfg)
    m.generation_config.eos_token_id = tok.eos_token_id; m.generation_config.pad_token_id = tok.pad_token_id
    m.generation_config.decoder_start_token_id = tok.eos_token_id
    s = TrocrSubject(model=m, tokenizer=tok, image_processor=proc, batch=2, generation={"max_new_tokens": 8, "num_beams": 1})
    assert "dec.out.weight" not in s.tensors() and "dec.embed.tokens" in s.tensors()
    assert s.describe()["aliases"] == {"dec.out.weight": "dec.embed.tokens"}
    n0 = s.describe()["by_component"]["dec.embed"]
    w = s.tensors()["dec.embed.tokens"].copy()
    s.set_tensor("dec.out.weight", w * 2)                       # the alias writes the one stored tensor
    assert np.allclose(s.tensors()["dec.embed.tokens"], 2 * w)
    assert np.allclose(m.state_dict()["decoder.output_projection.weight"].numpy(), 2 * w)
    assert s.describe()["by_component"]["dec.embed"] == n0      # counted once
    assert s.describe()["parameters"] == sum(p.numel() for p in m.parameters())


def test_a_three_pixel_high_crop_is_read_the_same_alone_and_in_a_batch(subject):
    """(3, W, 3) is ambiguous for transformers; the format must not depend on which image comes first."""
    from PIL import Image
    rng = np.random.default_rng(2)
    thin = {"key": "thin", "image": Image.fromarray(rng.integers(0, 255, (3, 90, 3), dtype=np.uint8))}
    other = {"key": "o", "image": Image.fromarray(rng.integers(0, 255, (20, 90, 3), dtype=np.uint8))}
    alone = subject._pixels([thin])[0]
    first = subject._pixels([thin, other])[0]
    second = subject._pixels([other, thin])[1]
    assert torch.equal(alone, first) and torch.equal(alone, second)
