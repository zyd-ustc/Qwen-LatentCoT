import torch

from qwen_latent_cot.utils.text_ops import (
    add_latent_pad_after_auxiliary_img,
    generate_labels_after_multi_token_start,
    replace_latent_placeholder_with_img_pad,
)


def test_replace_latent_placeholder_with_img_pad():
    text = (
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n"
        "<|im_start|>assistant\n<abs_vis_token></abs_vis_token><|im_end|>"
    )
    out = replace_latent_placeholder_with_img_pad(text)
    assert "<abs_vis_token></abs_vis_token>" not in out
    assert out.count("<|vision_start|><|image_pad|><|vision_end|>") == 2


def test_add_latent_pad_after_auxiliary_img():
    text = "<|im_start|>assistant\n<|vision_start|><|image_pad|><|vision_end|>"
    out = add_latent_pad_after_auxiliary_img([text], latent_size=3)[0]
    assert "<abs_vis_token>" in out
    assert out.count("<abs_vis_token_pad>") == 3


def test_generate_labels_after_multi_token_start():
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    start = torch.tensor([3, 4])
    labels = generate_labels_after_multi_token_start(input_ids, start, ignore_ids=[6])
    assert labels.tolist()[0][:4] == [-100, -100, -100, -100]
    assert labels.tolist()[0][4] == 5
    assert labels.tolist()[0][5] == -100
