"""4D attention-mask builders for latent CoT stages."""

from __future__ import annotations

from typing import Any

import torch


def _between(ids: torch.Tensor, start_pos: int, end_pos: int, wanted_id: int | None = None) -> torch.Tensor:
    s = start_pos + 1
    e = end_pos
    if s >= e:
        return torch.empty(0, dtype=torch.long, device=ids.device)
    if wanted_id is None:
        return torch.arange(s, e, device=ids.device, dtype=torch.long)
    mask = ids[s:e] == wanted_id
    return torch.nonzero(mask, as_tuple=False).squeeze(-1) + s


def find_segments_1d(ids: torch.Tensor, token_ids: dict[str, torch.Tensor]) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]]]:
    device = ids.device
    l = ids.numel()

    v_starts = torch.nonzero(ids == token_ids["v_start"], as_tuple=False).squeeze(-1)
    v_ends = torch.nonzero(ids == token_ids["v_end"], as_tuple=False).squeeze(-1)

    vs: list[int] = []
    ve: list[int] = []
    i = 0
    j = 0
    while i < v_starts.numel() and j < v_ends.numel():
        if v_starts[i] < v_ends[j]:
            vs.append(int(v_starts[i].item()))
            ve.append(int(v_ends[j].item()))
            i += 1
            j += 1
        else:
            j += 1

    if not vs:
        return torch.empty(0, dtype=torch.long, device=device), []

    q_img_idx = _between(ids, vs[0], ve[0], wanted_id=int(token_ids["img_pad"]))
    vs = vs[1:]
    ve = ve[1:]

    a_starts = torch.nonzero(ids == token_ids["abs_start"], as_tuple=False).squeeze(-1)
    a_ends = torch.nonzero(ids == token_ids["abs_end"], as_tuple=False).squeeze(-1)
    ass: list[int] = []
    aee: list[int] = []
    i = 0
    j = 0
    while i < a_starts.numel() and j < a_ends.numel():
        if a_starts[i] < a_ends[j]:
            ass.append(int(a_starts[i].item()))
            aee.append(int(a_ends[j].item()))
            i += 1
            j += 1
        else:
            j += 1

    obs_starts_all = torch.nonzero(ids == token_ids["obs_start"], as_tuple=False).squeeze(-1)
    obs_ends_all = torch.nonzero(ids == token_ids["obs_end"], as_tuple=False).squeeze(-1)

    segs: list[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]] = []
    n_steps = min(len(vs), len(ass))
    for s in range(n_steps):
        i_idx = _between(ids, vs[s], ve[s], wanted_id=int(token_ids["img_pad"]))
        a_idx = _between(ids, ass[s], aee[s], wanted_id=int(token_ids["abs_pad"]))

        t_end = vs[s + 1] if (s + 1) < len(vs) else l
        in_start = (obs_starts_all >= aee[s]) & (obs_starts_all < t_end)
        in_end = (obs_ends_all > aee[s]) & (obs_ends_all <= t_end)
        o_starts = obs_starts_all[in_start]
        o_ends = obs_ends_all[in_end]

        o_blocks: list[torch.Tensor] = []
        p = 0
        q = 0
        while p < o_starts.numel() and q < o_ends.numel():
            start_pos = int(o_starts[p].item())
            end_pos = int(o_ends[q].item())
            if start_pos < end_pos:
                block = _between(ids, start_pos, end_pos, wanted_id=None)
                if block.numel() > 0:
                    o_blocks.append(block)
                p += 1
                q += 1
            else:
                q += 1

        segs.append((i_idx, a_idx, o_blocks))

    return q_img_idx, segs


def build_4d_attn(
    input_ids: torch.Tensor,
    pad_mask: torch.Tensor,
    token_ids: dict[str, torch.Tensor],
    not_mask_image: bool = False,
    mask_latent: bool = False,
    observation_tokens_cannot_see_question_image: bool = False,
    observation_tokens_only_see_question_and_latent: bool = False,
    latent_can_see_all_previous: bool = True,
    mask_question_image: bool = False,
) -> tuple[torch.Tensor, list[Any]]:
    input_ids = input_ids.cpu()
    pad_mask = pad_mask.cpu()

    bsz, seq_len = input_ids.shape
    device = input_ids.device

    causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
    valid = pad_mask.bool()
    allowed = causal.unsqueeze(0).repeat(bsz, 1, 1)
    for b in range(bsz):
        allowed[b] &= valid[b].unsqueeze(0)
        allowed[b] &= valid[b].unsqueeze(1)

    batch_segs: list[Any] = []
    for b in range(bsz):
        q_img_idx, segs = find_segments_1d(input_ids[b], token_ids)
        batch_segs.append(segs)

        if mask_question_image and q_img_idx.numel() > 0:
            allowed[b][:, q_img_idx] = False

        for i_idx, a_idx, o_blocks in segs:
            if a_idx.numel() > 0:
                if not latent_can_see_all_previous:
                    allowed[b][a_idx, :] = False
                    if i_idx.numel() > 0:
                        allowed[b][a_idx.unsqueeze(1), i_idx] = True

                n = a_idx.numel()
                ar = torch.arange(n, device=a_idx.device)
                tri = ar.unsqueeze(1) >= ar.unsqueeze(0)
                rows = a_idx.unsqueeze(1).expand(n, n)
                cols = a_idx.unsqueeze(0).expand(n, n)
                allowed[b][rows, cols] = tri

                if i_idx.numel() > 0 and not not_mask_image:
                    not_a = torch.ones(seq_len, dtype=torch.bool, device=device)
                    not_a[a_idx] = False
                    not_a_idx = torch.nonzero(not_a, as_tuple=False).squeeze(-1)
                    if not_a_idx.numel() > 0:
                        allowed[b][not_a_idx[:, None], i_idx] = False

                if mask_latent:
                    r_idx = torch.arange(seq_len, device=device)
                    rows_to_block = (r_idx.unsqueeze(0) > a_idx.unsqueeze(1)).any(dim=0)
                    if rows_to_block.any():
                        allowed[b][
                            rows_to_block.nonzero(as_tuple=False).squeeze(-1)[:, None], a_idx
                        ] = False

            if not o_blocks:
                continue

            ids = input_ids[b]
            q_range = None
            if observation_tokens_cannot_see_question_image:
                starts = torch.nonzero(ids == token_ids["v_start"], as_tuple=False).squeeze(-1)
                ends = torch.nonzero(ids == token_ids["v_end"], as_tuple=False).squeeze(-1)
                if starts.numel() > 0 and ends.numel() > 0:
                    q_range = torch.arange(
                        int(starts[0].item()), int(ends[0].item()) + 1, device=device
                    )

            ans_start_pos = -1
            if "ans_start" in token_ids:
                pat = token_ids["ans_start"]
                k = int(pat.numel())
                for s in range(max(0, ids.numel() - k + 1)):
                    if torch.equal(ids[s : s + k], pat):
                        ans_start_pos = s
                        break

            for o_idx in o_blocks:
                if o_idx.numel() == 0:
                    continue

                if observation_tokens_only_see_question_and_latent:
                    allowed[b][o_idx, :] = False
                    ar = torch.arange(ids.size(0), device=device)
                    before_ans = ar < ans_start_pos if ans_start_pos != -1 else torch.zeros_like(ar, dtype=torch.bool)
                    non_image = (ids != token_ids["img_pad"]) & (ids != token_ids["v_start"]) & (ids != token_ids["v_end"])
                    question_idx = torch.nonzero(before_ans & non_image, as_tuple=False).squeeze(-1)
                    latent_before = torch.nonzero((ids == token_ids["abs_pad"]) & (ar < int(o_idx[0].item())), as_tuple=False).squeeze(-1)
                    if question_idx.numel() > 0:
                        allowed[b][o_idx.unsqueeze(1), question_idx] = True
                    if latent_before.numel() > 0:
                        allowed[b][o_idx.unsqueeze(1), latent_before] = True

                if q_range is not None and q_range.numel() > 0:
                    allowed[b][o_idx.unsqueeze(1), q_range] = False

                n = o_idx.numel()
                ar = torch.arange(n, device=o_idx.device)
                tri = ar.unsqueeze(1) >= ar.unsqueeze(0)
                rows = o_idx.unsqueeze(1).expand(n, n)
                cols = o_idx.unsqueeze(0).expand(n, n)
                allowed[b][rows, cols] = tri

    return allowed.unsqueeze(1), batch_segs


def find_segments_1d_wo_helper_images(
    ids: torch.Tensor, token_ids: dict[str, torch.Tensor]
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    l = ids.numel()

    a_starts = torch.nonzero(ids == token_ids["abs_start"], as_tuple=False).squeeze(-1)
    a_ends = torch.nonzero(ids == token_ids["abs_end"], as_tuple=False).squeeze(-1)

    ass: list[int] = []
    aee: list[int] = []
    i = 0
    j = 0
    while i < a_starts.numel() and j < a_ends.numel():
        if a_starts[i] < a_ends[j]:
            ass.append(int(a_starts[i].item()))
            aee.append(int(a_ends[j].item()))
            i += 1
            j += 1
        else:
            j += 1

    segs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for idx in range(len(ass)):
        a_idx = _between(ids, ass[idx], aee[idx], wanted_id=int(token_ids["abs_pad"]))
        t_end = ass[idx + 1] if idx + 1 < len(ass) else l

        o_starts = torch.nonzero(
            (ids == token_ids["obs_start"]) & (torch.arange(l, device=ids.device) >= aee[idx]) & (torch.arange(l, device=ids.device) < t_end),
            as_tuple=False,
        ).squeeze(-1)
        o_ends = torch.nonzero(
            (ids == token_ids["obs_end"]) & (torch.arange(l, device=ids.device) > aee[idx]) & (torch.arange(l, device=ids.device) <= t_end),
            as_tuple=False,
        ).squeeze(-1)

        o_all: list[torch.Tensor] = []
        p = 0
        q = 0
        while p < o_starts.numel() and q < o_ends.numel():
            if o_starts[p] < o_ends[q]:
                obs = _between(ids, int(o_starts[p].item()), int(o_ends[q].item()), None)
                if obs.numel() > 0:
                    o_all.append(obs)
                p += 1
                q += 1
            else:
                q += 1

        o_idx = (
            torch.cat(o_all, dim=0)
            if len(o_all) > 0
            else torch.empty(0, dtype=torch.long, device=ids.device)
        )
        segs.append((a_idx, o_idx))

    return segs


def build_4d_attn_wo_helper_images(
    input_ids: torch.Tensor,
    pad_mask: torch.Tensor,
    token_ids: dict[str, torch.Tensor],
    mask_latent: bool = False,
) -> torch.Tensor:
    input_ids = input_ids.cpu()
    pad_mask = pad_mask.cpu()

    bsz, seq_len = input_ids.shape
    device = input_ids.device

    causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
    valid = pad_mask.bool()
    allowed = causal.unsqueeze(0).repeat(bsz, 1, 1)
    for b in range(bsz):
        allowed[b] &= valid[b].unsqueeze(0)
        allowed[b] &= valid[b].unsqueeze(1)

    for b in range(bsz):
        segs = find_segments_1d_wo_helper_images(input_ids[b], token_ids)
        for a_idx, _ in segs:
            if a_idx.numel() == 0 or not mask_latent:
                continue
            r_idx = torch.arange(seq_len, device=device)
            rows_to_block = (r_idx.unsqueeze(0) >= a_idx.unsqueeze(1)).any(dim=0)
            if rows_to_block.any():
                allowed[b][rows_to_block.nonzero(as_tuple=False).squeeze(-1)[:, None], a_idx] = False

    return allowed.unsqueeze(1)
