import math
from typing import List, Optional, Tuple, Union

import maccel
import numpy as np
import torch
from torch import nn
from transformers.cache_utils import Cache, StaticCache
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
from transformers.utils import replace_return_docstrings

_CONFIG_FOR_DOC = "LlamaConfig"


class LlamaMXQ(LlamaPreTrainedModel, GenerationMixin):
    def __init__(
        self,
        config,
        mxq_path,
        embedding_weight_path,
        max_sub_seq: int = 192,
    ):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.acc = maccel.Accelerator(0)  # LLM allows only 1 accelerator
        mc = maccel.ModelConfig()  # LLM allows only 1 core
        mc.set_single_core_mode(1)
        self.mxq_model = maccel.Model(mxq_path, mc)
        self.mxq_model.launch(self.acc)
        self.mxq_path = mxq_path
        self.reset_cache()

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.embed_tokens.weight.data = torch.load(
            embedding_weight_path, weights_only=True, map_location=torch.device("cpu")
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

        self.is_mxq = True

        assert max_sub_seq > 0, "max_sub_seq should be greater than 0"
        # assert max_sub_seq % 64 == 0, "max_sub_seq should be multiple of 64"
        self.max_sub_seq = max_sub_seq

    def get_cache_position(self):
        return self.current_cache_position

    def reset_cache(self):
        self.current_cache_position = 0
        self.mxq_model.reset_cache_memory()

    def can_generate(self):
        return True

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        req_ret = kwargs.pop("req_ret", 1)  # required return logit length
        assert isinstance(req_ret, int), "req_ret should be an integer"

        if not ((input_ids is None) ^ (inputs_embeds is None)):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        # ------------------------ LlamaModel.forward() --------------------------------------------
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions  # False by default
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states  # False by default
        )
        use_cache = (
            use_cache if use_cache is not None else self.config.use_cache
        )  # True by default
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict  # True by default
        )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)  # (batch, seqlen, hidden_size)

        inputs_embeds_numpy = inputs_embeds.to(torch.float32).detach().numpy()

        if inputs_embeds_numpy.ndim == 3:
            inputs_embeds_numpy = np.expand_dims(
                inputs_embeds_numpy, 1
            )  # (batch, 1, seqlen,  hidden_size)

        # if cache_position is None:
        #    cache_position = torch.arange(
        #        0,
        #        inputs_embeds_numpy.shape[2],
        #        dtype=torch.long,
        #        device=inputs_embeds.device,
        #    )  # [0, 1, 2, ..., seqlen-1]
        #
        # inputs_embeds_numpy = inputs_embeds_numpy[:, :, cache_position.tolist(), :]

        seq_len = inputs_embeds_numpy.shape[2]
        assert (
            req_ret <= seq_len
        ), f"Requested return length {req_ret} is greater than input sequence length {seq_len}"
        res_ret = seq_len - req_ret + 1  # number of tokens that will not return logits
        # -------------------------------------------------------------------------------------------
        # Stage 1: Process tokens that will not return logits except the last one
        for i in range(
            math.ceil(res_ret / self.max_sub_seq)
        ):  # max_sub_seq tokens are processed at once
            seq_start = i * self.max_sub_seq
            seq_end = min((i + 1) * self.max_sub_seq, res_ret)
            tmp_logits = self.mxq_model.infer(
                inputs_embeds_numpy[:, :, seq_start:seq_end, :],
                None,
                self.current_cache_position,
            )
            self.current_cache_position += seq_end - seq_start

        out_logits = tmp_logits[0][0, 0, -1:, :]
        out_logits = torch.tensor(np.array([out_logits]), dtype=torch.float32)
        logits_all = [out_logits]
        # -------------------------------------------------------------------------------------------
        # Stage 2: Process tokens that will return logits
        for i in range(res_ret, seq_len):
            tmp_logits = self.mxq_model.infer(
                inputs_embeds_numpy[:, :, i : i + 1, :],
                None,
                self.current_cache_position,
            )
            self.current_cache_position += 1
            out_logits = tmp_logits[0][0, 0, -1:, :]
            out_logits = torch.tensor(np.array([out_logits]), dtype=torch.float32)
            logits_all.append(out_logits)

        logits = torch.cat(logits_all, dim=1)
        # -------------------------------------------------------------------------------------------
        loss = None
        if labels is not None:
            loss = ForCausalLMLoss(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (logits,)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(  # 실질적으로는 logits만 반환되고, 평가 또는 추론에 사용됨
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing inputs_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif (
                input_ids.shape[1] != cache_position.shape[0]
            ):  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {
                "input_ids": input_ids.clone(memory_format=torch.contiguous_format),
                "inputs_embeds": None,
            }

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full(
            (sequence_length, target_length),
            fill_value=min_dtype,
            dtype=dtype,
            device=device,
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(
            target_length, device=device
        ) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = (
                causal_mask.clone()
            )  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = (
                causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            )
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[
                :, :, :, :mask_length
            ].masked_fill(padding_mask, min_dtype)

    return causal_mask


def fixed_cross_entropy(
    source, target, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs
):
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(
        source, target, ignore_index=ignore_index, reduction=reduction
    )
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss


def ForCausalLMLoss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: int = None,
    ignore_index: int = -100,
    **kwargs,
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    labels = labels.to(logits.device)
    # Shift so that tokens < n predict n
    labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(logits.device)
    loss = fixed_cross_entropy(
        logits, shift_labels, num_items_in_batch, ignore_index, **kwargs
    )
    return loss
