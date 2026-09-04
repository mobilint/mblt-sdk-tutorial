import qbruntime
import torch
from transformers import BertModel


def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    expanded_mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
    return (token_embeddings * expanded_mask).sum(dim=1) / expanded_mask.sum(dim=1).clamp(min=1e-9)


class BertMXQ(torch.nn.Module):
    def __init__(self, mxq_path, model_path):
        super().__init__()

        source_model = BertModel.from_pretrained(model_path)
        self.embeddings = source_model.embeddings.eval()
        self.embeddings.requires_grad_(False)
        del source_model

        self.acc = qbruntime.Accelerator()
        model_config = qbruntime.ModelConfig()
        model_config.set_single_core_mode(
            None,
            [qbruntime.CoreId(qbruntime.Cluster.Cluster0, qbruntime.Core.Core0)],
        )
        self.model = qbruntime.Model(str(mxq_path), model_config)
        self.model.launch(self.acc)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ):
        if input_ids is None:
            raise ValueError("input_ids must be provided.")
        if attention_mask is None:
            raise ValueError("attention_mask must be provided.")
        if token_type_ids is None:
            raise ValueError("token_type_ids must be provided.")
        if input_ids.shape[0] != 1:
            raise ValueError("BertMXQ supports a batch size of 1.")
        if not torch.all(attention_mask == 1):
            raise ValueError("BertMXQ does not support padded inputs.")

        embedded_text = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )
        embedded_text = embedded_text.to(torch.float32).contiguous().numpy()
        output = self.model.infer([embedded_text])
        if output is None:
            raise RuntimeError("Model inference returned no outputs.")
        token_embeddings = torch.from_numpy(output[0]).reshape(input_ids.shape[0], input_ids.shape[1], -1)
        return mean_pooling(token_embeddings, attention_mask)

    def dispose(self):
        self.model.dispose()
