import qbruntime
import torch


class BertMXQ(torch.nn.Module):
    def __init__(self, mxq_path, weight_path):
        super().__init__()

        weight_dict = torch.load(weight_path, weights_only=True)
        self.word_embeddings = torch.nn.Embedding.from_pretrained(weight_dict["word_embeddings"])
        self.token_type_embeddings = torch.nn.Embedding.from_pretrained(weight_dict["token_type_embeddings"])
        self.position_embeddings = torch.nn.Embedding.from_pretrained(weight_dict["position_embeddings"])
        layernorm_weight = weight_dict["layernorm_weight"]
        layernorm_bias = weight_dict["layernorm_bias"]

        self.layernorm = torch.nn.LayerNorm(
            layernorm_weight.shape[0],
            eps=1e-12,
        )
        self.layernorm.weight.data = layernorm_weight
        self.layernorm.bias.data = layernorm_bias

        self.acc = qbruntime.Accelerator()
        mc = qbruntime.ModelConfig()
        mc.set_single_core_mode(None, [qbruntime.CoreId(qbruntime.Cluster.Cluster0, qbruntime.Core.Core0)])
        self.model = qbruntime.Model(mxq_path, mc)
        self.model.launch(self.acc)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ):
        if input_ids is None:
            raise ValueError("input_ids must be provided.")
        if token_type_ids is None:
            raise ValueError("token_type_ids must be provided.")

        word_embed = self.word_embeddings(input_ids)
        token_type_embed = self.token_type_embeddings(token_type_ids)
        position_embed = self.position_embeddings(torch.arange(input_ids.shape[1]))
        embedded_text = word_embed + token_type_embed + position_embed
        embedded_text = self.layernorm(embedded_text)
        embedded_text = embedded_text.to(torch.float32).contiguous().detach().numpy()
        output = self.model.infer([embedded_text])
        if output is None:
            raise RuntimeError("Model inference returned no outputs.")
        return torch.from_numpy(output[0]).squeeze()

    def dispose(self):
        self.model.dispose()
