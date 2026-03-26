"""Apply SpinQuant rotation to embedding weights for 4-bit quantized inference."""

from argparse import ArgumentParser

import torch


def save_rotated_embeddings(original_emb_path, rotation_matrix_path, save_emb_path):
    # Load original embedding weights: [vocab_size, embed_dim]
    emb = torch.load(original_emb_path)

    # Load SpinQuant R1 global rotation matrix (nn.Module, not a plain tensor)
    rot = torch.load(rotation_matrix_path, weights_only=False)
    rot_matrix = next(rot.parameters())

    # Apply rotation in float64 for numerical precision, then convert back to bfloat16
    # Result: rotated embedding [vocab_size, embed_dim] for 4-bit quantized inference
    emb = (emb.double() @ rot_matrix.double()).bfloat16()
    torch.save(emb, save_emb_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--embedding-path", type=str, default="./embedding.pt")
    parser.add_argument(
        "--rotation-matrix-path",
        type=str,
        default="./spinWeight/model/R1/global_rotation.pth",
    )
    parser.add_argument("--output-path", type=str, default="./embedding_rot.pt")
    args = parser.parse_args()

    save_rotated_embeddings(args.embedding_path, args.rotation_matrix_path, args.output_path)
