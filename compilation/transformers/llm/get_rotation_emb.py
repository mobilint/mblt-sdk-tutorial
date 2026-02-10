from argparse import ArgumentParser

import torch

def save_rotated_embeddings(original_emb_path, rotation_matrix_path, save_emb_path):
    emb = torch.load(original_emb_path)
    rot = torch.load(rotation_matrix_path, weights_only=False)
    rot_list = []
    for i in rot.parameters():
        rot_list.append(i)
    emb = (emb.double() @ rot_list[0].double()).bfloat16()
    torch.save(emb, save_emb_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--original-emb-path", type=str, default="./embedding.pt")
    parser.add_argument("--rotation-matrix-path", type=str, default="./spinWeight/model/R1/global_rotation.pth")
    parser.add_argument("--save-emb-path", type=str, default="./embedding_rot.pt")
    args = parser.parse_args()

    save_rotated_embeddings(args.original_emb_path, args.rotation_matrix_path, args.save_emb_path)

