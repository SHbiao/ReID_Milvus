import DataBase_Function as db
import config as cfg
#from models import reid_embedder as reid
from models import simple_cnn as scnn
import pymilvus as py
import numpy as np


def main():
    # === 1) 连接 Standalone ===
    client = db.milvuseStandalone_connect()

    # === 2) 准备向量（与 local_client_main.py 保持一致：simple_cnn + 256 维） ===
    embed_dim = 256
    print(f"[Embedding] scanning folder: {cfg.images_path}")
    embs, paths = scnn.folder_to_vecs(cfg.images_path, embed_dim=embed_dim)  # [N,256], [N]
    if embs is None or len(embs) == 0:
        raise RuntimeError("未从图片目录提取到任何向量，请检查 cfg.images_path。")
    print(f"[Embedding] total={len(embs)}, dim={embs.shape[1]}")

    # === 3) 创建集合（自动ID） ===
    co_name = "photos_scnn_256"
    if db.co_exist(client, co_name):
        db.co_drop(client, co_name)
    db.co_create_autoid(client, co_name, dim=embed_dim)

    # === 4) 插入全部向量 ===
    inserted = db.element_insert_autoid(client, co_name, vectors=embs.astype("float32"))
    print(f"[Insert] inserted rows: {inserted}")

    # === 5) Top-K 检索（以第1张图为查询向量） ===
    topk = 3
    q = embs[0].astype("float32")
    hits = db.closest_n(client, co_name, q.tolist(), topk=topk, metric="COSINE", return_vector=False)
    print(f"[Search] query = paths[0] -> {paths[0] if len(paths)>0 else '(unknown)'}")
    for i, h in enumerate(hits, 1):
        print(f"Top{i}: id={h['id']}, score={h['score']:.6f}")

    print("✅ All done.")

if __name__ == "__main__":
    main()

