import numpy as np
from typing import List, Dict, Any, Optional, Union
from pymilvus import MilvusClient
import config as cfg
import pymilvus as py
import os
# ===================== 基础连接 & 集合存在性判断 =====================

def milvuseStandalone_connect():
    """
    连接 Milvus Standalone (Docker/远程)，返回 MilvusClient
    """
    url = f"http://{cfg.milvus_host}:{cfg.milvus_port}"
    client = MilvusClient(uri=url)
    print(f"[MilvusStandalone] Connected to {url}")
    return client


# def local_connect():
#     """
#     支持三种输入：
#     1) cfg.local_client_path = "D:/ReID System/milvus.db"    # 纯路径（推荐）
#     2) cfg.local_client_path = "sqlite:///D:/ReID System/milvus.db"  # 完整 URI
#     3) cfg.local_client_path = "sqlite:///:memory:"         # 内存库
#     """
#     raw = cfg.local_client_path.strip()
#
#     # 1) 内存模式：直接连
#     if raw.lower() == "sqlite:///:memory:":
#         uri = "sqlite:///:memory:"
#         print("[MilvusLite] URI =", uri)
#         py.connections.connect(alias=cfg.local_client_alias, uri=uri)
#         return py.connections.get_connection(cfg.local_client_alias)
#
#     # 2) 若已是完整 URI，则提取真实文件路径用于建目录
#     if raw.lower().startswith("sqlite:"):
#         if raw.lower().startswith("sqlite:///"):
#             db_path = raw[10:]  # 去掉前缀 "sqlite:///"
#         else:
#             # 防止错误写成 "sqlite:D:\..." 或 "sqlite:\D:\..."
#             # 统一成 "sqlite:///D:/..." 形式
#             cleaned = raw.replace("\\", "/")
#             if cleaned.lower().startswith("sqlite://"):
#                 # 只有两个斜杠 -> 补一个
#                 cleaned = "sqlite:///" + cleaned[9:]
#             else:
#                 # 只有 "sqlite:" -> 补全
#                 cleaned = "sqlite:///" + cleaned[7:]
#             raw = cleaned
#             db_path = raw[10:]
#         db_path = db_path.replace("\\", "/")
#         os.makedirs(os.path.dirname(db_path), exist_ok=True)
#         uri = f"sqlite:///{db_path}"
#     else:
#         # 3) 纯文件路径 -> 规范化 & 建目录 -> 拼 URI
#         db_path = raw.replace("\\", "/")
#         os.makedirs(os.path.dirname(db_path), exist_ok=True)
#         uri = f"sqlite:///{db_path}"
#
#     print("[MilvusLite] URI =", uri)  # 调试用，看到的必须是 sqlite:///D:/...
#     py.connections.connect(alias=cfg.local_client_alias, uri=uri)
#     return py.connections.get_connection(cfg.local_client_alias)
#
# def cloud_connect(url: str, password: str) -> MilvusClient:
#     """
#     连接到 Milvus/Zilliz。
#     说明：Zilliz Cloud 用 uri + token；这里沿用你的签名，把 url 传给 uri，把 password 传给 token。
#     本地/自建 Milvus 可把 password 设为空字符串。
#     """
#     client = MilvusClient(uri=url, token=password)
#     return client

def co_exist(client: MilvusClient, co_name: str) -> bool:
    """
    判断集合是否存在，存在返回 True，否则 False。
    """
    if client.has_collection(collection_name=co_name):
        print("collection found")
        return True
    else:
        print("collection not found")
        return False

# ===================== 创建 / 删除集合 =====================

def co_create_manualid(client, co_name: str, dim: int = 768) -> bool:
    """
    创建集合：手动指定 id
    - 默认维度 768
    - auto_id=False
    - consistency_level="Strong" (默认值)
    """
    if client.has_collection(collection_name=co_name):
        return False

    client.create_collection(
        collection_name=co_name,
        dimension=dim,            # 特征向量维度
        auto_id=False,            # 手动指定 id
        consistency_level="Strong"  # 默认一致性等级
    )
    return True


def co_create_autoid(client, co_name: str, dim: int = 768) -> bool:
    """
    创建集合：自动生成 id
    - 默认维度 768
    - auto_id=True
    - consistency_level="Strong" (默认值)
    """
    if client.has_collection(collection_name=co_name):
        return False

    client.create_collection(
        collection_name=co_name,
        dimension=dim,            # 特征向量维度
        auto_id=True,             # 自动生成 id
        consistency_level="Strong"  # 默认一致性等级
    )
    return True

def co_drop(client: MilvusClient, co_name: str) -> bool:
    """
    删除集合（drop）。若存在则删除并返回 True，不存在返回 False。
    """
    if co_exist(client, co_name):
        client.drop_collection(collection_name=co_name)  # 注意：MilvusClient 使用 drop_collection
        print("collection dropped")
        return True
    else:
        print("collection does not exist")
        return False

# ===================== 插入 / 删除元素 =====================

def element_insert_manualid(client, co_name: str, ids: List[Union[int, str]], vectors: np.ndarray) -> int:
    """
    插入数据到手动ID集合
    - ids: 主键列表（长度必须与向量数量一致）
    - vectors: numpy 数组，形状 [N, dim]
    返回: 插入条数
    """
    if not co_exist(client,co_name):
        raise ValueError(f"Collection {co_name} not found")

    data = [{"id": i, "vector": v.astype("float32").tolist()} for i, v in zip(ids, vectors)]
    client.insert(collection_name=co_name, data=data)
    return len(data)


def element_insert_autoid(client, co_name: str, vectors: np.ndarray) -> int:
    """
    插入数据到自动ID集合
    - vectors: numpy 数组，形状 [N, dim]
    返回: 插入条数
    """
    if not co_exist(client,co_name):
        raise ValueError(f"Collection {co_name} not found")

    data = [{"vector": v.astype("float32").tolist()} for v in vectors]
    client.insert(collection_name=co_name, data=data)
    return len(data)

def element_delete(client, co_name: str, ids: List[Union[int, str]]) -> int:
    """
    按 ID 删除集合中的数据
    - ids: 要删除的主键列表
    返回: 删除条数
    """
    if not client.has_collection(co_name):
        raise ValueError(f"Collection {co_name} not found")

    client.delete(collection_name=co_name, ids=ids)
    return len(ids)



# ===================== Top-K 相似检索 =====================

def closest_n(client: MilvusClient,co_name: str,tensor: List[float],topk: int,
              metric: str = "COSINE",return_vector: bool = False) -> List[Dict[str, Any]]:
    """
    在指定集合中对给定查询向量进行 Top-K 相似检索。
    返回：[{id: ..., score: ...}]；当 return_vector=True 时附加 "vector" 字段。
    """
    if not co_exist(client, co_name):
        return []

    # 准备查询向量（float32、一维）
    query_vector = np.asarray(tensor, dtype=np.float32).reshape(-1)
    # 可选：余弦相似下做归一化
    if metric.upper() == "COSINE":
        nrm = np.linalg.norm(query_vector) + 1e-12
        query_vector = (query_vector / nrm).astype(np.float32)

    # 输出字段
    out_fields = ["id"]
    if return_vector:
        out_fields.append("vector")

    results = client.search(
        collection_name=co_name,
        data=[query_vector.tolist()],
        limit=topk,
        output_fields=out_fields,
        search_params={"metric_type": metric.upper()}
    )

    info: List[Dict[str, Any]] = []
    if results:
        for hit in results[0]:
            row = {"id": hit["id"], "score": float(hit["distance"])}
            if return_vector and "vector" in hit:
                row["vector"] = hit["vector"]
            info.append(row)
    return info


# ===================== 属性和内容的展示  =====================
def collection_elements_display(client, co_name: str, limit: int | str | None = None):
    """
    展示集合元素：
      - limit 为 None 或 "all"：输出全部元素（⚠️ 大集合可能非常慢/占内存）
      - limit 为数字：输出前 limit 条
    """
    try:
        # 计算最终 limit
        if limit is None or (isinstance(limit, str) and limit.lower() == "all"):
            stats = client.get_collection_stats(co_name)
            row_count = int(stats["row_count"])
            limit_val = row_count
            print(f"[Info] limit=all, collection '{co_name}' total rows={row_count}")
        else:
            limit_val = int(limit)

        if limit_val == 0:
            print(f"[Info] collection '{co_name}' is empty.")
            return []

        results = client.query(
            collection_name=co_name,
            filter="",            # 不加过滤条件
            output_fields=["*"],  # 返回所有字段
            limit=limit_val
        )

        print(f"Collection '{co_name}' first {limit_val} elements:")
        for i, r in enumerate(results, 1):
            print(f"{i}: {r}")
        return results

    except Exception as e:
        print(f"[Error] query collection failed: {e}")
        return None


def collection_display(client, co_name: str | None = None):
    """
    展示集合信息：
      - co_name 为 None：输出所有集合（名称、行数、字段/schema）
      - 指定 co_name：仅输出该集合（行数、字段/schema）
    """
    try:
        names = [co_name] if co_name else client.list_collections()
        if not names:
            print("[Info] No collections found.")
            return []

        print("=== Collections in Milvus ===")
        for name in names:
            # 行数
            try:
                stats = client.get_collection_stats(name)
                row_count = int(stats.get("row_count", 0))
            except Exception as se:
                row_count = "N/A"
                print(f"[Warn] get_collection_stats failed for '{name}': {se}")

            # schema / 字段
            try:
                desc = client.describe_collection(name)
            except Exception as de:
                print(f"[Error] describe_collection failed for '{name}': {de}")
                continue

            print(f"\nCollection: {name}")
            print(f"  - row_count : {row_count}")
            print(f"  - description: {desc.get('description', '')}")
            print(f"  - auto_id   : {desc.get('auto_id', False)}")
            print(f"  - fields    :")
            for f in desc.get("fields", []):
                is_pk = " [primary]" if f.get("is_primary") else ""
                is_auto = " [auto_id]" if f.get("auto_id") else ""
                print(f"      * {f['name']} ({f['type']}){is_pk}{is_auto}")

        return names

    except Exception as e:
        print(f"[Error] collection_display failed: {e}")
        return None



# =============== 测试场景 ===============

# def test_manual_id_flow(client: MilvusClient):
#     """手动ID集合：建 -> 插 -> 查 -> 删元素 -> 删集合"""
#     co = "test_manual_768"
#     print("\n=== [ManualID] create ===")
#     created = co_create_manualid(client, co, dim=768)
#     print("created:", created)
#
#     print("\n=== [ManualID] insert 3 rows ===")
#     vecs = np.random.rand(3, 768).astype("float32")
#     count = element_insert_manualid(client, co, ids=[101, 102, 103], vectors=vecs)
#     print("inserted:", count)
#
#     print("\n=== [ManualID] search top-3 ===")
#     q = vecs[0]  # 以第一条为查询向量
#     hits = closest_n(client, co, q.tolist(), topk=1, metric="COSINE", return_vector=False)
#     for i, h in enumerate(hits, 1):
#         print(f"#{i}: id={h['id']}, score={h['score']}")
#
#     print("\n=== [ManualID] delete 2 rows (101, 102) ===")
#     deleted = element_delete(client, co, ids=[101, 102])
#     print("deleted:", deleted)
#
#     print("\n=== [ManualID] drop collection ===")
#     dropped = co_drop(client, co)
#     print("dropped:", dropped)
#
#
# def test_auto_id_flow(client: MilvusClient):
#     """自动ID集合：建 -> 插 -> 查 -> 删集合（不演示按id删，因为不知道系统分配的主键）"""
#     co = "test_auto_768"
#     print("\n=== [AutoID] create ===")
#     created = co_create_autoid(client, co, dim=768)
#     print("created:", created)
#
#     print("\n=== [AutoID] insert 5 rows ===")
#     vecs = np.random.rand(5, 768).astype("float32")
#     count = element_insert_autoid(client, co, vectors=vecs)
#     print("inserted:", count)
#
#     print("\n=== [AutoID] search top-3 ===")
#     q = vecs[0]
#     hits = closest_n(client, co, q.tolist(), topk=1, metric="COSINE", return_vector=False)
#     for i, h in enumerate(hits, 1):
#         print(f"#{i}: id={h['id']}, score={h['score']}")
#
#     print("\n=== [AutoID] drop collection ===")
#     dropped = co_drop(client, co)
#     print("dropped:", dropped)

# =============== 入口 ===============

# if __name__ == "__main__":
#     # 推荐用环境变量管理凭证（避免把 token 写进代码）
#     url = "https://in03-085b4a676ed7610.serverless.aws-eu-central-1.cloud.zilliz.com"
#     password = "4b9e6d8f16129c77c71f0268ffd9c14683c88feb986736080a9bfa621f73e8e5bf21088fee4accacd26bcf379af1f05301fd1e65"
#     URL = url
#     TOKEN = password
#
#     client = cloud_connect(URL, TOKEN)
#
#     print("== collections before test ==")
#     print(client.list_collections())
#
#     # 跑两套流程
#     test_manual_id_flow(client)
#     test_auto_id_flow(client)
#
#     print("\n== collections after test ==")
#     print(client.list_collections())
#
#     print("\nAll tests done ✅")