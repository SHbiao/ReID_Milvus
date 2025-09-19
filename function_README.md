# function_README.md

# Milvus Function 使用说明

本文件介绍了 `DataBase_Function.py` 中封装的所有函数，便于快速查找和使用。函数基于 **MilvusClient** 封装，主要覆盖：**连接、集合管理、数据插入与删除、检索、信息展示**。

---

## 📑 目录
- [连接](#连接)
  - [milvuseStandalone_connect](#milvusestandalone_connect)
  - [co_exist](#co_exist)
- [集合管理](#集合管理)
  - [co_create_manualid](#co_create_manualid)
  - [co_create_autoid](#co_create_autoid)
  - [co_drop](#co_drop)
- [数据操作](#数据操作)
  - [element_insert_manualid](#element_insert_manualid)
  - [element_insert_autoid](#element_insert_autoid)
  - [element_delete](#element_delete)
- [检索](#检索)
  - [closest_n](#closest_n)
- [展示](#展示)
  - [collection_elements_display](#collection_elements_display)
  - [collection_display](#collection_display)

---

## 连接

### `milvuseStandalone_connect`
连接 Milvus Standalone（Docker / 远程），返回 `MilvusClient` 实例。  

```python
client = milvuseStandalone_connect()
```
配置参数读取自 `config.py` (`milvus_host`, `milvus_port`)。  

---

### `co_exist`
判断集合是否存在。  

```python
if co_exist(client, "my_collection"):
    print("已存在")
else:
    print("不存在")
```

---

## 集合管理

### `co_create_manualid`
创建集合（手动 ID）。  

```python
co_create_manualid(client, "manual_768", dim=768)
```
- `auto_id=False`，插入时必须提供主键 ID。  

---

### `co_create_autoid`
创建集合（自动 ID）。  

```python
co_create_autoid(client, "auto_512", dim=512)
```
- `auto_id=True`，插入时无需提供主键。  

---

### `co_drop`
删除指定集合。  

```python
co_drop(client, "auto_512")
```
存在则删除并返回 `True`，否则返回 `False`。  

---

## 数据操作

### `element_insert_manualid`
插入带 **自定义 ID** 的数据。  

```python
ids = [1, 2, 3]
vecs = np.random.rand(3, 768).astype("float32")
element_insert_manualid(client, "manual_768", ids, vecs)
```

---

### `element_insert_autoid`
插入数据（由系统自动分配 ID）。  

```python
vecs = np.random.rand(5, 768).astype("float32")
element_insert_autoid(client, "auto_512", vecs)
```

---

### `element_delete`
按主键 ID 删除数据。  

```python
element_delete(client, "manual_768", ids=[1, 2])
```

---

## 检索

### `closest_n`
在集合中进行 **Top-K 相似检索**。  

```python
q = np.random.rand(768).astype("float32")
hits = closest_n(client, "manual_768", q.tolist(), topk=3, metric="COSINE")

for h in hits:
    print("id:", h["id"], "score:", h["score"])
```

- 支持 `COSINE`, `L2`, `IP` 等度量方式。  
- 可选参数 `return_vector=True` 返回完整向量。  

---

## 展示

### `collection_elements_display`
展示集合中的数据。  

```python
# 前 5 条
collection_elements_display(client, "manual_768", limit=5)

# 全部数据
collection_elements_display(client, "manual_768", limit="all")
```

- `limit` = 数字 → 返回前 N 条  
- `limit` = `"all"` 或 `None` → 返回全部  

---

### `collection_display`
展示集合信息。  

```python
# 查看所有集合
collection_display(client)

# 查看单个集合
collection_display(client, co_name="manual_768")
```

输出集合的：
- 行数  
- 描述  
- 主键/auto_id 配置  
- 字段 schema  

---

## ✅ 提示
- 所有函数都基于 `MilvusClient`，不支持 `connections.connect` 的 ORM API。  
- 推荐先 `milvuseStandalone_connect()`，再执行后续操作。  
- 插入前必须先 `co_create_autoid` 或 `co_create_manualid` 创建集合。  
