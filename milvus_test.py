from pymilvus import connections, utility
import DataBase_Function as db
# 连接到 Milvus
client=db.milvuseStandalone_connect()
co_name="photos_scnn_256"
# 列出所有集合
# print("All collections:", utility.list_collections())
db.collection_display(client)

db.collection_display(client,co_name)
#db.collection_elements_display(client,co_name)
# db.collection_elements_display(client,co_name,10)