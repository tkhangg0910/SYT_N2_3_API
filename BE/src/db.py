from pymilvus import connections, Collection, FieldSchema, DataType, CollectionSchema, utility

collection_name = "demo_collection"

connections.connect("default", host="localhost", port="19530")
def initialize_collection():
    # utility.drop_collection("demo_collection")
    if collection_name not in utility.list_collections():
        print(f"Create collection: {collection_name}")
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, max_length=128, auto_id=True),
            FieldSchema(name="company_name", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=4096),
        ]
        schema = CollectionSchema(fields, "demo collection")
        collection = Collection(collection_name, schema,)
        
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
    else:
        print(f"reference to collection: {collection_name}")
        collection = Collection(collection_name)
    
    collection.load()
    return collection

def show_data(collection):
    try:
        
        results = collection.query(
            expr="",
            output_fields=["company_name", "text"],
            limit=10 
        )
        
        print("Data in the collection:")
        for record in results:
            print(f"User ID: {record['company_name']}, Vectors: {record['text']}")
        
        return results

    except Exception as e:
        print(f"Error retrieving data: {e}")
        return None
    
show_data(initialize_collection())    


    

# def initialize_collection():
#     client = MilvusClient("./milvus_demo.db")
#     collection_name = "demo_collection"

#     if collection_name not in client.list_collections():
#         print(f"Create collection: {collection_name}")
#         client.create_collection(
#             collection_name=collection_name,
#             dimension=4096
#         )
#     else:
#         print(f"Reference to collection: {collection_name}")

#     return client, collection_name
