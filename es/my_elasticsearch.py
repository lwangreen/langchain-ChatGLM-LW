from elasticsearch import Elasticsearch
from langchain.docstore.document import Document

class MyElasticsearch(Elasticsearch):
    def __init__(self, url, index_name="linewell-policy", fields=None) -> None:
        self.index_name = index_name
        self.client = Elasticsearch(url)
        self.fields = ["标题", "子标题", "内容"] if fields is None else fields
    def search(self, query, top_k=3) -> list:
        query_body = {
            "query": {
                "multi_match": {
                    "analyzer": "ik_smart",
                    "query": query,
                    "fields": self.fields
                }
            }    
        }
        response = self.client.search(index=self.index_name, body=query_body)
        # res = [{
        #         "page_content": hit['_source']['子标题'] + hit['_source']['内容'],
        #         "metadata": {
        #             "source": hit['_source']['标题'],
        #             "score": hit["_score"],
        #         }
        #     } for hit in response["hits"]["hits"]]
        res = [
            Document(
                page_content=hit['_source']['子标题'] + hit['_source']['内容'],
                metadata={
                    "source": hit['_source']['标题'],
                    "score": hit["_score"],
                }
            ) for hit in response["hits"]["hits"]
        ]
        if top_k == 0:
            return res
        else:
            top_k = min(top_k, len(res))
            return res[:top_k]
