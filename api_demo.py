import os
from typing import List, Optional

import nltk
import pydantic
import uvicorn
from fastapi import Body, FastAPI, File, Form, Query, UploadFile, WebSocket
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing_extensions import Annotated
from starlette.responses import RedirectResponse

from configs.model_config import (KB_ROOT_PATH, EMBEDDING_DEVICE,
                                  EMBEDDING_MODEL, NLTK_DATA_PATH,
                                  VECTOR_SEARCH_TOP_K, LLM_HISTORY_LEN, OPEN_CROSS_DOMAIN, KNOWLEDGE_BASE_NAME)
import models.shared as shared
from models.loader.args import parser
from models.loader import LoaderCheckPoint
from chains.chat_model import ChatModel

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path


class BaseResponse(BaseModel):
    code: int = pydantic.Field(200, description="HTTP status code")
    msg: str = pydantic.Field("success", description="HTTP status message")

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
            }
        }


class ListDocsResponse(BaseResponse):
    data: List[str] = pydantic.Field(..., description="List of document names")

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
                "data": ["doc1.docx", "doc2.pdf", "doc3.txt"],
            }
        }


class ChatMessage(BaseModel):
    question: str = pydantic.Field(..., description="Question text")
    response: str = pydantic.Field(..., description="Response text")
    history: List[List[str]] = pydantic.Field(..., description="History text")
    source_documents: List[str] = pydantic.Field(
        ..., description="List of source documents and their scores"
    )

    class Config:
        schema_extra = {
            "example": {
                "question": "工伤保险如何办理？",
                "response": "根据已知信息，可以总结如下：\n\n1. 参保单位为员工缴纳工伤保险费，以保障员工在发生工伤时能够获得相应的待遇。\n2. 不同地区的工伤保险缴费规定可能有所不同，需要向当地社保部门咨询以了解具体的缴费标准和规定。\n3. 工伤从业人员及其近亲属需要申请工伤认定，确认享受的待遇资格，并按时缴纳工伤保险费。\n4. 工伤保险待遇包括工伤医疗、康复、辅助器具配置费用、伤残待遇、工亡待遇、一次性工亡补助金等。\n5. 工伤保险待遇领取资格认证包括长期待遇领取人员认证和一次性待遇领取人员认证。\n6. 工伤保险基金支付的待遇项目包括工伤医疗待遇、康复待遇、辅助器具配置费用、一次性工亡补助金、丧葬补助金等。",
                "history": [
                    [
                        "工伤保险是什么？",
                        "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
                    ]
                ],
                "source_documents": [
                    "出处 [1] 广州市单位从业的特定人员参加工伤保险办事指引.docx：\n\n\t( 一)  从业单位  (组织)  按“自愿参保”原则，  为未建 立劳动关系的特定从业人员单项参加工伤保险 、缴纳工伤保 险费。",
                    "出处 [2] ...",
                    "出处 [3] ...",
                ],
            }
        }


def get_folder_path(local_doc_id: str):
    return os.path.join(KB_ROOT_PATH, local_doc_id, "content")


def get_vs_path(local_doc_id: str):
    return os.path.join(KB_ROOT_PATH, local_doc_id, "vector_store")


def get_file_path(local_doc_id: str, doc_name: str):
    return os.path.join(KB_ROOT_PATH, local_doc_id, "content", doc_name)


async def local_doc_chat(
        # knowledge_base_id: str = Body(..., description="Knowledge Base Name", example="kb1"),
        question: str = Body(..., description="Question", example="工伤保险是什么？"),
        history: List[List[str]] = Body(
            [],
            description="History of previous questions and answers",
            example=[
                [
                    "工伤保险是什么？",
                    "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
                ]
            ],
        ),
):
    if not os.path.exists(vs_path):
        return ChatMessage(
            question=question,
            response=f"vs_path {vs_path} not found",
            history=history,
            source_documents=[],
        )
    else:
        for resp, history in chat_model.get_knowledge_based_answer(
                query=question, vs_path=vs_path, chat_history=history, streaming=False
        ):
            pass
        source_documents = [
            f"""出处 [{inum + 1}] {os.path.split(doc.metadata['source'])[-1]}：\n\n{doc.page_content}\n\n"""
            f"""相关度：{doc.metadata['score']}\n\n"""
            for inum, doc in enumerate(resp["source_documents"])
        ]

        return ChatMessage(
            question=question,
            response=resp["result"],
            history=history,
            source_documents=source_documents,
        )

async def document():
    return RedirectResponse(url="/docs")

def get_vs_list():
    if not os.path.exists(KB_ROOT_PATH):
        return []
    lst = os.listdir(KB_ROOT_PATH)
    if not lst:
        return []
    lst.sort()
    return lst

def get_existing_vs_path(local_doc_id=None):
    if local_doc_id is not None:
        vs_path = get_vs_path(local_doc_id)
        if os.path.exists(vs_path):
            print(f"Exsiting knowledge base loaded from exsiting {vs_path}")
            return vs_path
        else:
            print(f"Error: {vs_path} is not a valid vector store path")
    vs_list = get_vs_list()
    if len(vs_list) > 0:
        print(f"Exsiting knowledge base loaded from {os.path.join(KB_ROOT_PATH, vs_list[-1], 'vector_store')}")
        return os.path.join(KB_ROOT_PATH, vs_list[-1], "vector_store")
    else:
        print("Error: no exsiting vector store found")
        return None

def get_new_vs_path(filepath):
    if filepath is not None:
        vs_path, _ = chat_model.init_knowledge_vector_store(filepath)
        if vs_path is None:
            print(f"Error: {filepath} is not a valid file path")
        else:
            print(f"New knowledge base loaded from {filepath}")
            return vs_path
    else:
        vs_path = None
    while not vs_path:
        print("注意输入的路径是完整的文件路径，例如knowledge_base/`knowledge_base_id`/content/file.md，多个路径用英文逗号分割")
        filepath = input("Input your local knowledge file path 请输入本地知识文件路径：")
        
        # 判断 filepath 是否为空，如果为空的话，重新让用户输入,防止用户误触回车
        if not filepath:
            continue
        vs_path, _ = chat_model.init_knowledge_vector_store(filepath)
        if vs_path is not None:
            print(f"New knowledge base loaded from {filepath}")
            return vs_path

def api_start(args):
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.get("/", response_model=BaseResponse)(document)
    app.post("/local_doc_qa/local_doc_chat", response_model=ChatMessage)(local_doc_chat)
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="the address of the api server")
    parser.add_argument("--port", type=int, default=7861,
                        help="the port of the api server")
    parser.add_argument("--new_vs", type=bool, default=False, 
                        help="whether to create a new vector store")
    parser.add_argument("--filepath", type=str, default="data/cleaned_data",
                        help="path to the local knowledge file")
    parser.add_argument("--model_name", type=str, default="chatglm2-6b",
                        help="the name of the llm model")
    parser.add_argument("--es_url", type=str, default="http://127.0.0.1:9200",
                        help="the url of the es server")
    parser.add_argument("--top_k", type=int, default=3,
                        help="the top k of the vector search")
    parser.add_argument("--history_len", type=int, default=LLM_HISTORY_LEN,
                        help="the history len of the llm model")
    parser.add_argument("--es_top_k", type=int, default=3,
                        help="the top k of the es search")
    parser.add_argument("--rerank_type", type=str, default="cross-encoder",
                        help="the type of the rerank model, support `cross-encoder` and `text2vec`, other values will turn off the rerank")
    parser.add_argument("--rerank_model", type=str, default="/root/share/cross-encoder-bert-base",
                        help="the name or path of the rerank model")
    args = parser.parse_args()
    
    vs_path = get_existing_vs_path() if not args.new_vs else get_new_vs_path(args.filepath)
    
    args_dict = vars(args)
    chat_model = ChatModel()
    kw_list = ["es_url", "top_k", "history_len", "es_top_k", "rerank_type", "rerank_model"]
    kwarg_dict = {k: args_dict[k] for k in kw_list}
    chat_model.init_cfg(args_dict=args_dict, **kwarg_dict)
    
    print("init llm success")
    answer_result_stream_result = chat_model.llm_model_chain(
        {"prompt": "你好", "history": [], "streaming": False})
    for answer_result in answer_result_stream_result['answer_result_stream']:
        print(answer_result.llm_output)
    
    api_start(args)
