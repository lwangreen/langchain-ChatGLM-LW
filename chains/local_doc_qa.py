from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from vectorstores import MyFAISS
from langchain.document_loaders import UnstructuredFileLoader, TextLoader, CSVLoader, Docx2txtLoader
from configs.model_config import *
import datetime
from textsplitter import ChineseTextSplitter
from typing import List
from utils import torch_gc
from tqdm import tqdm
from pypinyin import lazy_pinyin
from loader import UnstructuredPaddleImageLoader, UnstructuredPaddlePDFLoader
from models.base import (BaseAnswer,
                         AnswerResult)
from models.loader.args import parser
from models.loader import LoaderCheckPoint
import models.shared as shared
from agent import bing_search
from langchain.docstore.document import Document
from functools import lru_cache
from textsplitter.zh_title_enhance import zh_title_enhance
from langchain.chains.base import Chain


# patch HuggingFaceEmbeddings to make it hashable
def _embeddings_hash(self):
    return hash(self.model_name)


HuggingFaceEmbeddings.__hash__ = _embeddings_hash


# will keep CACHED_VS_NUM of vector store caches
@lru_cache(CACHED_VS_NUM)
def load_vector_store(vs_path, embeddings):
    return MyFAISS.load_local(vs_path, embeddings)


def tree(filepath, ignore_dir_names=None, ignore_file_names=None):
    """返回两个列表，第一个列表为 filepath 下全部文件的完整路径, 第二个为对应的文件名"""
    if ignore_dir_names is None:
        ignore_dir_names = []
    if ignore_file_names is None:
        ignore_file_names = []
    ret_list = []
    if isinstance(filepath, str):
        if not os.path.exists(filepath):
            print("路径不存在")
            return None, None
        elif os.path.isfile(filepath) and os.path.basename(filepath) not in ignore_file_names:
            return [filepath], [os.path.basename(filepath)]
        elif os.path.isdir(filepath) and os.path.basename(filepath) not in ignore_dir_names:
            for file in os.listdir(filepath):
                fullfilepath = os.path.join(filepath, file)
                if os.path.isfile(fullfilepath) and os.path.basename(fullfilepath) not in ignore_file_names:
                    ret_list.append(fullfilepath)
                if os.path.isdir(fullfilepath) and os.path.basename(fullfilepath) not in ignore_dir_names:
                    ret_list.extend(tree(fullfilepath, ignore_dir_names, ignore_file_names)[0])
    return ret_list, [os.path.basename(p) for p in ret_list]


def load_file(filepath, sentence_size=SENTENCE_SIZE, using_zh_title_enhance=ZH_TITLE_ENHANCE):
    if filepath.lower().endswith(".md"):
        loader = UnstructuredFileLoader(filepath, mode="elements")
        textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        docs = loader.load_and_split(text_splitter=textsplitter)
        #docs = loader.load()
    elif filepath.lower().endswith(".txt"):
        loader = TextLoader(filepath, autodetect_encoding=True)
        textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        docs = loader.load_and_split(textsplitter)
    elif filepath.lower().endswith(".pdf"):
        loader = UnstructuredPaddlePDFLoader(filepath)
        textsplitter = ChineseTextSplitter(pdf=True, sentence_size=sentence_size)
        docs = loader.load_and_split(textsplitter)
    elif filepath.lower().endswith(".jpg") or filepath.lower().endswith(".png"):
        loader = UnstructuredPaddleImageLoader(filepath, mode="elements")
        textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        docs = loader.load_and_split(text_splitter=textsplitter)
    elif filepath.lower().endswith(".csv"):
        loader = CSVLoader(filepath)
        docs = loader.load()
    elif filepath.lower().endswith(".docx") or filepath.lower().endswith(".doc"):   # 单独读取 docx 类型文件，保留换行符。yunze 2023-07-10
        loader = Docx2txtLoader(filepath)
        textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        docs = loader.load_and_split(text_splitter=textsplitter)
    else:
        loader = UnstructuredFileLoader(filepath, mode="elements")
        textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        docs = loader.load_and_split(text_splitter=textsplitter)
    if using_zh_title_enhance:
        docs = zh_title_enhance(docs)
    write_check_file(filepath, docs)
    return docs


def write_check_file(filepath, docs):
    folder_path = os.path.join(os.path.dirname(filepath), "tmp_files")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fp = os.path.join(folder_path, 'load_file.txt')
    # with open(fp, 'a+', encoding='utf-8') as fout:
    with open(fp, 'w', encoding='utf-8') as fout:
        fout.write("filepath=%s,len=%s" % (filepath, len(docs)))
        fout.write('\n')
        for i in docs:
            fout.write(str(i))
            fout.write('\n')
        fout.close()


def generate_prompt(related_docs: List[str],
                    query: str,
                    prompt_template: str = PROMPT_TEMPLATE, ) -> str:
    context = "\n".join([doc.page_content for doc in related_docs])
    prompt = prompt_template.replace("{question}", query).replace("{context}", context.replace('\n',''))
    return prompt

def generate_prompt_with_history(related_docs: List[str],
                    query: str, history_query: str,
                    prompt_template: str = PROMPT_TEMPLATE_WITH_HISTORY, ) -> str:
    context = "\n".join([doc.page_content for doc in related_docs])
    prompt = prompt_template.replace("{question}", query).replace("{context}", context.replace('\n','')).replace("{history}", history_query)
    return prompt

def generate_autoprompt(doc_page_content: str,
                    prompt_template: str = AUTOPROMPT_TEMPLATE, ) -> str:
    prompt = prompt_template.replace("{context}", doc_page_content.replace('\n',''))
    return prompt


def search_result2docs(search_results):
    docs = []
    for result in search_results:
        doc = Document(page_content=result["snippet"] if "snippet" in result.keys() else "",
                       metadata={"source": result["link"] if "link" in result.keys() else "",
                                 "filename": result["title"] if "title" in result.keys() else ""})
        docs.append(doc)
    return docs


class LocalDocQA:
    llm_model_chain: Chain = None
    embeddings: object = None
    top_k: int = VECTOR_SEARCH_TOP_K
    chunk_size: int = CHUNK_SIZE
    chunk_conent: bool = True
    score_threshold: int = VECTOR_SEARCH_SCORE_THRESHOLD

    def init_cfg(self,
                 embedding_model: str = EMBEDDING_MODEL,
                 embedding_device=EMBEDDING_DEVICE,
                 llm_model: Chain = None,
                 top_k=VECTOR_SEARCH_TOP_K,
                 ):
        self.llm_model_chain = llm_model
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model],
                                                model_kwargs={'device': embedding_device})
        self.top_k = top_k

    def init_knowledge_vector_store(self,
                                    filepath: str or List[str],
                                    vs_path: str or os.PathLike = None,
                                    sentence_size=SENTENCE_SIZE):
        loaded_files = []
        failed_files = []
        if isinstance(filepath, str):
            if not os.path.exists(filepath):
                print("路径不存在")
                return None
            elif os.path.isfile(filepath):
                file = os.path.split(filepath)[-1]
                try:
                    docs = load_file(filepath, sentence_size)
                    logger.info(f"{file} 已成功加载")
                    loaded_files.append(filepath)
                except Exception as e:
                    logger.error(e)
                    logger.info(f"{file} 未能成功加载")
                    return None
            elif os.path.isdir(filepath):
                docs = []
                for fullfilepath, file in tqdm(zip(*tree(filepath, ignore_dir_names=['tmp_files'])), desc="加载文件"):
                    try:
                        docs += load_file(fullfilepath, sentence_size)
                        loaded_files.append(fullfilepath)
                    except Exception as e:
                        logger.error(e)
                        failed_files.append(file)

                if len(failed_files) > 0:
                    logger.info("以下文件未能成功加载：")
                    for file in failed_files:
                        logger.info(f"{file}\n")

        else:
            docs = []
            for file in filepath:
                try:
                    docs += load_file(file)
                    logger.info(f"{file} 已成功加载")
                    loaded_files.append(file)
                except Exception as e:
                    logger.error(e)
                    logger.info(f"{file} 未能成功加载")
        if len(docs) > 0:
            logger.info("文件加载完毕，正在生成向量库")
            if vs_path and os.path.isdir(vs_path) and "index.faiss" in os.listdir(vs_path):
                vector_store = load_vector_store(vs_path, self.embeddings)
                vector_store.add_documents(docs)
                torch_gc()
            else:
                if not vs_path:
                    vs_path = os.path.join(KB_ROOT_PATH,
                                           f"""{"".join(lazy_pinyin(os.path.splitext(file)[0]))}_FAISS_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}""",
                                           "vector_store")
                vector_store = MyFAISS.from_documents(docs, self.embeddings)  # docs 为Document列表
                torch_gc()

            vector_store.save_local(vs_path)
            return vs_path, loaded_files
        else:
            logger.info("文件均未成功加载，请检查依赖包或替换为其他文件再次上传。")

            return None, loaded_files

    def one_knowledge_add(self, vs_path, one_title, one_conent, one_content_segmentation, sentence_size):
        try:
            if not vs_path or not one_title or not one_conent:
                logger.info("知识库添加错误，请确认知识库名字、标题、内容是否正确！")
                return None, [one_title]
            docs = [Document(page_content=one_conent + "\n", metadata={"source": one_title})]
            if not one_content_segmentation:
                text_splitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
                docs = text_splitter.split_documents(docs)
            if os.path.isdir(vs_path) and os.path.isfile(vs_path + "/index.faiss"):
                vector_store = load_vector_store(vs_path, self.embeddings)
                vector_store.add_documents(docs)
            else:
                vector_store = MyFAISS.from_documents(docs, self.embeddings)  ##docs 为Document列表
            torch_gc()
            vector_store.save_local(vs_path)
            return vs_path, [one_title]
        except Exception as e:
            logger.error(e)
            return None, [one_title]
    
    #Luming added 20230630
    def load_selected_file_knowledge(self, doc_list):
        docs = []
        for file in doc_list:
            docs += load_file(file)
        partial_vector = MyFAISS.from_documents(docs, self.embeddings)  ##docs 为Document列表
        return partial_vector
    
    #Luming added 20230712
    def get_keywords_from_autoprompt():
        keywords = []

        return keywords

    # 在指定文档中搜索
    def similarity_search_within_docx_files(self, vector_store, query, loaded_files):
        #print("OUTPUT match_doc_name: ", match_doc_names)
        if(USE_HIERARCHY_FAISS):
            match_doc_names = vector_store.compare_similarity_query_doc(query, loaded_files, doc_name_mode=True)
            if(len(match_doc_names)>0):
                partial_vectorstore = self.load_selected_file_knowledge(match_doc_names) #inputs=[select_vs, files, sentence_size, chatbot, vs_add, vs_add]
                related_docs_with_score, len_context = partial_vectorstore.similarity_search_with_score(query, k=self.top_k, match_docs=match_doc_names,)
                #print("OUTPUT len_context1:", len_context)
                if(len_context < self.chunk_size*self.top_k): # Cannot get sufficient information from local knowledge. # 放宽限制，移除长度限制 *self.top_k 用以扩大文档内搜索结果的适用范围。 yunze 2023-07-10
                    print("IN regenerate answer, hierarchy faiss")
                    related_docs_with_score, len_context = partial_vectorstore.similarity_search_with_score(query, k=self.top_k, match_docs=match_doc_names)
                    #print("OUTPUT len_context2:", len_context)
            else:
                print("IN regenerate answer, hierarchy faiss, no match files")
                related_docs_with_score, _ = vector_store.similarity_search_with_score(query, k=self.top_k, match_docs = [])
        else:
            print("IN regenerate answer")
            related_docs_with_score, _ = vector_store.similarity_search_with_score(query, k=self.top_k, match_docs = [])
        return related_docs_with_score
    
    def generate_intent_keywords(self, query_autoprompt):
        answer_result = self.llm.generatorAnswer(prompt=query_autoprompt, streaming=False)
        resp = next(answer_result).llm_output["answer"]
        resp = resp.replace("意图：","").replace("关键词：","").replace("\n","").replace  ("，"," ")
        print("OUTPUT intent keywords:", resp)
        return resp


    def get_knowledge_based_answer(self, query, vs_path, loaded_files=[], chat_history=[], streaming: bool = STREAMING):
        vector_store = load_vector_store(vs_path, self.embeddings)
        vector_store.chunk_size = self.chunk_size
        vector_store.chunk_conent = self.chunk_conent
        vector_store.score_threshold = self.score_threshold
        #Luming modified 20230630
        #print("DEBUG, ", query, loaded_files)
        if not len(loaded_files):
            for d in os.listdir(DOC_PATH):
              if os.path.isfile(DOC_PATH+'/'+d):
                   loaded_files.append(DOC_PATH+'/'+d)
            print("DEBUG, ", loaded_files)

        # if AUTO_PROMPT: # Call Autoprompt
        #     doc_page_contents = vector_store.similarity_search_in_doc_for_autoprompt(query, k=self.top_k)
        #     print("OUTPUT doc_page_contents:", doc_page_contents)
        #     for page_content in doc_page_contents:
        #         query_autoprompt = generate_autoprompt(page_content)
        #         intent_keywords = self.generate_intent_keywords(query_autoprompt)
        #         print("OUTPUT intent_keywords:", intent_keywords)

        #if loaded_files[0].endswith(".docx"):

        # Yunze. 23/08/12.
        # 多轮对话功能，思路：首先假设单轮对话可行，若阈值低于设定则进入多轮对话模式，即迭代加入先前的问题，直到相似度满足条件或 history 用完
        # 多轮对话可尝试修改 Prompt，即给出 历史问题 和 本轮问题，至于是否需要纳入历史回答有待考察。

        # 假设单轮对话
        if len(loaded_files):
            related_docs_with_score = self.similarity_search_within_docx_files(vector_store, query, loaded_files)
        else:
            related_docs_with_score, _ = vector_store.similarity_search_with_score(query, k=self.top_k, match_docs = [])
        # 判断是否进入多轮对话
        #print('history: ', chat_history)
        history_query = ""
        if related_docs_with_score[0].metadata['score'] >= MULTI_DIALOGUE_THRESHOLD:
            print("Multi Diag Mode")
            for history in chat_history[::-1]: # 依次加入历史
                history_query = history[0]
                if len(loaded_files):
                    related_docs_with_score = self.similarity_search_within_docx_files(vector_store, history_query+' '+query, loaded_files)
                else:
                    related_docs_with_score, _ = vector_store.similarity_search_with_score(history_query+' '+query, k=self.top_k, match_docs = [])
                if related_docs_with_score[0].metadata['score'] < MULTI_DIALOGUE_THRESHOLD:
                    print("Find suitable history: {}".format(history_query))
                    break
        
        #print("OUTPUT related_docs_with_score:", related_docs_with_score)
        torch_gc()


        if len(related_docs_with_score) > 0:
            if len(history_query):
                prompt = generate_prompt_with_history(related_docs_with_score, query, history_query)
            else:
                prompt = generate_prompt(related_docs_with_score, query)
        else:
            prompt = query
        # for answer_result in self.llm.generatorAnswer(prompt=prompt, history=chat_history,
                                                    #   streaming=streaming):

        answer_result_stream_result = self.llm_model_chain(
                {"prompt": prompt, "history": chat_history, "streaming": streaming})

        for answer_result in answer_result_stream_result['answer_result_stream']:
            resp = answer_result.llm_output["answer"]
            history = answer_result.history
            history[-1][0] = query
            response = {"query": query,
                        "result": resp,
                        "source_documents": related_docs_with_score}
            #print("OUTPUT response:", response)
            #print("OUTPUT history:", history)
            yield response, history

    # query      查询内容
    # vs_path    知识库路径
    # chunk_conent   是否启用上下文关联
    # score_threshold    搜索匹配score阈值
    # vector_search_top_k   搜索知识库内容条数，默认搜索5条结果
    # chunk_sizes    匹配单段内容的连接上下文长度
    def get_knowledge_based_conent_test(self, query, vs_path, chunk_conent,
                                        score_threshold=VECTOR_SEARCH_SCORE_THRESHOLD,
                                        vector_search_top_k=VECTOR_SEARCH_TOP_K, chunk_size=CHUNK_SIZE):
        vector_store = load_vector_store(vs_path, self.embeddings)
        # FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector
        vector_store.chunk_conent = chunk_conent
        vector_store.score_threshold = score_threshold
        vector_store.chunk_size = chunk_size
        related_docs_with_score = vector_store.similarity_search_with_score(query, k=vector_search_top_k)
        if not related_docs_with_score:
            response = {"query": query,
                        "source_documents": []}
            return response, ""
        torch_gc()
        prompt = "\n".join([doc.page_content for doc in related_docs_with_score])
        response = {"query": query,
                    "source_documents": related_docs_with_score}
        return response, prompt

    def get_search_result_based_answer(self, query, chat_history=[], streaming: bool = STREAMING):
        results = bing_search(query)
        result_docs = search_result2docs(results)
        prompt = generate_prompt(result_docs, query)

        answer_result_stream_result = self.llm_model_chain(
            {"prompt": prompt, "history": chat_history, "streaming": streaming})

        for answer_result in answer_result_stream_result['answer_result_stream']:
            resp = answer_result.llm_output["answer"]
            history = answer_result.history
            history[-1][0] = query
            response = {"query": query,
                        "result": resp,
                        "source_documents": result_docs}
            yield response, history

    def delete_file_from_vector_store(self,
                                      filepath: str or List[str],
                                      vs_path):
        vector_store = load_vector_store(vs_path, self.embeddings)
        status = vector_store.delete_doc(filepath)
        return status

    def update_file_from_vector_store(self,
                                      filepath: str or List[str],
                                      vs_path,
                                      docs: List[Document], ):
        vector_store = load_vector_store(vs_path, self.embeddings)
        status = vector_store.update_doc(filepath, docs)
        return status

    def list_file_from_vector_store(self,
                                    vs_path,
                                    fullpath=False):
        vector_store = load_vector_store(vs_path, self.embeddings)
        docs = vector_store.list_docs()
        if fullpath:
            return docs
        else:
            return [os.path.split(doc)[-1] for doc in docs]


if __name__ == "__main__":
    # 初始化消息
    args = None
    args = parser.parse_args(args=['--model-dir', '/media/checkpoint/', '--model', 'chatglm-6b', '--no-remote-model'])

    args_dict = vars(args)
    shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
    llm_model_ins = shared.loaderLLM()

    local_doc_qa = LocalDocQA()
    local_doc_qa.init_cfg(llm_model=llm_model_ins)
    query = "本项目使用的embedding模型是什么，消耗多少显存"
    vs_path = "/media/gpt4-pdf-chatbot-langchain/dev-langchain-ChatGLM/vector_store/test"
    last_print_len = 0
    # for resp, history in local_doc_qa.get_knowledge_based_answer(query=query,
    #                                                              vs_path=vs_path,
    #                                                              chat_history=[],
    #                                                              streaming=True):
    for resp, history in local_doc_qa.get_search_result_based_answer(query=query,
                                                                     chat_history=[],
                                                                     streaming=True):
        print(resp["result"][last_print_len:], end="", flush=True)
        last_print_len = len(resp["result"])
    source_text = [f"""出处 [{inum + 1}] {doc.metadata['source'] if doc.metadata['source'].startswith("http")
    else os.path.split(doc.metadata['source'])[-1]}：\n\n{doc.page_content}\n\n"""
                   # f"""相关度：{doc.metadata['score']}\n\n"""
                   for inum, doc in
                   enumerate(resp["source_documents"])]
    logger.info("\n" + "\n".join(source_text))
    pass
