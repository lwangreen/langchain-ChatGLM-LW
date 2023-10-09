# Filename: llama2_llm.py
# Author: wangtz
# Created Date: 2023-09-25
# support https://github.com/FlagAlpha/Llama2-Chinese

from abc import ABC
from langchain.chains.base import Chain
from typing import Any, Dict, List, Optional, Generator, Union
from langchain.callbacks.manager import CallbackManagerForChainRun
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList
from models.loader import LoaderCheckPoint
from models.base import (BaseAnswer,
                         AnswerResult,
                         AnswerResultStream,
                         AnswerResultQueueSentinelTokenListenerQueue)
import torch
import transformers


class LLama2LLMChain(BaseAnswer, Chain, ABC):
    checkPoint: LoaderCheckPoint = None
    # history = []
    history_len: int = 3
    max_new_tokens: int = 2048
    num_beams: int = 1
    temperature: float = 0.3
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.3
    logits_processor: LogitsProcessorList = None
    stopping_criteria: Optional[StoppingCriteriaList] = None
    streaming_key: str = "streaming"  #: :meta private:
    history_key: str = "history"  #: :meta private:
    prompt_key: str = "prompt"  #: :meta private:
    output_key: str = "answer_result_stream"  #: :meta private:

    def __init__(self, checkPoint: LoaderCheckPoint = None):
        super().__init__()
        self.checkPoint = checkPoint

    @property
    def _chain_type(self) -> str:
        return "LLaMA2LLMChain"

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return [self.prompt_key]

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    @property
    def _check_point(self) -> LoaderCheckPoint:
        return self.checkPoint


    # 将历史对话数组转换为文本格式
    def history_to_text(self, query, history):
        """
        历史对话软提示
            这段代码首先定义了一个名为 history_to_text 的函数，用于将 self.history
            数组转换为所需的文本格式。然后，我们将格式化后的历史文本
            再用 self.encode 将其转换为向量表示。最后，将历史对话向量与当前输入的对话向量拼接在一起。
        """
        formatted_history = ''
        history = history[-self.history_len:] if self.history_len > 0 else []
        if len(history) > 0:
            for i, (old_query, response) in enumerate(history):
                formatted_history += "<s>Human: {}\n</s><s>Assistant: {}\n".format(old_query, response)
        formatted_history += "<s>Human: {}\n</s><s>Assistant: ".format(query)
        return formatted_history

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Generator]:
        generator = self.generatorAnswer(inputs=inputs, run_manager=run_manager)
        return {self.output_key: generator}

    def _generate_answer(self,
                         inputs: Dict[str, Any],
                         run_manager: Optional[CallbackManagerForChainRun] = None,
                         generate_with_callback: AnswerResultStream = None) -> None:

        history = inputs[self.history_key]
        streaming = inputs[self.streaming_key]
        prompt = inputs[self.prompt_key]

        print(f"prompt: {prompt}")

        # Create the StoppingCriteriaList with the stopping strings
        self.stopping_criteria = transformers.StoppingCriteriaList()
        # 定义模型stopping_criteria 队列，在每次响应时将 torch.LongTensor, torch.FloatTensor同步到AnswerResult
        listenerQueue = AnswerResultQueueSentinelTokenListenerQueue()
        self.stopping_criteria.append(listenerQueue)
        
        soft_prompt = self.history_to_text(query=prompt, history=history)

        print(f"input: {soft_prompt}")
        if self.logits_processor is None:
            self.logits_processor = LogitsProcessorList()

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "num_beams": self.num_beams,
            "top_p": self.top_p,
            "do_sample": True,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "temperature": self.temperature,
            "eos_token_id": self.checkPoint.tokenizer.eos_token_id,
            "bos_token_id": self.checkPoint.tokenizer.bos_token_id,
            "pad_token_id": self.checkPoint.tokenizer.pad_token_id,
            "logits_processor": self.logits_processor,
            "stopping_criteria": self.stopping_criteria,
        }

        input_ids = self.checkPoint.tokenizer(
            [soft_prompt], 
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.to(self.checkPoint.llm_device)
        print(f"input_ids: {input_ids}")

        gen_kwargs.update({"input_ids": input_ids})
        output_ids = self.checkPoint.model.generate(**gen_kwargs)
        print(f"output_ids: {output_ids}")

        new_token_len = len(output_ids[0]) - len(input_ids[0])
        reply = self.checkPoint.tokenizer.decode(output_ids[0][-new_token_len:])
        print(f"reply: {reply}")

        answer_result = AnswerResult()
        history += [[prompt, reply]]
        answer_result.history = history
        answer_result.llm_output = {"answer": reply}
        generate_with_callback(answer_result)
