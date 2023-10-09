# Filename: baichuan_llm.py
# Author: wangtz
# Created Date: 2023-09-23

from abc import ABC
from langchain.chains.base import Chain
from typing import Any, Dict, List, Optional, Generator
from langchain.callbacks.manager import CallbackManagerForChainRun
# from transformers.generation.logits_process import LogitsProcessor
# from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList
from models.loader import LoaderCheckPoint
from models.base import (BaseAnswer,
                         AnswerResult,
                         AnswerResultStream,
                         AnswerResultQueueSentinelTokenListenerQueue)
# import torch
import transformers
from transformers.generation.utils import GenerationConfig


class BaichuanLLMChain(BaseAnswer, Chain, ABC):
    checkPoint: LoaderCheckPoint = None
    # history = []
    history_len: int = 10
    streaming_key: str = "streaming"  #: :meta private:
    history_key: str = "history"  #: :meta private:
    prompt_key: str = "prompt"  #: :meta private:
    output_key: str = "answer_result_stream"  #: :meta private:

    def __init__(self, checkPoint: LoaderCheckPoint = None):
        super().__init__()
        self.checkPoint = checkPoint
        self.checkPoint.model.generation_config = GenerationConfig(
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            user_token_id=195,
            assistant_token_id=196,
            max_new_tokens=2048,
            temperature=0.3,
            top_k=5,
            top_p=0.85,
            repetition_penalty=1.05,
            do_sample=True,
        )

    @property
    def _chain_type(self) -> str:
        return "BaichuanLLMChain"

    @property
    def _check_point(self) -> LoaderCheckPoint:
        return self.checkPoint

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
        messages = []
        for x in history:
            messages.append({"role": "user", "content": x[0]})
            messages.append({"role": "assistant", "content": x[1]})
        messages.append({"role": "user", "content": prompt})
        print(f"input: {prompt}")
        # Create the StoppingCriteriaList with the stopping strings
        stopping_criteria_list = transformers.StoppingCriteriaList()
        # 定义模型stopping_criteria 队列，在每次响应时将 torch.LongTensor, torch.FloatTensor同步到AnswerResult
        listenerQueue = AnswerResultQueueSentinelTokenListenerQueue()
        stopping_criteria_list.append(listenerQueue)
        if streaming:
            history += [[]]
            # position = 0
            for resp in self.checkPoint.model.chat(
                    self.checkPoint.tokenizer,
                    messages,
                    stream=True
            ):
                # print(resp[position:], end='', flush=True)
                # position = len(resp)
                history[-1] = [prompt, resp]
                answer_result = AnswerResult()
                answer_result.history = history
                answer_result.llm_output = {"answer": resp}
                generate_with_callback(answer_result)
            self.checkPoint.clear_torch_cache()
        else:
            resp = self.checkPoint.model.chat(
                self.checkPoint.tokenizer,
                messages,
                streami=False
            )
            # print(f"resp: {resp}")
            self.checkPoint.clear_torch_cache()
            history += [[prompt, resp]]
            answer_result = AnswerResult()
            answer_result.history = history
            answer_result.llm_output = {"answer": resp}
            generate_with_callback(answer_result)

