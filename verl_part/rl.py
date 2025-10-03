# Previous Trial with TRL, and a empirical reward fuction awared of difficulty and length
import torch
import re
import numpy as np
import logging
import torch
import transformers
from datasets import load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from transformers import AutoConfig, PreTrainedModel, TrainerCallback
from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.grpo_trainer import DisGRPOTrainer
from trl.trainer.utils import disable_dropout_in_model
from accelerate.utils import is_peft_model
from trl.trainer.callbacks import SyncRefModelCallback
from trl.rewards import think_format_reward
from trl.models import prepare_fsdp, prepare_deepspeed
from deepspeed import DeepSpeedEngine
import os

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    ################
    # Model & Processor
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    ################
    # Dataset
    ################
    f_dataset = load_dataset("json", data_files="bigmath_2_rl.json")
    split_dataset = f_dataset['train'].train_test_split(test_size=0.05, seed=42, shuffle=True)
    train_dataset, eval_dataset = split_dataset["train"], split_dataset["test"]
    # llama8b_solve_rate
    SYSTEM_PROMPT = "Please reason step by step, start your response with <think>, and put your final answer within \\boxed{}.\n"

    def make_conversation(example):
        return {
            "prompt": [{"role": "user", "content": f"{SYSTEM_PROMPT}Question:{example['problem']}"}]
        }

    train_dataset = train_dataset.map(make_conversation)
    eval_dataset = eval_dataset.map(make_conversation)

    # train_dataset = train_dataset.remove_columns(["messages", "problem"])
    # eval_dataset = eval_dataset.remove_columns(["messages", "problem"])

    ################
    # Reward Function for Training
    ################

    PARSE_CONFIG = [
        LatexExtractionConfig(
            normalization_config=NormalizationConfig(
                nits=False,
                malformed_operators=False,
                basic_latex=True,
                boxed="all",
                units=True,
            ),
            boxed_match_priority=0,
            try_extract_without_anchor=False,
        )
    ]

    def _verify(completion: str, answer: str) -> bool:
        """
        Verifies if the model output is correct using the math_verify library.
        Inspired by the implementation in sft_set_filter.py.
        """
        if not isinstance(completion, str) or not isinstance(answer, str):
            return False
        try:
            # Parse the ground truth answer
            answer = f"${answer}$"
            gold_parsed = parse(answer, extraction_mode="first_match")
            if not gold_parsed:
                # If the ground truth cannot be parsed, fall back to string matching on the content of \boxed{}
                match = re.search(r"\boxed{(.*?)}", completion)
                if match:
                    extracted_completion = match.groups()[-1].strip()
                    print(f"answer:{extracted_completion.lower()} Vs GT: {answer.strip().lower()}")
                    return extracted_completion.lower() == answer.strip().lower()
                return completion.strip().lower() == answer.strip().lower()

            answer_parsed = parse(completion, extraction_config=PARSE_CONFIG, extraction_mode="first_match")
            if not answer_parsed:
                return False
            return verify(gold_parsed, answer_parsed)
        except Exception as e:
            logging.warning(f"Verification failed: {e}. Falling back to text matching. Model output: '{completion}', Ground truth: '{answer}'")
            # Fallback on any exception
            match = re.search(r"\boxed{(.*?)}", completion)
            if match:
                extracted_completion = match.groups()[-1].strip()
                return extracted_completion.lower() == answer.strip().lower()
            return completion.strip().lower() == answer.strip().lower()
    
    def _calculate_matching(n: int, d: float, n_min: int = 0, n_max: int = 200, gamma: float = 10.0) -> float:
        n = np.clip(n, n_min, n_max)
        d = np.clip(d, 0, 1)

        log_n = np.log(n + 1)
        log_n_min = np.log(n_min + 1)
        log_n_max = np.log(n_max + 1)

        n_norm = (log_n - log_n_min) / (log_n_max - log_n_min)

        return np.exp(-6 * (n_norm -1.3* d**2)**2)
    
    def _count_reflection_words(completion: str) -> int:
        reflection_words = ["but", "wait", "alternatively", "however", "check", "alternative", "double-check", "hmm"]
        pattern = r"\b(" + "|".join(re.escape(word) for word in reflection_words) + r")\b"
        matches = re.findall(pattern, completion, re.IGNORECASE)
        return len(matches)
    def reward_acc(completions, prompts, **kwargs) -> list[float]:
        rewards = []
        ground_truth = kwargs["answer"]

        for i, completion in enumerate(completions):
            answer = ground_truth[i]
            completion = completion[0]['content']
            reward = 0.0

            if _verify(completion, answer):
                reward += 1.0
            else:
                pass

            rewards.append(reward)
        return rewards
        
    def reward_diff(completions, prompts, **kwargs) -> list[float]:
        """Reward function that checks if the completion matches the ground truth.
          If both gold and prediction are parseable use math verification.
          If not parseable compare as normalized text.
        """

        rewards = []
        ns = []
        solve_rate = kwargs["llama8b_solve_rate"]
        ground_truth = kwargs["answer"]

        for i, completion in enumerate(completions):
            prompt_solve_rate = solve_rate[i]
            answer = ground_truth[i]
            completion = completion[0]['content']
            reward = 0.0

            if _verify(completion, answer):
                reward += 1.0
            else:
                pass
                
            # TODO: student model can conquer format if _right_format(completion):
            n = _count_reflection_words(completion)
            ns.append(n)
            matching_coefficient = _calculate_matching(n, d=1-prompt_solve_rate)
            if reward > 0:
                reward = reward * matching_coefficient

            rewards.append(reward)
        print(f"\nRewards:{rewards}")
        print(f"Reflective_words_count:{ns}")
        print(f"solve_rate:{solve_rate[0]}\n")
        return rewards

    ################
    # Training
    ################
    
    #trainer = DisGRPOTrainer(
    #    model=model_args.model_name_or_path,
    #    ref_model="/home/i/i0002066/model/dqw-7b", # !!!!!!SHIT-HARD_CODING!!!!!!
    #    args=training_args,
    #    reward_funcs=[reward_diff, think_format_reward],
    #    train_dataset=train_dataset,
    #    eval_dataset=eval_dataset,
    #    peft_config=get_peft_config(model_args),
    #)
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=[reward_acc, think_format_reward],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)