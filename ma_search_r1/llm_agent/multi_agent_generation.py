"""
Multi-Agent Generation Manager for MA-Search-R1.

This module implements a multi-agent system with three roles:
- Planner: Plans search strategy and generates search queries
- Researcher: Executes searches and retrieves information
- Synthesizer: Synthesizes information and generates final answers
"""

import torch
import re
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from search_r1.llm_agent.tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
import requests


class AgentRole(Enum):
    """Agent roles in the multi-agent system."""
    PLANNER = "planner"
    RESEARCHER = "researcher"
    SYNTHESIZER = "synthesizer"


@dataclass
class MultiAgentConfig:
    """Configuration for multi-agent generation."""
    max_turns: int  # Maximum number of interaction turns
    max_start_length: int
    max_prompt_length: int
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    search_url: str = None
    topk: int = 3
    
    # Agent-specific configs
    planner_model_path: str = None  # If None, uses shared model
    researcher_model_path: str = None
    synthesizer_model_path: str = None
    
    # Use shared model for all agents if True
    use_shared_model: bool = True


class MultiAgentGenerationManager:
    """
    Multi-agent generation manager that coordinates three agents:
    1. Planner: Plans search queries
    2. Researcher: Executes searches
    3. Synthesizer: Synthesizes and generates final answers
    """
    
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg_planner,  # Planner worker group
        actor_rollout_wg_researcher,  # Researcher worker group (can be same as planner)
        actor_rollout_wg_synthesizer,  # Synthesizer worker group (can be same as planner)
        config: MultiAgentConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg_planner = actor_rollout_wg_planner
        self.actor_rollout_wg_researcher = actor_rollout_wg_researcher
        self.actor_rollout_wg_synthesizer = actor_rollout_wg_synthesizer
        self.config = config
        self.is_validation = is_validation
        
        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))
    
    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses,
            add_special_tokens=False,
            return_tensors='pt',
            padding="longest"
        )['input_ids']
    
    def _add_planner_instruction(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Add Planner role instruction to the prompt."""
        # Decode the original prompt
        original_prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        
        # Extract question from original prompt (assuming it's at the end)
        # The original prompt format is: "Answer the given question. ... Question: {question}\n"
        planner_prompts = []
        for prompt in original_prompts:
            # Extract question - handle both chat template format and plain text
            # Try to find "Question:" in the prompt
            question_match = re.search(r'Question:\s*(.+?)(?:\n|$)', prompt, re.DOTALL)
            if question_match:
                question = question_match.group(1).strip()
            else:
                # Fallback: try to extract from chat template format
                # Chat templates might have special tokens, try to find the actual question text
                # For Qwen models, it might be after "user" or similar markers
                question_match = re.search(r'(?:user|User|question|Question)[^:]*:\s*(.+?)(?:\n|$)', prompt, re.DOTALL | re.IGNORECASE)
                if question_match:
                    question = question_match.group(1).strip()
                else:
                    # Last fallback: use the whole prompt
                    question = prompt.strip()
            
            # Construct Planner prompt
            planner_prompt = f"""You are a Planner agent. Analyze the question and create a search plan.
Generate search queries that will help answer the question. Output your plan in the format:
<plan>query1, query2, query3</plan>

Question: {question}
"""
            planner_prompts.append(planner_prompt)
        
        # Apply chat template if available, otherwise use plain text
        if self.tokenizer.chat_template:
            formatted_prompts = []
            for prompt in planner_prompts:
                formatted = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], 
                    add_generation_prompt=True, 
                    tokenize=False
                )
                formatted_prompts.append(formatted)
            planner_prompts = formatted_prompts
        
        # Tokenize the new prompts
        planner_inputs = self.tokenizer(
            planner_prompts,
            add_special_tokens=not self.tokenizer.chat_template,  # Chat template already adds special tokens
            return_tensors='pt',
            padding="longest"
        )
        
        # Truncate if too long
        max_len = min(self.config.max_prompt_length, planner_inputs['input_ids'].shape[1])
        return planner_inputs['input_ids'][:, -max_len:]
    
    def _add_synthesizer_instruction(self, input_ids: torch.Tensor, information: List[str] = None) -> torch.Tensor:
        """Add Synthesizer role instruction to the prompt."""
        # Decode the original prompt
        original_prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        
        synthesizer_prompts = []
        for i, prompt in enumerate(original_prompts):
            # Extract question - handle both chat template format and plain text
            question_match = re.search(r'Question:\s*(.+?)(?:\n|$)', prompt, re.DOTALL)
            if question_match:
                question = question_match.group(1).strip()
            else:
                # Fallback: try to extract from chat template format
                question_match = re.search(r'(?:user|User|question|Question)[^:]*:\s*(.+?)(?:\n|$)', prompt, re.DOTALL | re.IGNORECASE)
                if question_match:
                    question = question_match.group(1).strip()
                else:
                    question = prompt.strip()
            
            # Add information if provided
            info_text = ""
            if information and i < len(information) and information[i]:
                info_text = f"\nInformation: {information[i]}\n"
            
            # Construct Synthesizer prompt
            synthesizer_prompt = f"""You are a Synthesizer agent. Based on the information provided, either:
1. Provide a final answer: <answer>your answer</answer>
2. Request more information: <need_more>explain what information is still needed</need_more>

Question: {question}{info_text}

Your response:"""
            synthesizer_prompts.append(synthesizer_prompt)
        
        # Apply chat template if available, otherwise use plain text
        if self.tokenizer.chat_template:
            formatted_prompts = []
            for prompt in synthesizer_prompts:
                formatted = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], 
                    add_generation_prompt=True, 
                    tokenize=False
                )
                formatted_prompts.append(formatted)
            synthesizer_prompts = formatted_prompts
        
        # Tokenize the new prompts
        synthesizer_inputs = self.tokenizer(
            synthesizer_prompts,
            add_special_tokens=not self.tokenizer.chat_template,  # Chat template already adds special tokens
            return_tensors='pt',
            padding="longest"
        )
        
        # Truncate if too long
        max_len = min(self.config.max_prompt_length, synthesizer_inputs['input_ids'].shape[1])
        return synthesizer_inputs['input_ids'][:, -max_len:]
    
    def _postprocess_planner_response(self, responses: List[str]) -> Tuple[List[str], List[str]]:
        """
        Process planner responses to extract search queries.
        Planner should output: <plan>query1, query2, ...</plan>
        """
        queries_list = []
        processed_responses = []
        
        for resp in responses:
            # Extract queries from <plan> tags
            pattern = r'<plan>(.*?)</plan>'
            matches = re.findall(pattern, resp, re.DOTALL)
            
            if matches:
                # Split queries by comma or newline
                queries = [q.strip() for q in re.split(r'[,;\n]', matches[-1]) if q.strip()]
                queries_list.append(queries)
                processed_responses.append(f"<plan>{', '.join(queries)}</plan>")
            else:
                queries_list.append([])
                processed_responses.append(resp)
        
        return processed_responses, queries_list
    
    def _postprocess_researcher_response(self, responses: List[str]) -> List[str]:
        """
        Process researcher responses.
        Researcher should output: <search>query</search> or <done>message</done>
        """
        processed = []
        for resp in responses:
            # Keep the response as is, just ensure it's properly formatted
            if '<search>' in resp or '<done>' in resp:
                processed.append(resp)
            else:
                processed.append(resp)
        return processed
    
    def _postprocess_synthesizer_response(self, responses: List[str]) -> Tuple[List[str], List[bool]]:
        """
        Process synthesizer responses.
        Synthesizer should output: <answer>answer</answer> or <need_more>reason</need_more>
        """
        processed = []
        is_final = []
        
        for resp in responses:
            if '<answer>' in resp:
                # Extract answer
                pattern = r'<answer>(.*?)</answer>'
                match = re.search(pattern, resp, re.DOTALL)
                if match:
                    processed.append(f"<answer>{match.group(1).strip()}</answer>")
                    is_final.append(True)
                else:
                    processed.append(resp)
                    is_final.append(False)
            elif '<need_more>' in resp:
                processed.append(resp)
                is_final.append(False)
            else:
                processed.append(resp)
                is_final.append(False)
        
        return processed, is_final
    
    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        next_obs_ids = self.tokenizer(
            next_obs,
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,
        )['input_ids']
        
        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG: {next_obs_ids.shape[1]} > {self.config.max_obs_length}")
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]
        
        return next_obs_ids
    
    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor,
                              next_obs_ids: torch.Tensor) -> DataProto:
        """Update rolling state with new responses and observations."""
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)
        
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings
    
    def _generate_with_gpu_padding(self, active_batch: DataProto, worker_group) -> DataProto:
        """Wrapper for generation that handles multi-GPU padding requirements."""
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return worker_group.generate_sequences(active_batch)
        
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        
        if remainder == 0:
            return worker_group.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)
        
        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()
        
        padded_output = worker_group.generate_sequences(padded_active_batch)
        
        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
        
        padded_output.batch = trimmed_batch
        return padded_output
    
    def batch_search(self, queries: List[str] = None) -> List[str]:
        """Batchified search for queries."""
        if not queries:
            return []
        
        results = self._batch_search(queries)['result']
        return [self._passages2string(result) for result in results]
    
    def _batch_search(self, queries):
        """Call search API."""
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        return requests.post(self.config.search_url, json=payload).json()
    
    def _passages2string(self, retrieval_result):
        """Format retrieval results as string."""
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference
    
    def run_multi_agent_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """
        Run multi-agent generation loop.
        
        Flow:
        1. Planner generates search plan
        2. Researcher executes searches
        3. Synthesizer synthesizes and decides: answer or need more info
        4. If need more, go back to Planner
        """
        # Store the original prompt for extracting questions
        original_prompt_ids = initial_input_ids[:, -self.config.max_start_length:]
        
        original_left_side = {'input_ids': original_prompt_ids}
        original_right_side = {
            'responses': initial_input_ids[:, []],
            'responses_with_info_mask': initial_input_ids[:, []]
        }
        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        
        rollings = gen_batch
        
        # Track agent outputs for each example
        agent_outputs = {
            'planner': [],
            'researcher': [],
            'synthesizer': []
        }
        
        # Main multi-agent interaction loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # ========== Step 1: Planner generates search plan ==========
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            
            # Add Planner instruction for first turn, or use original prompt for subsequent turns
            if step == 0:
                # First turn: add Planner instruction
                planner_input_ids = self._add_planner_instruction(rollings_active.batch['input_ids'])
                planner_attention_mask = (planner_input_ids != self.tokenizer.pad_token_id).long()
                planner_position_ids = self.tensor_fn.create_position_ids(planner_attention_mask)
                
                planner_rollings = DataProto.from_dict({
                    'input_ids': planner_input_ids,
                    'attention_mask': planner_attention_mask,
                    'position_ids': planner_position_ids
                })
                planner_rollings.meta_info = rollings_active.meta_info
            else:
                # Subsequent turns: use rollings as is (already contains conversation history)
                planner_rollings = rollings_active
            
            planner_output = self._generate_with_gpu_padding(
                planner_rollings, self.actor_rollout_wg_planner
            )
            
            planner_responses_raw = self.tokenizer.batch_decode(
                planner_output.batch['responses'], skip_special_tokens=True
            )
            planner_responses_str, queries_list = self._postprocess_planner_response(planner_responses_raw)
            planner_responses_ids = self._batch_tokenize(planner_responses_str)
            planner_responses_ids, planner_responses_str = self.tensor_fn._example_level_pad(
                planner_responses_ids, planner_responses_str, active_mask
            )
            
            # ========== Step 2: Researcher executes searches ==========
            search_results_list = []
            for queries in queries_list:
                if queries:
                    search_results = self.batch_search(queries)
                    # Combine all search results
                    combined_results = "\n\n".join(search_results)
                    search_results_list.append(combined_results)
                else:
                    search_results_list.append("")
            
            # Format search results as observation
            next_obs = []
            for i, (active, results) in enumerate(zip(active_mask, search_results_list)):
                if active and results:
                    next_obs.append(f'\n\n<information>{results.strip()}</information>\n\n')
                else:
                    next_obs.append('')
            
            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update rollings with planner output and search results
            # For first turn, use the modified planner prompt; for subsequent turns, use rollings as is
            if step == 0:
                # Use the modified planner prompt as the base for rollings
                # We need to expand planner_rollings to full batch size
                batch_size = gen_batch.batch['input_ids'].shape[0]
                full_planner_input_ids = torch.zeros((batch_size, planner_rollings.batch['input_ids'].shape[1]), 
                                                     dtype=planner_rollings.batch['input_ids'].dtype,
                                                     device=planner_rollings.batch['input_ids'].device)
                full_planner_attention_mask = torch.zeros((batch_size, planner_rollings.batch['attention_mask'].shape[1]),
                                                          dtype=planner_rollings.batch['attention_mask'].dtype,
                                                          device=planner_rollings.batch['attention_mask'].device)
                full_planner_position_ids = torch.zeros((batch_size, planner_rollings.batch['position_ids'].shape[1]),
                                                        dtype=planner_rollings.batch['position_ids'].dtype,
                                                        device=planner_rollings.batch['position_ids'].device)
                
                # Fill active examples with modified prompt
                active_indices = torch.where(active_mask)[0]
                for idx, active_idx in enumerate(active_indices):
                    full_planner_input_ids[active_idx] = planner_rollings.batch['input_ids'][idx]
                    full_planner_attention_mask[active_idx] = planner_rollings.batch['attention_mask'][idx]
                    full_planner_position_ids[active_idx] = planner_rollings.batch['position_ids'][idx]
                
                # Fill non-active examples with original prompt
                non_active_indices = torch.where(~active_mask)[0]
                for idx in non_active_indices:
                    full_planner_input_ids[idx] = rollings.batch['input_ids'][idx]
                    full_planner_attention_mask[idx] = rollings.batch['attention_mask'][idx]
                    full_planner_position_ids[idx] = rollings.batch['position_ids'][idx]
                
                planner_rollings_full = DataProto.from_dict({
                    'input_ids': full_planner_input_ids,
                    'attention_mask': full_planner_attention_mask,
                    'position_ids': full_planner_position_ids
                })
                planner_rollings_full.meta_info = rollings.meta_info
                
                rollings = self._update_rolling_state(
                    planner_rollings_full,
                    planner_responses_ids,
                    next_obs_ids
                )
            else:
                rollings = self._update_rolling_state(
                    rollings,
                    planner_responses_ids,
                    next_obs_ids
                )
            
            # ========== Step 3: Synthesizer synthesizes and decides ==========
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            
            # Add Synthesizer instruction
            # For first turn, we need to append Synthesizer instruction to the current rollings
            # which already contains: Planner prompt + Planner output + info
            if step == 0:
                # Extract question from original prompt
                original_active_prompt_ids = original_prompt_ids[active_mask]
                original_prompt_texts = self.tokenizer.batch_decode(original_active_prompt_ids, skip_special_tokens=True)
                
                # Decode current rollings to get conversation history
                rollings_texts = self.tokenizer.batch_decode(rollings_active.batch['input_ids'], skip_special_tokens=True)
                
                # Construct Synthesizer prompts by appending instruction to rollings
                synthesizer_prompts = []
                for i, (rolling_text, original_text) in enumerate(zip(rollings_texts, original_prompt_texts)):
                    # Extract question from original prompt
                    question_match = re.search(r'Question:\s*(.+?)(?:\n|$)', original_text, re.DOTALL)
                    if question_match:
                        question = question_match.group(1).strip()
                    else:
                        question_match = re.search(r'(?:user|User|question|Question)[^:]*:\s*(.+?)(?:\n|$)', original_text, re.DOTALL | re.IGNORECASE)
                        if question_match:
                            question = question_match.group(1).strip()
                        else:
                            question = original_text.strip()
                    
                    # Append Synthesizer instruction to the conversation history
                    synthesizer_instruction = f"""

You are a Synthesizer agent. Based on the information provided, either:
1. Provide a final answer: <answer>your answer</answer>
2. Request more information: <need_more>explain what information is still needed</need_more>

Question: {question}

Your response:"""
                    combined_prompt = rolling_text + synthesizer_instruction
                    synthesizer_prompts.append(combined_prompt)
                
                # Apply chat template if available
                if self.tokenizer.chat_template:
                    formatted_prompts = []
                    for prompt in synthesizer_prompts:
                        formatted = self.tokenizer.apply_chat_template(
                            [{"role": "user", "content": prompt}], 
                            add_generation_prompt=True, 
                            tokenize=False
                        )
                        formatted_prompts.append(formatted)
                    synthesizer_prompts = formatted_prompts
                
                # Tokenize
                synthesizer_inputs = self.tokenizer(
                    synthesizer_prompts,
                    add_special_tokens=not self.tokenizer.chat_template,
                    return_tensors='pt',
                    padding="longest"
                )
                synthesizer_input_ids = synthesizer_inputs['input_ids']
                max_len = min(self.config.max_prompt_length, synthesizer_input_ids.shape[1])
                synthesizer_input_ids = synthesizer_input_ids[:, -max_len:]
            else:
                # Subsequent turns: use rollings as is (already contains conversation history)
                synthesizer_input_ids = rollings_active.batch['input_ids']
            
            synthesizer_attention_mask = (synthesizer_input_ids != self.tokenizer.pad_token_id).long()
            synthesizer_position_ids = self.tensor_fn.create_position_ids(synthesizer_attention_mask)
            
            synthesizer_rollings = DataProto.from_dict({
                'input_ids': synthesizer_input_ids,
                'attention_mask': synthesizer_attention_mask,
                'position_ids': synthesizer_position_ids
            })
            synthesizer_rollings.meta_info = rollings_active.meta_info
            
            synthesizer_output = self._generate_with_gpu_padding(
                synthesizer_rollings, self.actor_rollout_wg_synthesizer
            )
            
            synthesizer_responses_raw = self.tokenizer.batch_decode(
                synthesizer_output.batch['responses'], skip_special_tokens=True
            )
            synthesizer_responses_str, is_final = self._postprocess_synthesizer_response(
                synthesizer_responses_raw
            )
            synthesizer_responses_ids = self._batch_tokenize(synthesizer_responses_str)
            synthesizer_responses_ids, synthesizer_responses_str = self.tensor_fn._example_level_pad(
                synthesizer_responses_ids, synthesizer_responses_str, active_mask
            )
            
            # Check if synthesizer wants more information
            need_more = []
            for resp in synthesizer_responses_str:
                need_more.append('<need_more>' in resp and '<answer>' not in resp)
            
            # Update dones based on final answers
            dones = []
            valid_action = []
            is_search = []
            
            for i, (active, final, more) in enumerate(zip(active_mask, is_final, need_more)):
                if not active:
                    dones.append(1)
                    valid_action.append(0)
                    is_search.append(0)
                elif final:
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                elif more:
                    # Need more info, continue
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(0)
                else:
                    # Invalid response
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            
            # Update right side with all agent outputs
            # Combine planner and synthesizer responses
            combined_responses = []
            combined_responses_with_mask = []
            
            for i in range(len(planner_responses_str)):
                if active_mask[i] if i < len(active_mask) else False:
                    # Combine planner plan + synthesizer response
                    combined = planner_responses_str[i] + "\n" + synthesizer_responses_str[i]
                    combined_responses.append(combined)
                else:
                    combined_responses.append("")
            
            combined_responses_ids = self._batch_tokenize(combined_responses)
            combined_responses_ids, _ = self.tensor_fn._example_level_pad(
                combined_responses_ids, combined_responses, active_mask
            )
            
            original_right_side = self._update_right_side(
                original_right_side,
                combined_responses_ids,
                next_obs_ids if any(active_mask) else None
            )
            
            # Update rollings to include Synthesizer prompt for logging
            # This ensures the final output contains the full conversation history with multi-agent prompts
            if step == 0:
                # Expand synthesizer_rollings to full batch size
                batch_size = gen_batch.batch['input_ids'].shape[0]
                synth_len = synthesizer_rollings.batch['input_ids'].shape[1]
                full_synthesizer_input_ids = torch.zeros((batch_size, synth_len), 
                                                         dtype=synthesizer_rollings.batch['input_ids'].dtype,
                                                         device=synthesizer_rollings.batch['input_ids'].device)
                full_synthesizer_attention_mask = torch.zeros((batch_size, synth_len),
                                                              dtype=synthesizer_rollings.batch['attention_mask'].dtype,
                                                              device=synthesizer_rollings.batch['attention_mask'].device)
                full_synthesizer_position_ids = torch.zeros((batch_size, synth_len),
                                                            dtype=synthesizer_rollings.batch['position_ids'].dtype,
                                                            device=synthesizer_rollings.batch['position_ids'].device)
                
                # Fill active examples with Synthesizer prompt (which includes full conversation history)
                active_indices = torch.where(active_mask)[0]
                for idx, active_idx in enumerate(active_indices):
                    full_synthesizer_input_ids[active_idx] = synthesizer_rollings.batch['input_ids'][idx]
                    full_synthesizer_attention_mask[active_idx] = synthesizer_rollings.batch['attention_mask'][idx]
                    full_synthesizer_position_ids[active_idx] = synthesizer_rollings.batch['position_ids'][idx]
                
                # Fill non-active examples: pad current rollings to match synthesizer prompt length
                non_active_indices = torch.where(~active_mask)[0]
                for idx in non_active_indices:
                    current_ids = rollings.batch['input_ids'][idx]
                    current_mask = rollings.batch['attention_mask'][idx]
                    current_pos = rollings.batch['position_ids'][idx]
                    current_len = current_ids.shape[0]
                    
                    if current_len < synth_len:
                        # Pad with pad_token_id
                        pad_len = synth_len - current_len
                        pad_tokens = torch.full((pad_len,), self.tokenizer.pad_token_id, 
                                               dtype=current_ids.dtype, device=current_ids.device)
                        full_synthesizer_input_ids[idx] = torch.cat([current_ids, pad_tokens])
                        full_synthesizer_attention_mask[idx] = torch.cat([current_mask, 
                                                                         torch.zeros(pad_len, dtype=current_mask.dtype, 
                                                                                    device=current_mask.device)])
                        full_synthesizer_position_ids[idx] = torch.cat([current_pos,
                                                                       torch.zeros(pad_len, dtype=current_pos.dtype,
                                                                                  device=current_pos.device)])
                    else:
                        # Truncate to match length
                        full_synthesizer_input_ids[idx] = current_ids[:synth_len]
                        full_synthesizer_attention_mask[idx] = current_mask[:synth_len]
                        full_synthesizer_position_ids[idx] = current_pos[:synth_len]
                
                # Update rollings to include Synthesizer prompt (which contains full conversation history)
                rollings.batch['input_ids'] = full_synthesizer_input_ids
                rollings.batch['attention_mask'] = full_synthesizer_attention_mask
                rollings.batch['position_ids'] = full_synthesizer_position_ids
            
            # If all done or no more info needed, break
            if not active_mask.sum() or not any(need_more):
                break
        
        # Final synthesizer generation if still active
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            
            final_output = self._generate_with_gpu_padding(
                rollings_active, self.actor_rollout_wg_synthesizer
            )
            
            final_responses_str = self.tokenizer.batch_decode(
                final_output.batch['responses'], skip_special_tokens=True
            )
            final_responses_str, is_final = self._postprocess_synthesizer_response(final_responses_str)
            final_responses_ids = self._batch_tokenize(final_responses_str)
            final_responses_ids, final_responses_str = self.tensor_fn._example_level_pad(
                final_responses_ids, final_responses_str, active_mask
            )
            
            original_right_side = self._update_right_side(
                original_right_side,
                final_responses_ids,
            )
        
        meta_info = {
            'turns_stats': turns_stats.tolist(),
            'active_mask': active_mask.tolist(),
            'valid_action_stats': valid_action_stats.tolist(),
            'valid_search_stats': valid_search_stats.tolist(),
        }
        
        # Use rollings as the final prompt, which contains the full conversation history
        # including the modified Planner/Synthesizer prompts
        # The rollings.batch['input_ids'] contains: modified prompt + planner output + info + synthesizer output
        # We need to extract just the prompt part (before responses)
        # For multi-agent, the "prompt" is everything before the final synthesizer response
        
        # The rollings already contains the full conversation, so we use it as the prompt
        # and original_right_side contains the final responses
        final_left_side = {
            'input_ids': rollings.batch['input_ids']
        }
        
        return self._compose_final_output(final_left_side, original_right_side, meta_info)
    
    def _update_right_side(self, right_side: Dict,
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids is not None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                right_side['responses'],
                right_side['responses_with_info_mask'],
                cur_responses,
                next_obs_ids,
                pad_to_left=False
            )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                right_side['responses'],
                right_side['responses_with_info_mask'],
                cur_responses,
                pad_to_left=False
            )
        
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {
            'responses': responses[:, :max_len],
            'responses_with_info_mask': responses_with_info_mask[:, :max_len]
        }
    
    def _info_masked_concatenate_with_padding(self,
                prompt: torch.Tensor,
                prompt_with_mask: torch.Tensor,
                response: torch.Tensor,
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Concatenate tensors and handle padding with info mask."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device)
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)
        
        return padded_tensor, padded_tensor_with_info
    
    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

