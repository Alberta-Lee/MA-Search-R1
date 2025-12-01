"""
Multi-agent rollout manager for CoSearch-R1.
Orchestrates Planner, Explorer, and Synthesizer agents in a collaborative search process.
"""

import torch
import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

from verl import DataProto
from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig
import requests
from search_r1.llm_agent.tensor_helper import TensorHelper, TensorConfig
from search_r1.multi_agent.protocol import (
    MultiAgentProtocol, AgentSegment, AgentRole
)
from search_r1.multi_agent.templates import MultiAgentTemplate


@dataclass
class MultiAgentTrajectory:
    """Stores trajectory information for multi-agent rollout."""
    segments: List[AgentSegment] = field(default_factory=list)
    information_blocks: List[Tuple[str, str]] = field(default_factory=list)  # (agent_name, content)
    full_dialog: List[str] = field(default_factory=list)
    agent_stats: Dict[str, Dict[str, Any]] = field(default_factory=lambda: defaultdict(dict))
    
    def add_segment(self, segment: AgentSegment):
        """Add an agent segment to trajectory."""
        self.segments.append(segment)
        role = segment.role
        
        # Update stats
        if role not in self.agent_stats:
            self.agent_stats[role] = {
                'num_segments': 0,
                'num_searches': 0,
                'has_answer': False
            }
        
        self.agent_stats[role]['num_segments'] += 1
        self.agent_stats[role]['num_searches'] += len(segment.search_queries)
        if segment.has_answer:
            self.agent_stats[role]['has_answer'] = True
    
    def add_information(self, agent_name: str, content: str):
        """Add an information block to trajectory."""
        self.information_blocks.append((agent_name, content))


class MultiAgentRolloutManager:
    """
    Manages multi-agent rollout process.
    Coordinates Planner, Explorer, and Synthesizer agents.
    """
    
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        multi_agent_config: Dict,
        is_validation: bool = False,
    ):
        """
        Initialize multi-agent rollout manager.
        
        Args:
            tokenizer: Tokenizer instance
            actor_rollout_wg: Actor rollout worker group
            config: Generation config (from single-agent)
            multi_agent_config: Multi-agent specific config
            is_validation: Whether in validation mode
        """
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.multi_agent_config = multi_agent_config
        self.is_validation = is_validation
        
        # Multi-agent specific settings
        self.num_explorers = multi_agent_config.get('num_explorers', 2)
        self.max_explore_rounds = multi_agent_config.get('max_explore_rounds', 2)
        self.roles = multi_agent_config.get('roles', ['Planner', 'Explorer', 'Synthesizer'])
        
        # Logging settings
        self.show_steps = multi_agent_config.get('show_steps', False)  # Whether to show intermediate steps
        self.log_sample_count = multi_agent_config.get('log_sample_count', 2)  # Number of samples to log
        self.log_step_freq = multi_agent_config.get('log_step_freq', 1)  # Log every N steps (0 = only first step)
        
        # Reuse tensor helper from single-agent
        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))
        
        # Search client (will be set by caller)
        self.search_client = None
        
        # Track step count for logging frequency
        self._step_count = 0
    
    def set_search_client(self, search_client):
        """Set the search client for retrieving information."""
        self.search_client = search_client
    
    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses,
            add_special_tokens=False,
            return_tensors='pt',
            padding="longest"
        )['input_ids']
    
    def _generate_agent_responses_batch(
        self,
        dialog_contexts: List[str],
        agent_name: str,
        stop_tokens: List[str],
        active_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[str]]:
        print(f"[CoSearch-R1] _generate_agent_responses_batch called for {agent_name} (active={active_mask.sum().item() if active_mask is not None else 'None'}/{len(dialog_contexts)})")
        """
        Generate responses for a batch of agents, each with independent dialog context.
        
        Args:
            dialog_contexts: List of dialog contexts (one per sample in batch)
            agent_name: Name of the agent to generate for
            stop_tokens: List of stop token sequences
            active_mask: Mask for active examples
            
        Returns:
            Tuple of (token_ids, response_strings)
        """
        batch_size = len(dialog_contexts)
        if active_mask is not None:
            batch_size = active_mask.shape[0]
        
        # Validate dialog_contexts
        if not dialog_contexts or len(dialog_contexts) == 0:
            print(f"[WARNING] dialog_contexts is empty, returning empty responses")
            empty_responses = torch.empty((batch_size, 0), dtype=torch.long)
            empty_responses_str = [""] * batch_size
            return empty_responses, empty_responses_str
        
        # Filter out empty dialogs and use fallback for empty ones
        valid_dialogs = []
        for i, dialog in enumerate(dialog_contexts):
            if not dialog or not dialog.strip():
                print(f"[WARNING] Empty dialog at index {i} in dialog_contexts, using fallback")
                valid_dialogs.append("Answer the question.")
            else:
                valid_dialogs.append(dialog)
        
        # Tokenize all dialog contexts
        input_ids_list = []
        attention_masks_list = []
        max_len = 0
        
        for i, dialog in enumerate(valid_dialogs):
            # Debug: print first dialog to help diagnose
            if i == 0:
                print(f"[DEBUG] First dialog (first 300 chars): {dialog[:300]}")
            
            # Clean dialog: remove excessive special tokens that might be treated as pad
            # Remove consecutive <|endoftext|> tokens
            import re
            cleaned_dialog = re.sub(r'<\|endoftext\|>+', '', dialog)
            # Remove leading/trailing whitespace
            cleaned_dialog = cleaned_dialog.strip()
            
            # If cleaned dialog is empty, use fallback
            if not cleaned_dialog:
                print(f"[WARNING] Dialog at index {i} cleaned to empty, using fallback")
                cleaned_dialog = "Answer the question."
            
            encoded = self.tokenizer.encode(
                cleaned_dialog,
                add_special_tokens=False,
                return_tensors='pt'
            )
            if encoded.shape[1] == 0:
                print(f"[ERROR] Dialog at index {i} tokenized to empty sequence")
                print(f"[ERROR] Original dialog (first 500 chars): {dialog[:500]}")
                print(f"[ERROR] Cleaned dialog: {cleaned_dialog[:500]}")
                # Use fallback
                encoded = self.tokenizer.encode("Answer the question.", add_special_tokens=False, return_tensors='pt')
            
            # Check if encoded contains only pad tokens
            pad_token_id = self.tokenizer.pad_token_id
            eos_token_id = self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else None
            
            # Check for pad tokens (and eos tokens if they're treated as pad)
            non_pad_mask = (encoded[0] != pad_token_id)
            if eos_token_id is not None and eos_token_id != pad_token_id:
                non_pad_mask = non_pad_mask & (encoded[0] != eos_token_id)
            
            if not non_pad_mask.any():
                print(f"[WARNING] Dialog at index {i} tokenized to all pad/eos tokens, using fallback")
                print(f"[WARNING] Original dialog (first 500 chars): {dialog[:500]}")
                print(f"[WARNING] Cleaned dialog: {cleaned_dialog[:500]}")
                # Use fallback
                encoded = self.tokenizer.encode("Answer the question.", add_special_tokens=False, return_tensors='pt')
            
            input_ids_list.append(encoded[0])
            max_len = max(max_len, encoded.shape[1])
        
        # Pad all to same length
        pad_token_id = self.tokenizer.pad_token_id
        padded_input_ids = []
        padded_attention_masks = []
        
        for input_ids in input_ids_list:
            pad_len = max_len - input_ids.shape[0]
            if pad_len > 0:
                padding = torch.full((pad_len,), pad_token_id, dtype=input_ids.dtype)
                padded = torch.cat([input_ids, padding], dim=0)
            else:
                padded = input_ids[:max_len]
            padded_input_ids.append(padded)
            padded_attention_masks.append((padded != pad_token_id).long())
        
        # Stack into batch tensor
        input_ids_batch = torch.stack(padded_input_ids, dim=0)
        attention_mask_batch = torch.stack(padded_attention_masks, dim=0)
        
        # Filter active samples before generation (to avoid vLLM error with all-pad sequences)
        if active_mask is not None and not active_mask.all():
            # Only generate for active samples
            active_indices = active_mask.nonzero(as_tuple=True)[0]
            if len(active_indices) == 0:
                # No active samples, return empty responses
                empty_responses = torch.empty((batch_size, 0), dtype=torch.long)
                empty_responses_str = [""] * batch_size
                return empty_responses, empty_responses_str
            
            input_ids_active = input_ids_batch[active_indices]
            attention_mask_active = attention_mask_batch[active_indices]
            dialog_contexts_active = [dialog_contexts[i] for i in active_indices.tolist()]
        else:
            # All samples are active
            active_indices = None
            input_ids_active = input_ids_batch
            attention_mask_active = attention_mask_batch
            dialog_contexts_active = dialog_contexts
        
        # Verify that we have at least one non-empty sequence
        pad_token_id = self.tokenizer.pad_token_id
        has_valid_sequence = False
        valid_sequence_indices = []
        
        for i in range(input_ids_active.shape[0]):
            if (input_ids_active[i] != pad_token_id).any():
                has_valid_sequence = True
                valid_sequence_indices.append(i)
        
        if not has_valid_sequence:
            # All sequences are empty (all pad), this shouldn't happen if dialog_contexts are valid
            # Debug: print dialog_contexts to help diagnose
            print(f"[ERROR] All input sequences are empty (all pad tokens).")
            print(f"[ERROR] dialog_contexts length: {len(dialog_contexts)}")
            print(f"[ERROR] First few dialog_contexts (first 200 chars each):")
            for idx, d in enumerate(dialog_contexts[:min(3, len(dialog_contexts))]):
                print(f"  [{idx}]: {d[:200] if d else '(empty)'}")
            print(f"[ERROR] input_ids_active shape: {input_ids_active.shape}")
            if attention_mask_active.shape[0] > 0:
                print(f"[ERROR] attention_mask_active sum: {attention_mask_active.sum(dim=1)}")
            
            # Return empty responses gracefully instead of raising error
            if active_indices is not None:
                empty_responses = torch.empty((batch_size, 0), dtype=torch.long)
                empty_responses_str = [""] * batch_size
                return empty_responses, empty_responses_str
            else:
                # Last resort: create a minimal valid sequence
                print(f"[WARNING] Creating minimal valid sequence as fallback")
                try:
                    min_valid_text = "Answer the question."
                    min_valid_ids = self.tokenizer.encode(min_valid_text, add_special_tokens=False, return_tensors='pt')[0]
                    # Replace first sequence with valid one
                    if input_ids_active.shape[0] > 0 and input_ids_active.shape[1] > 0:
                        seq_len = input_ids_active.shape[1]
                        if min_valid_ids.shape[0] <= seq_len:
                            padded_valid = torch.full((seq_len,), pad_token_id, dtype=min_valid_ids.dtype)
                            padded_valid[:min_valid_ids.shape[0]] = min_valid_ids
                            input_ids_active[0] = padded_valid
                            attention_mask_active[0] = (padded_valid != pad_token_id).long()
                            has_valid_sequence = True
                        else:
                            # Sequence too long, truncate
                            input_ids_active[0] = min_valid_ids[:seq_len]
                            attention_mask_active[0] = (min_valid_ids[:seq_len] != pad_token_id).long()
                            has_valid_sequence = True
                except Exception as e:
                    print(f"[ERROR] Failed to create fallback sequence: {e}")
                
                if not has_valid_sequence:
                    # Final fallback: return empty responses
                    print(f"[ERROR] All fallback attempts failed, returning empty responses")
                    empty_responses = torch.empty((batch_size, 0), dtype=torch.long)
                    empty_responses_str = [""] * batch_size
                    return empty_responses, empty_responses_str
        
        # Create DataProto for generation (only active samples)
        # Ensure all tensors are correct dtype (long for input_ids and position_ids)
        position_ids = self.tensor_fn.create_position_ids(attention_mask_active)
        gen_batch = DataProto.from_dict({
            'input_ids': input_ids_active.long(),
            'attention_mask': attention_mask_active.long() if attention_mask_active.dtype != torch.long else attention_mask_active,
            'position_ids': position_ids.long() if position_ids.dtype != torch.long else position_ids
        })
        
        # Generate (reuse single-agent generation logic with GPU padding)
        print(f"[CoSearch-R1] Calling _generate_with_gpu_padding for {agent_name}...")
        gen_output = self._generate_with_gpu_padding(gen_batch)
        print(f"[CoSearch-R1] _generate_with_gpu_padding completed for {agent_name}")
        
        # CRITICAL: Ensure all tensors are long/int type after generation
        # This is necessary because sharding managers or other operations may change dtype
        for key in gen_output.batch.keys():
            tensor = gen_output.batch[key]
            # Only convert if it's not already an integer type
            if tensor.dtype not in (torch.long, torch.int, torch.int32, torch.int64):
                gen_output.batch[key] = tensor.long()
            else:
                # Ensure it's long type
                gen_output.batch[key] = tensor.long()
        
        # Decode responses
        responses_ids = gen_output.batch['responses']
        responses_str = self.tokenizer.batch_decode(
            responses_ids,
            skip_special_tokens=False
        )
        
        # Post-process to stop at agent boundary or stop tokens
        processed_responses = []
        for resp in responses_str:
            # Stop at </agent> if present
            if '</agent>' in resp:
                resp = resp.split('</agent>')[0] + '</agent>'
            # Stop at other stop tokens
            for stop_token in stop_tokens:
                if stop_token in resp:
                    resp = resp.split(stop_token)[0] + stop_token
                    break
            processed_responses.append(resp)
        
        # Re-tokenize processed responses
        processed_ids = self._batch_tokenize(processed_responses)
        
        # Pad back to original batch size if we filtered inactive samples
        if active_indices is not None:
            # Use _example_level_pad to pad inactive samples
            processed_ids, processed_responses = self.tensor_fn._example_level_pad(
                processed_ids, 
                processed_responses, 
                active_mask
            )
        
        return processed_ids, processed_responses
    
    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
        Wrapper for generation that handles multi-GPU padding requirements.
        Reuses logic from single-agent LLMGenerationManager.
        """
        print(f"[CoSearch-R1] _generate_with_gpu_padding: batch_size={active_batch.batch['input_ids'].shape[0]}, num_gpus={self.config.num_gpus}")
        # Ensure all tensors are long/int type (not float) before generation
        for key in active_batch.batch.keys():
            tensor = active_batch.batch[key]
            # Only convert if it's not already an integer type
            if tensor.dtype not in (torch.long, torch.int, torch.int32, torch.int64):
                active_batch.batch[key] = tensor.long()
            else:
                # Ensure it's long type
                active_batch.batch[key] = tensor.long()
        
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            output = self.actor_rollout_wg.generate_sequences(active_batch)
            # Ensure all tensors in output are long/int type (not float)
            for key in output.batch.keys():
                tensor = output.batch[key]
                # Only convert if it's not already an integer type
                if tensor.dtype not in (torch.long, torch.int, torch.int32, torch.int64):
                    output.batch[key] = tensor.long()
                else:
                    # Ensure it's long type
                    output.batch[key] = tensor.long()
            return output
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        # Ensure all tensors are long/int type (not float)
        for key in active_batch.batch.keys():
            tensor = active_batch.batch[key]
            # Only convert if it's not already an integer type
            if tensor.dtype not in (torch.long, torch.int, torch.int32, torch.int64):
                active_batch.batch[key] = tensor.long()
            else:
                # Ensure it's long type
                active_batch.batch[key] = tensor.long()
        
        if remainder == 0:
            output = self.actor_rollout_wg.generate_sequences(active_batch)
            # Ensure all tensors in output are long/int type (not float)
            for key in output.batch.keys():
                tensor = output.batch[key]
                # Only convert if it's not already an integer type
                if tensor.dtype not in (torch.long, torch.int, torch.int32, torch.int64):
                    output.batch[key] = tensor.long()
                else:
                    # Ensure it's long type
                    output.batch[key] = tensor.long()
            return output
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        # Find a non-empty sequence to use as padding template
        # Check if first sequence is valid (not all pad)
        pad_token_id = self.tokenizer.pad_token_id
        input_ids = active_batch.batch['input_ids']
        template_idx = 0
        
        # Find first non-empty sequence
        found_valid = False
        for i in range(batch_size):
            if (input_ids[i] != pad_token_id).any():
                template_idx = i
                found_valid = True
                break
        
        # If no valid sequence found, use first one anyway (shouldn't happen if validation passed)
        if not found_valid:
            print(f"[WARNING] No valid sequence found for padding template, using first sequence")
            template_idx = 0
        
        for k, v in active_batch.batch.items():
            # Use template sequence (non-empty) as padding template
            pad_sequence = v[template_idx:template_idx+1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        # Ensure all tensors are long/int type (not float)
        for key in padded_active_batch.batch.keys():
            tensor = padded_active_batch.batch[key]
            # Only convert if it's not already an integer type
            if tensor.dtype not in (torch.long, torch.int, torch.int32, torch.int64):
                padded_active_batch.batch[key] = tensor.long()
            else:
                # Ensure it's long type
                padded_active_batch.batch[key] = tensor.long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        
        # Ensure all tensors in output are long/int type (not float)
        for key in padded_output.batch.keys():
            tensor = padded_output.batch[key]
            # Only convert if it's not already an integer type
            if tensor.dtype not in (torch.long, torch.int, torch.int32, torch.int64):
                padded_output.batch[key] = tensor.long()
            else:
                # Ensure it's long type
                padded_output.batch[key] = tensor.long()

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Ensure all tensors in trimmed_batch are long/int type (not float)
        for key in trimmed_batch.keys():
            tensor = trimmed_batch[key]
            # Only convert if it's not already an integer type
            if tensor.dtype not in (torch.long, torch.int, torch.int32, torch.int64):
                trimmed_batch[key] = tensor.long()
            else:
                # Ensure it's long type
                trimmed_batch[key] = tensor.long()
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size] if len(v.shape) > 0 and v.shape[0] > padding_size else v
                else:
                    trimmed_meta[k] = v
        else:
            trimmed_meta = {}
        
        trimmed_output = DataProto.from_dict(trimmed_batch)
        trimmed_output.meta_info = trimmed_meta
        
        return trimmed_output
    
    def _execute_search(self, queries: List[str]) -> List[str]:
        """
        Execute batch search.
        
        Args:
            queries: List of search queries
            
        Returns:
            List of formatted search results
        """
        if not self.search_client:
            print(f"[WARNING] Search client not set. Returning empty search results.")
            return [""] * len(queries) if queries else []
        
        if not queries:
            return []
        
        # Use the search client's batch_search method
        results = self.search_client.batch_search(queries)
        return results
    
    def run_multi_agent_rollout(
        self,
        gen_batch: DataProto,
        initial_input_ids: torch.Tensor,
        sample_prompts: List[str],
        global_step: int = 0
    ) -> Tuple[DataProto, MultiAgentTrajectory]:
        print(f"[CoSearch-R1] run_multi_agent_rollout called (global_step={global_step}, batch_size={gen_batch.batch['input_ids'].shape[0]})")
        """
        Run complete multi-agent rollout.
        
        Args:
            gen_batch: Initial batch data
            initial_input_ids: Initial input token IDs
            sample_prompts: Original prompts for each sample
            global_step: Current global training step (for logging frequency control)
            
        Returns:
            Tuple of (final_output, trajectory)
        """
        batch_size = gen_batch.batch['input_ids'].shape[0]
        trajectory = MultiAgentTrajectory()
        
        # Initialize dialog contexts (one per sample)
        # Extract questions from prompts and build multi-agent templates
        dialogs = []
        pad_token_id = self.tokenizer.pad_token_id
        
        # Helper function to extract question from prompt text
        def extract_question_from_prompt(prompt_text: str) -> str:
            """Extract the actual question from a formatted prompt."""
            import re
            # Clean the prompt first
            cleaned = re.sub(r'<\|endoftext\|>+', '', prompt_text).strip()
            
            # Format 1: Extract from chat template (user message) - most common format
            if '<|im_start|>user' in cleaned:
                match = re.search(r'<\|im_start\|>user\s*(.+?)<\|im_end\|>', cleaned, re.DOTALL)
                if match:
                    user_content = match.group(1).strip()
                    # Try to extract question from user content
                    # Look for "Question: ..." pattern
                    q_match = re.search(r'[Qq]uestion:\s*(.+?)(?:\n|$|<\|)', user_content, re.DOTALL)
                    if q_match:
                        question = q_match.group(1).strip()
                        # Remove trailing special tokens
                        question = re.sub(r'<\|.*?\|>', '', question).strip()
                        if question:
                            return question
                    # If no "Question:" found, try to extract the last sentence or meaningful content
                    # Remove the instruction part and get the question
                    # Pattern: "Answer the given question. ... Question: ..."
                    q_match2 = re.search(r'[Qq]uestion:\s*(.+?)(?:\n|$|<\|)', user_content, re.DOTALL)
                    if q_match2:
                        question = q_match2.group(1).strip()
                        question = re.sub(r'<\|.*?\|>', '', question).strip()
                        if question:
                            return question
                    # If still no match, return the user content (might be the question itself)
                    user_content_clean = re.sub(r'<\|.*?\|>', '', user_content).strip()
                    if user_content_clean and len(user_content_clean) > 10:  # Must be meaningful
                        return user_content_clean
            
            # Format 2: "Question: ..." (single-agent format, without chat template)
            match = re.search(r'[Qq]uestion:\s*(.+?)(?:\n|$|<\|)', cleaned, re.DOTALL)
            if match:
                question = match.group(1).strip()
                question = re.sub(r'<\|.*?\|>', '', question).strip()
                if question:
                    return question
            
            # Format 3: "Answer the given question. ... Question: ..."
            match = re.search(r'Answer the given question[^Q]*[Qq]uestion:\s*(.+?)(?:\n|$|<\|)', cleaned, re.DOTALL)
            if match:
                question = match.group(1).strip()
                question = re.sub(r'<\|.*?\|>', '', question).strip()
                if question:
                    return question
            
            # Fallback: try to extract meaningful content after removing chat template markers
            cleaned = re.sub(r'<\|im_start\|>.*?<\|im_end\|>', '', cleaned, flags=re.DOTALL).strip()
            cleaned = re.sub(r'<\|.*?\|>', '', cleaned).strip()
            # If cleaned content is meaningful (not just instructions), return it
            if cleaned and len(cleaned) > 20 and not cleaned.startswith("Answer the given question"):
                return cleaned
            
            # Last resort fallback
            return "Answer the question."
        
        # Build dialogs using multi-agent template
        for i in range(batch_size):
            # Try to get prompt from sample_prompts first
            prompt_text = None
            if sample_prompts and i < len(sample_prompts):
                prompt_text = sample_prompts[i]
            
            # If sample_prompts is empty or invalid, decode from initial_input_ids
            if not prompt_text or not prompt_text.strip() or prompt_text.strip().startswith('<|endoftext|>'):
                if i < initial_input_ids.shape[0]:
                    prompt_ids = initial_input_ids[i]
                    prompt_mask = gen_batch.batch.get('attention_mask', None)
                    
                    if prompt_mask is not None and i < prompt_mask.shape[0]:
                        valid_length = prompt_mask[i].sum().item()
                        valid_prompt = prompt_ids[:valid_length] if valid_length > 0 else prompt_ids
                    else:
                        # Find valid tokens (non-pad)
                        non_pad_mask = (prompt_ids != pad_token_id)
                        if non_pad_mask.any():
                            valid_prompt = prompt_ids[non_pad_mask]
                        else:
                            valid_prompt = prompt_ids
                    
                    prompt_text = self.tokenizer.decode(valid_prompt, skip_special_tokens=False)
                else:
                    prompt_text = "Answer the question."
            
            # Clean excessive <|endoftext|> tokens
            import re
            prompt_text = re.sub(r'<\|endoftext\|>+', '', prompt_text).strip()
            
            # Extract question from prompt
            question = extract_question_from_prompt(prompt_text)
            
            # Debug: log extracted question for first few samples
            if i < 2:
                print(f"[DEBUG] Sample {i} - Original prompt (first 200 chars): {prompt_text[:200]}")
                print(f"[DEBUG] Sample {i} - Extracted question: {question[:200] if len(question) > 200 else question}")
            
            # Build multi-agent template
            multi_agent_prompt = MultiAgentTemplate.get_multi_agent_template(
                question=question,
                num_explorers=self.num_explorers,
                roles=self.roles
            )
            
            dialogs.append(multi_agent_prompt)
        
        # Validate dialogs - ensure all are non-empty
        for i, dialog in enumerate(dialogs):
            if not dialog or not dialog.strip():
                # Last resort: create a minimal valid prompt
                print(f"[WARNING] Empty dialog at index {i}, using fallback prompt")
                dialogs[i] = "Answer the question."
        
        # Track active samples
        active_mask = torch.ones(batch_size, dtype=torch.bool)
        
        # Determine if we should log this step
        should_log = self.show_steps and (
            self.log_step_freq == 0 or 
            global_step == 0 or 
            (self.log_step_freq > 0 and global_step % self.log_step_freq == 0)
        )
        
        # Store all generated tokens and info masks
        all_responses = []
        all_responses_with_info_mask = []
        all_info_blocks = []
        
        # Step 1: Planner generates initial analysis
        planner_stop_tokens = ["</agent>"]
        
        print(f"[CoSearch-R1] Starting Planner generation (batch_size={batch_size}, active={active_mask.sum().item()})")
        if should_log:
            print(f"\n{'='*80}")
            print(f"[CoSearch-R1] Step {global_step} - Planner Generation")
            print(f"{'='*80}")
        
        # Add explicit Planner instruction before generation
        planner_instruction = MultiAgentTemplate.get_role_instruction("Planner")
        planner_prompt_suffix = f"\n\n{planner_instruction}\n\nNow, as Planner, begin your analysis:\n"
        
        # Add instruction to each dialog context
        dialogs_with_planner_instruction = []
        for dialog in dialogs:
            dialogs_with_planner_instruction.append(dialog + planner_prompt_suffix)
        
        print(f"[CoSearch-R1] Calling _generate_agent_responses_batch for Planner...")
        planner_ids, planner_strs = self._generate_agent_responses_batch(
            dialog_contexts=dialogs_with_planner_instruction,
            agent_name="Planner",
            stop_tokens=planner_stop_tokens,
            active_mask=active_mask
        )
        print(f"[CoSearch-R1] Planner generation completed. Response count: {len(planner_strs)}")
        
        # Parse planner segments and log
        for i, planner_str in enumerate(planner_strs):
            if active_mask[i]:
                parse_result = MultiAgentProtocol.parse_agent_segment(planner_str)
                if parse_result is not None:
                    role, content = parse_result
                    if role:
                        segment = MultiAgentProtocol.parse_segment(planner_str, role)
                        if segment:
                            trajectory.add_segment(segment)
                        dialogs[i] += "\n" + planner_str + "\n"
                        
                        # Log sample outputs
                        if should_log and i < self.log_sample_count:
                            print(f"\n--- Sample {i+1} - Planner ---")
                            print(planner_str[:500] + "..." if len(planner_str) > 500 else planner_str)
                            # Debug: show dialog after Planner
                            if should_log and i < 2:
                                print(f"[DEBUG] Dialog after Planner (first 400 chars): {dialogs[i][:400]}")
                else:
                    # No agent tag found, still append to dialog
                    print(f"[WARNING] Planner response at index {i} has no <agent> tag, appending as-is")
                    dialogs[i] += "\n" + planner_str + "\n"
                    # Debug: show dialog after Planner
                    if should_log and i < 2:
                        print(f"[DEBUG] Dialog after Planner (first 400 chars): {dialogs[i][:400]}")
        
        all_responses.append(planner_ids)
        all_responses_with_info_mask.append(planner_ids)  # Planner doesn't have info blocks
        
        # Step 2: Multiple Explorer rounds
        print(f"[CoSearch-R1] Starting Explorer rounds (max_rounds={self.max_explore_rounds}, num_explorers={self.num_explorers})")
        for round_idx in range(self.max_explore_rounds):
            if not active_mask.any():
                print(f"[CoSearch-R1] No active samples, breaking Explorer loop at round {round_idx+1}")
                break
            
            print(f"[CoSearch-R1] Explorer Round {round_idx+1}/{self.max_explore_rounds}")
            for explorer_idx in range(self.num_explorers):
                if not active_mask.any():
                    print(f"[CoSearch-R1] No active samples, breaking Explorer loop at explorer {explorer_idx+1}")
                    break
                
                agent_name = f"Explorer_{explorer_idx + 1}"
                explorer_stop_tokens = ["</agent>"]
                
                print(f"[CoSearch-R1] Generating {agent_name} response (round {round_idx+1})...")
                if should_log:
                    print(f"\n--- Round {round_idx+1} - {agent_name} ---")
                
                # Add explicit role instruction for Explorer before generation
                # This emphasizes that Explorer MUST search
                explorer_instruction = MultiAgentTemplate.get_role_instruction(agent_name)
                explorer_prompt_suffix = f"\n\n{explorer_instruction}\n\nNow, as {agent_name}, you MUST search for information. Begin your response:\n"
                
                # Add instruction to each dialog context
                dialogs_with_instruction = []
                for dialog in dialogs:
                    # Only add instruction if this is the first round or if no search has been performed yet
                    # Check if dialog already contains search results
                    if "<information" not in dialog or round_idx == 0:
                        dialogs_with_instruction.append(dialog + explorer_prompt_suffix)
                    else:
                        dialogs_with_instruction.append(dialog + f"\n\nAs {agent_name}, continue searching if needed:\n")
                
                # Generate explorer response (each sample has independent dialog context)
                print(f"[CoSearch-R1] Calling _generate_agent_responses_batch for {agent_name}...")
                explorer_ids, explorer_strs = self._generate_agent_responses_batch(
                    dialog_contexts=dialogs_with_instruction,
                    agent_name=agent_name,
                    stop_tokens=explorer_stop_tokens,
                    active_mask=active_mask
                )
                print(f"[CoSearch-R1] {agent_name} generation completed. Response count: {len(explorer_strs)}")
                
                # Parse and process explorer segments
                all_search_queries = []
                explorer_info_map = {}  # Map sample idx to queries
                
                for i, explorer_str in enumerate(explorer_strs):
                    if active_mask[i]:
                        # Always try to extract search queries first (even without agent tag)
                        search_queries = MultiAgentProtocol.extract_search_queries(explorer_str)
                        
                        parse_result = MultiAgentProtocol.parse_agent_segment(explorer_str)
                        if parse_result is not None:
                            role, content = parse_result
                            if role:
                                segment = MultiAgentProtocol.parse_segment(explorer_str, role)
                                if segment:
                                    trajectory.add_segment(segment)
                                dialogs[i] += "\n" + explorer_str + "\n"
                                
                                # Log sample outputs
                                if should_log and i < self.log_sample_count:
                                    print(f"\n  Sample {i+1} - {agent_name}:")
                                    print(f"  {explorer_str[:300] + '...' if len(explorer_str) > 300 else explorer_str}")
                                
                                # Collect search queries (use segment.search_queries if available, otherwise use extracted)
                                if segment and segment.search_queries:
                                    explorer_info_map[i] = segment.search_queries
                                    all_search_queries.extend(segment.search_queries)
                                    
                                    # Log search queries
                                    if should_log and i < self.log_sample_count:
                                        print(f"  Search queries (from segment): {segment.search_queries}")
                                elif search_queries:
                                    # Use extracted search queries if segment doesn't have them
                                    explorer_info_map[i] = search_queries
                                    all_search_queries.extend(search_queries)
                                    
                                    # Log search queries
                                    if should_log and i < self.log_sample_count:
                                        print(f"  Search queries (extracted): {search_queries}")
                                else:
                                    # No search queries found
                                    if should_log and i < self.log_sample_count:
                                        print(f"  [INFO] No search queries found in {agent_name} response")
                        else:
                            # No agent tag found, still append to dialog and try to extract search queries
                            print(f"[WARNING] {agent_name} response at index {i} has no <agent> tag, appending as-is")
                            dialogs[i] += "\n" + explorer_str + "\n"
                            
                            # Use extracted search queries
                            if search_queries:
                                explorer_info_map[i] = search_queries
                                all_search_queries.extend(search_queries)
                                
                                # Log search queries
                                if should_log and i < self.log_sample_count:
                                    print(f"  Search queries (extracted, no agent tag): {search_queries}")
                            else:
                                # Debug: show why no search queries were found
                                if should_log and i < self.log_sample_count:
                                    print(f"  [DEBUG] No search queries found. Response contains '<search>': {'<search>' in explorer_str}")
                                    print(f"  [DEBUG] Response snippet: {explorer_str[:200]}")
                
                # Log total search queries collected
                if should_log:
                    print(f"\n  [INFO] Total search queries collected for {agent_name}: {len(all_search_queries)}")
                    if all_search_queries:
                        print(f"  [INFO] Search queries: {all_search_queries[:5]}")  # Show first 5
                
                all_responses.append(explorer_ids)
                all_responses_with_info_mask.append(explorer_ids)
                
                # Execute searches and insert information blocks
                if all_search_queries:
                    if should_log:
                        print(f"\n  [INFO] Executing {len(all_search_queries)} search queries for {agent_name}...")
                    search_results = self._execute_search(all_search_queries)
                    
                    if should_log:
                        print(f"  [INFO] Received {len(search_results)} search results")
                        if search_results:
                            print(f"  [INFO] First result length: {len(search_results[0])} chars")
                    
                    # Map results back to samples
                    result_idx = 0
                    batch_info_blocks = []  # List of info blocks for this batch (one per sample)
                    
                    for i in range(batch_size):
                        if i in explorer_info_map:
                            queries = explorer_info_map[i]
                            num_queries = len(queries)
                            sample_results = search_results[result_idx:result_idx + num_queries]
                            result_idx += num_queries
                            
                            # Format information block
                            combined_results = "\n".join(sample_results)
                            info_block = MultiAgentProtocol.wrap_information_block(
                                combined_results,
                                agent_name=agent_name
                            )
                            dialogs[i] += info_block
                            
                            trajectory.add_information(agent_name, combined_results)
                            
                            # Log search results
                            if should_log and i < self.log_sample_count:
                                print(f"  Search results for Sample {i+1} ({len(sample_results)} docs):")
                                for j, result in enumerate(sample_results[:2]):  # Show first 2 results
                                    print(f"    Doc {j+1}: {result[:200] + '...' if len(result) > 200 else result}")
                            
                            # Tokenize info block for masking (per sample)
                            info_ids = self.tokenizer.encode(
                                info_block,
                                add_special_tokens=False,
                                return_tensors='pt'
                            )
                            batch_info_blocks.append(info_ids[0])  # Remove batch dimension
                        else:
                            batch_info_blocks.append(torch.empty((0,), dtype=torch.long))
                    
                    # Store info blocks for this explorer (will be interleaved later)
                    all_info_blocks.append(batch_info_blocks)
                    
                    # Update responses_with_info_mask (info blocks should be masked)
                    # This will be handled in _compose_final_output
                else:
                    if should_log:
                        print(f"\n  [WARNING] No search queries to execute for {agent_name}")
                    # No search queries, so no info blocks for this explorer
                    batch_info_blocks = [torch.empty((0,), dtype=torch.long) for _ in range(batch_size)]
                    all_info_blocks.append(batch_info_blocks)
        
        # Step 3: Synthesizer generates final answer
        if active_mask.any():
            print(f"[CoSearch-R1] Starting Synthesizer generation (active={active_mask.sum().item()})")
            synthesizer_stop_tokens = ["</answer>", "</agent>"]
            
            if should_log:
                print(f"\n--- Synthesizer Generation ---")
            
            # Add explicit Synthesizer instruction before generation
            synthesizer_instruction = MultiAgentTemplate.get_role_instruction("Synthesizer")
            synthesizer_prompt_suffix = f"\n\n{synthesizer_instruction}\n\nNow, as Synthesizer, provide the final answer:\n"
            
            # Add instruction to each dialog context
            dialogs_with_synthesizer_instruction = []
            for dialog in dialogs:
                dialogs_with_synthesizer_instruction.append(dialog + synthesizer_prompt_suffix)
            
            print(f"[CoSearch-R1] Calling _generate_agent_responses_batch for Synthesizer...")
            synthesizer_ids, synthesizer_strs = self._generate_agent_responses_batch(
                dialog_contexts=dialogs_with_synthesizer_instruction,
                agent_name="Synthesizer",
                stop_tokens=synthesizer_stop_tokens,
                active_mask=active_mask
            )
            print(f"[CoSearch-R1] Synthesizer generation completed. Response count: {len(synthesizer_strs)}")
            
            # Parse synthesizer segments and log
            for i, synthesizer_str in enumerate(synthesizer_strs):
                if active_mask[i]:
                    parse_result = MultiAgentProtocol.parse_agent_segment(synthesizer_str)
                    if parse_result is not None:
                        role, content = parse_result
                        if role:
                            segment = MultiAgentProtocol.parse_segment(synthesizer_str, role)
                            if segment:
                                trajectory.add_segment(segment)
                            dialogs[i] += "\n" + synthesizer_str + "\n"
                            
                            # Log final answer
                            if should_log and i < self.log_sample_count:
                                print(f"\n  Sample {i+1} - Synthesizer:")
                    else:
                        # No agent tag found, still append to dialog
                        print(f"[WARNING] Synthesizer response at index {i} has no <agent> tag, appending as-is")
                        dialogs[i] += "\n" + synthesizer_str + "\n"
                        
                        # Log final answer even without agent tag
                        if should_log and i < self.log_sample_count:
                            print(f"\n  Sample {i+1} - Synthesizer:")
                            print(f"  {synthesizer_str[:400] + '...' if len(synthesizer_str) > 400 else synthesizer_str}")
                            
                            # Extract and show final answer
                            final_answer = MultiAgentProtocol.extract_answer(synthesizer_str)
                            if final_answer:
                                print(f"  → Final Answer: {final_answer}")
            
            all_responses.append(synthesizer_ids)
            all_responses_with_info_mask.append(synthesizer_ids)
            
            if should_log:
                print(f"\n{'='*80}\n")
        
        # Print ACTIVE_TRAJ_NUM for compatibility (multi-agent always completes all steps)
        active_num_list = [batch_size] * (1 + self.max_explore_rounds * self.num_explorers + 1)  # Planner + Explorers + Synthesizer
        print(f"[CoSearch-R1] ACTIVE_TRAJ_NUM: {active_num_list}")
        
        # Compose final output
        final_output = self._compose_final_output(
            initial_input_ids,
            all_responses,
            all_responses_with_info_mask,
            all_info_blocks,
            trajectory
        )
        
        return final_output, trajectory
    
    def _compose_final_output(
        self,
        prompts: torch.Tensor,
        responses: List[torch.Tensor],
        responses_with_info_mask: List[torch.Tensor],
        info_blocks: List[torch.Tensor],
        trajectory: MultiAgentTrajectory
    ) -> DataProto:
        """
        Compose final output similar to single-agent version.
        
        Args:
            prompts: Initial prompt token IDs
            responses: List of response token tensors
            responses_with_info_mask: Responses with info blocks masked
            info_blocks: Information block tokens
            trajectory: Multi-agent trajectory
            
        Returns:
            DataProto with final output
        """
        batch_size = prompts.shape[0]
        
        # Concatenate all responses
        if responses:
            all_responses = torch.cat(responses, dim=1)
        else:
            all_responses = torch.empty((batch_size, 0), dtype=torch.long)
        
        # Create responses_with_info_mask (same structure but with masked info blocks)
        if responses_with_info_mask:
            all_responses_with_mask = torch.cat(responses_with_info_mask, dim=1)
        else:
            all_responses_with_mask = all_responses
        
        # Process info blocks (these will be masked in loss)
        # info_blocks is a list of lists: [batch_info_blocks_round1, batch_info_blocks_round2, ...]
        # Each batch_info_blocks is a list of tensors, one per sample
        if info_blocks and len(info_blocks) > 0:
            # Flatten: collect all info blocks per sample across all rounds
            # Structure: info_blocks_per_sample[i] = [info_block_round1, info_block_round2, ...] for sample i
            info_blocks_per_sample = [[] for _ in range(batch_size)]
            for batch_info_blocks in info_blocks:
                for sample_idx, sample_info_block in enumerate(batch_info_blocks):
                    if sample_info_block.shape[0] > 0:  # Non-empty
                        info_blocks_per_sample[sample_idx].append(sample_info_block)
            
            # Concatenate info blocks for each sample, then pad to same length
            concatenated_info_per_sample = []
            for sample_info_list in info_blocks_per_sample:
                if sample_info_list:
                    concatenated_info_per_sample.append(torch.cat(sample_info_list, dim=0))
                else:
                    concatenated_info_per_sample.append(torch.empty((0,), dtype=torch.long))
            
            # Find max length and pad (only if there are non-empty info blocks)
            non_empty_info = [ib.shape[0] for ib in concatenated_info_per_sample if ib.shape[0] > 0]
            if non_empty_info:
                max_info_len = max(non_empty_info)
            else:
                max_info_len = 0
            
            if max_info_len > 0:
                padded_info_blocks = []
                for ib in concatenated_info_per_sample:
                    pad_len = max_info_len - ib.shape[0]
                    if pad_len > 0:
                        pad = torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=ib.dtype, device=ib.device)
                        padded_ib = torch.cat([ib, pad], dim=0)
                    else:
                        padded_ib = ib[:max_info_len]
                    padded_info_blocks.append(padded_ib)
                
                # Stack into batch tensor: (batch_size, max_info_len)
                all_info = torch.stack(padded_info_blocks, dim=0)
            else:
                all_info = torch.empty((batch_size, 0), dtype=torch.long)
        else:
            all_info = torch.empty((batch_size, 0), dtype=torch.long)
        
        # Cut prompts to max_start_length
        prompts = prompts[:, -self.config.max_start_length:]
        
        # Concatenate prompts, responses, and info blocks for final input
        # Note: In single-agent, info blocks are interleaved with responses
        # For multi-agent, we append info blocks at the end (simplified implementation)
        if all_info.shape[1] > 0:
            final_input_ids = torch.cat([prompts, all_responses, all_info], dim=1)
        else:
            final_input_ids = torch.cat([prompts, all_responses], dim=1)
        
        # Create attention mask (includes info blocks as valid tokens for attention, but masked in loss)
        attention_mask = self.tensor_fn.create_attention_mask(final_input_ids)
        
        # Create info_mask (mask out information blocks)
        # Format must match single-agent exactly: cat([prompt_mask, response_mask_with_info_mask])
        # In single-agent: responses_with_info_mask has info blocks replaced with pad_token_id
        # So we create attention mask from responses_with_info_mask (which masks info blocks)
        prompt_mask = self.tensor_fn.create_attention_mask(prompts)
        
        # Concatenate responses and info blocks for the full response sequence
        # But for info_mask, we use responses_with_info_mask (which has info masked as pad)
        if all_info.shape[1] > 0:
            # Concatenate responses_with_mask and info blocks
            # responses_with_mask already has info blocks masked (as pad_token_id in the original responses)
            # We need to append info blocks (which should be masked) to responses_with_mask
            # But actually, in single-agent, info blocks are inserted into responses, not appended
            # For now, we'll create a mask where info blocks are treated as pad
            response_with_info_for_mask = torch.cat([all_responses_with_mask, all_info], dim=1)
            # Info blocks should be masked (treated as pad_token_id)
            # Create mask: responses_with_mask (valid) + info blocks (masked as pad)
            response_mask_part = self.tensor_fn.create_attention_mask(all_responses_with_mask)
            info_mask_part = torch.zeros_like(all_info, dtype=torch.long)  # 0 = pad = masked
            response_info_mask = torch.cat([response_mask_part, info_mask_part], dim=1)
        else:
            # No info blocks, just use response mask
            response_with_info_for_mask = all_responses_with_mask
            response_info_mask = self.tensor_fn.create_attention_mask(all_responses_with_mask)
        
        # Final info_mask: prompt (all valid) + response_with_info (info blocks masked)
        # This matches single-agent format exactly
        info_mask = torch.cat([prompt_mask, response_info_mask], dim=1)
        
        # Create position IDs
        position_ids = self.tensor_fn.create_position_ids(attention_mask)
        
        # Compose final output
        final_output = DataProto.from_dict({
            'prompts': prompts,
            'responses': all_responses,
            'input_ids': final_input_ids,
            'attention_mask': attention_mask,
            'info_mask': info_mask,
            'position_ids': position_ids,
            'responses_with_info_mask': all_responses_with_mask
        })
        
        # Add trajectory metadata
        final_output.meta_info = {
            'trajectory': trajectory,
            'agent_stats': dict(trajectory.agent_stats),
            'num_segments': len(trajectory.segments),
            'num_information_blocks': len(trajectory.information_blocks)
        }
        
        return final_output

