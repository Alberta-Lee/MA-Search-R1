"""
Multi-Agent Ray PPO Trainer for MA-Search-R1.

This trainer extends RayPPOTrainer to support multi-agent training with
three agents: Planner, Researcher, and Synthesizer.
"""

import torch
from typing import Dict
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role, ResourcePoolManager
from ma_search_r1.llm_agent.multi_agent_generation import MultiAgentGenerationManager, MultiAgentConfig


class MultiAgentRayPPOTrainer(RayPPOTrainer):
    """
    Multi-agent PPO trainer that extends RayPPOTrainer.
    Supports training three agents collaboratively.
    
    This trainer modifies the generation loop to use multi-agent generation
    instead of single-agent generation.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_shared_model = self.config.get('multi_agent', {}).get('use_shared_model', True)
        self.ma_generation_manager = None
    
    def _init_multi_agent_generation_manager(self):
        """Initialize multi-agent generation manager."""
        gen_config = MultiAgentConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            search_url=self.config.retriever.url,
            topk=self.config.retriever.topk,
            use_shared_model=self.use_shared_model,
        )
        
        # Get worker groups for each agent
        if self.use_shared_model:
            # All agents use the same model
            planner_wg = self.actor_rollout_wg
            researcher_wg = self.actor_rollout_wg
            synthesizer_wg = self.actor_rollout_wg
        else:
            # Each agent has its own model (would need separate worker groups)
            # For now, use shared model
            planner_wg = self.actor_rollout_wg
            researcher_wg = self.actor_rollout_wg
            synthesizer_wg = self.actor_rollout_wg
        
        self.ma_generation_manager = MultiAgentGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg_planner=planner_wg,
            actor_rollout_wg_researcher=researcher_wg,
            actor_rollout_wg_synthesizer=synthesizer_wg,
            config=gen_config,
            is_validation=False,
        )
    
    def fit(self):
        """
        Override fit to use multi-agent generation.
        We replace the generation_manager with ma_generation_manager in the training loop.
        """
        # Initialize multi-agent generation manager
        self._init_multi_agent_generation_manager()
        
        # Ensure search mode is on
        self.config.do_search = True
        
        # We need to override the fit method to use multi-agent generation
        # Instead of calling super().fit(), we'll copy the fit logic but use ma_generation_manager
        from verl.trainer.ppo.ray_trainer import _timer, reduce_metrics, apply_kl_penalty, compute_advantage
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
        from pprint import pprint
        import numpy as np
        import uuid
        
        logger = self.logger
        self.global_steps = 0
        
        # Validation
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return
        
        self.global_steps += 1
        
        # Start training loop
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                print(f'epoch {epoch}, step {self.global_steps}')
                metrics = {}
                timing_raw = {}
                
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)
                
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                
                with _timer('step', timing_raw):
                    if not self.config.do_search:
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)
                    else:
                        # Use multi-agent generation
                        first_input_ids = gen_batch.batch['input_ids'][:, -self.config.data.max_start_length:].clone().long()
                        
                        with _timer('gen', timing_raw):
                            self.ma_generation_manager.timing_raw = timing_raw
                            final_gen_batch_output = self.ma_generation_manager.run_multi_agent_loop(
                                gen_batch=gen_batch,
                                initial_input_ids=first_input_ids,
                            )
                            
                            for key in final_gen_batch_output.batch.keys():
                                final_gen_batch_output.batch[key] = final_gen_batch_output.batch[key].long()
                            
                            with torch.no_grad():
                                output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                                final_gen_batch_output = final_gen_batch_output.union(output)
                            
                            batch.non_tensor_batch['uid'] = batch.non_tensor_batch['index'].copy()
                            batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                            batch = batch.union(final_gen_batch_output)
                
                # Continue with rest of training loop (same as parent)
                self._balance_batch(batch, metrics=metrics)
                batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()
                
                for key in batch.batch.keys():
                    if key != 'old_log_probs':
                        batch.batch[key] = batch.batch[key].long()
                
                if self.use_reference_policy:
                    with _timer('ref', timing_raw):
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)
                
                if self.use_critic:
                    with _timer('values', timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)
                
                with _timer('adv', timing_raw):
                    if self.use_rm:
                        reward_tensor = self.rm_wg.compute_rm_score(batch)
                        batch = batch.union(reward_tensor)
                    
                    reward_tensor = self.reward_fn(batch)
                    batch.batch['token_level_scores'] = reward_tensor
                    
                    if not self.config.actor_rollout_ref.actor.use_kl_loss:
                        batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl,
                                                             kl_penalty=self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
                
                with _timer('adv', timing_raw):
                    batch = compute_advantage(batch, self.config.algorithm.adv_estimator,
                                             gamma=self.config.algorithm.gamma,
                                             lam=self.config.algorithm.lam,
                                             num_repeat=self.config.actor_rollout_ref.rollout.n)
                
                with _timer('actor', timing_raw):
                    actor_metrics = self.actor_rollout_wg.update_actor(batch)
                    metrics.update(actor_metrics)
                
                if self.use_critic:
                    with _timer('critic', timing_raw):
                        critic_metrics = self.critic_wg.update_critic(batch)
                        metrics.update(critic_metrics)
                
                metrics.update(reduce_metrics(timing_raw))
                from verl.trainer.ppo.ray_trainer import compute_data_metrics
                metrics.update(compute_data_metrics(batch, use_critic=self.use_critic))
                
                logger.log(data=metrics, step=self.global_steps)
                
                if self.global_steps % self.config.trainer.save_freq == 0:
                    self._save_checkpoint()
                
                if self.global_steps % self.config.trainer.test_freq == 0:
                    val_metrics = self._validate()
                    logger.log(data=val_metrics, step=self.global_steps)
                
                self.global_steps += 1
                
                if self.global_steps > self.total_training_steps:
                    break
            
            if self.global_steps > self.total_training_steps:
                break

