# MA-Search-R1: Multi-Agent Search-R1

MA-Search-R1 extends Search-R1 by decomposing the single "thinking + searching" agent into three specialized agents that collaborate through multi-agent reinforcement learning.

## Architecture

### Three Agents

1. **Planner (规划者)**
   - Analyzes the question
   - Plans search strategy
   - Generates search queries in the format: `<plan>query1, query2, ...</plan>`

2. **Researcher (检索者)**
   - Executes searches based on Planner's queries
   - Retrieves information from the search engine
   - Returns results in the format: `<information>...</information>`

3. **Synthesizer (汇总者)**
   - Synthesizes information from Researcher
   - Decides whether more information is needed or can provide final answer
   - Outputs either:
     - `<answer>...</answer>` for final answer
     - `<need_more>reason</need_more>` to request more information

### Interaction Flow

```
Question → Planner → [queries] → Researcher → [results] → Synthesizer
                                                              ↓
                                              [need more?] ← ← ← ← ←
                                                              ↓
                                                          [answer]
```

## Usage

### Training

1. **Prepare data** (same as Search-R1):
```bash
python scripts/data_process/nq_search.py
```

2. **Launch retrieval server** (same as Search-R1):
```bash
conda activate retriever
bash retrieval_launch.sh
```

3. **Train multi-agent system**:
```bash
conda activate searchr1
bash train_ma_ppo.sh
```

### Configuration

Key configuration parameters in `train_ma_ppo.sh`:

- `multi_agent.use_shared_model=true`: Use shared model for all agents (recommended for initial experiments)
- `max_turns=2`: Maximum interaction turns
- `retriever.url`: Search engine API URL
- `retriever.topk`: Number of documents to retrieve

## Implementation Details

### Code Structure

- `ma_search_r1/llm_agent/multi_agent_generation.py`: Multi-agent generation manager
- `verl/trainer/main_ma_ppo.py`: Multi-agent PPO training script
- `verl/trainer/ppo/ma_ray_trainer.py`: Multi-agent Ray trainer

### Key Differences from Search-R1

1. **Generation Loop**: Uses `run_multi_agent_loop()` instead of `run_llm_loop()`
2. **Agent Coordination**: Three agents interact in sequence (Planner → Researcher → Synthesizer)
3. **Response Format**: Each agent has specific output formats for coordination

### Shared vs. Separate Models

- **Shared Model** (`use_shared_model=true`): All three agents use the same model with different prompts. More efficient and easier to train initially.
- **Separate Models** (`use_shared_model=false`): Each agent has its own model. More flexible but requires more resources.

## Extending

To add new agent roles or modify interaction patterns:

1. Add new agent role in `AgentRole` enum
2. Implement agent-specific generation logic in `MultiAgentGenerationManager`
3. Update interaction flow in `run_multi_agent_loop()`

## Notes

- MA-Search-R1 reuses all Search-R1 components (retrieval system, reward functions, etc.)
- The multi-agent system is designed to be compatible with existing Search-R1 infrastructure
- Initial experiments suggest shared models work well for collaborative tasks

