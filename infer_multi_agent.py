"""
Multi-agent inference script for CoSearch-R1.
Supports both single-agent and multi-agent modes.
"""

import transformers
import torch
import argparse
import requests
from search_r1.multi_agent.templates import MultiAgentTemplate
from search_r1.multi_agent.protocol import MultiAgentProtocol


def get_query(text):
    """Extract search query from text."""
    import re
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None


def search(query: str, search_url: str = "http://127.0.0.1:8000/retrieve", topk: int = 3):
    """Execute search query."""
    payload = {
        "queries": [query],
        "topk": topk,
        "return_scores": True
    }
    results = requests.post(search_url, json=payload).json()['result']
    
    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    return _passages2string(results[0])


def infer_single_agent(question: str, model, tokenizer, device, search_url: str, topk: int):
    """Single-agent inference (original Search-R1)."""
    question = question.strip()
    if question[-1] != '?':
        question += '?'
    
    curr_eos = [151645, 151643]  # for Qwen2.5 series models
    curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'
    
    prompt = MultiAgentTemplate.get_single_agent_template(question)
    
    # Define stopping criteria
    class StopOnSequence(transformers.StoppingCriteria):
        def __init__(self, target_sequences, tokenizer):
            self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) 
                             for target_sequence in target_sequences]
            self.target_lengths = [len(target_id) for target_id in self.target_ids]
            self._tokenizer = tokenizer

        def __call__(self, input_ids, scores, **kwargs):
            targets = [torch.as_tensor(target_id, device=input_ids.device) 
                      for target_id in self.target_ids]
            if input_ids.shape[1] < min(self.target_lengths):
                return False
            for i, target in enumerate(targets):
                if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                    return True
            return False
    
    target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
    stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])
    
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], 
            add_generation_prompt=True, 
            tokenize=False
        )
    
    print('\n\n################# [Single-Agent: Start Reasoning + Searching] ##################\n\n')
    print(prompt)
    
    while True:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)
        
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7
        )

        if outputs[0][-1].item() in curr_eos:
            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print(output_text)
            break

        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
        if tmp_query:
            search_results = search(tmp_query, search_url, topk)
        else:
            search_results = ''

        search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
        prompt += search_text
        print(search_text)


def infer_multi_agent(question: str, model, tokenizer, device, search_url: str, topk: int, 
                     num_explorers: int = 2, max_explore_rounds: int = 2):
    """Multi-agent inference (CoSearch-R1)."""
    question = question.strip()
    if question[-1] != '?':
        question += '?'
    
    # Get multi-agent template
    prompt = MultiAgentTemplate.get_multi_agent_template(question, num_explorers=num_explorers)
    
    # Define stopping criteria for agent boundaries
    class StopOnSequence(transformers.StoppingCriteria):
        def __init__(self, target_sequences, tokenizer):
            self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) 
                             for target_sequence in target_sequences]
            self.target_lengths = [len(target_id) for target_id in self.target_ids]
            self._tokenizer = tokenizer

        def __call__(self, input_ids, scores, **kwargs):
            targets = [torch.as_tensor(target_id, device=input_ids.device) 
                      for target_id in self.target_ids]
            if input_ids.shape[1] < min(self.target_lengths):
                return False
            for i, target in enumerate(targets):
                if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                    return True
            return False
    
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], 
            add_generation_prompt=True, 
            tokenize=False
        )
    
    print('\n\n################# [Multi-Agent: Start Collaborative Search] ##################\n\n')
    print(prompt)
    
    dialog = prompt
    current_role = "Planner"
    explore_round = 0
    
    # Step 1: Planner
    print(f"\n--- {current_role} ---")
    target_sequences = ["</agent>"]
    stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])
    
    input_ids = tokenizer.encode(dialog, return_tensors='pt').to(device)
    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        stopping_criteria=stopping_criteria,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7
    )
    
    planner_output = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=False)
    if '</agent>' in planner_output:
        planner_output = planner_output.split('</agent>')[0] + '</agent>'
    dialog += "\n" + planner_output
    print(planner_output)
    
    # Step 2: Explorers
    for round_idx in range(max_explore_rounds):
        for explorer_idx in range(num_explorers):
            agent_name = f"Explorer_{explorer_idx + 1}"
            print(f"\n--- {agent_name} ---")
            
            target_sequences = ["</agent>"]
            stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])
            
            input_ids = tokenizer.encode(dialog, return_tensors='pt').to(device)
            outputs = model.generate(
                input_ids,
                max_new_tokens=512,
                stopping_criteria=stopping_criteria,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7
            )
            
            explorer_output = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=False)
            if '</agent>' in explorer_output:
                explorer_output = explorer_output.split('</agent>')[0] + '</agent>'
            dialog += "\n" + explorer_output
            print(explorer_output)
            
            # Check for search queries and execute
            search_queries = MultiAgentProtocol.extract_search_queries(explorer_output)
            if search_queries:
                for query in search_queries:
                    search_results = search(query, search_url, topk)
                    info_block = MultiAgentProtocol.wrap_information_block(search_results, agent_name=agent_name)
                    dialog += info_block
                    print(f"\n--- Search Results for {agent_name} ---")
                    print(info_block)
    
    # Step 3: Synthesizer
    current_role = "Synthesizer"
    print(f"\n--- {current_role} ---")
    target_sequences = ["</answer>", "</agent>"]
    stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])
    
    input_ids = tokenizer.encode(dialog, return_tensors='pt').to(device)
    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        stopping_criteria=stopping_criteria,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7
    )
    
    synthesizer_output = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=False)
    if '</answer>' in synthesizer_output:
        synthesizer_output = synthesizer_output.split('</answer>')[0] + '</answer>'
    elif '</agent>' in synthesizer_output:
        synthesizer_output = synthesizer_output.split('</agent>')[0] + '</agent>'
    
    dialog += "\n" + synthesizer_output
    print(synthesizer_output)
    
    # Extract final answer
    final_answer = MultiAgentProtocol.extract_answer(synthesizer_output)
    if final_answer:
        print(f"\n\n################# [Final Answer] ##################\n{final_answer}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for Search-R1 / CoSearch-R1")
    parser.add_argument("--model_id", type=str, 
                       default="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo",
                       help="Model ID or path")
    parser.add_argument("--question", type=str,
                       default="Mike Barnett negotiated many contracts including which player that went on to become general manager of CSKA Moscow of the Kontinental Hockey League?",
                       help="Question to answer")
    parser.add_argument("--multi_agent", action="store_true",
                       help="Enable multi-agent mode (CoSearch-R1)")
    parser.add_argument("--num_explorers", type=int, default=2,
                       help="Number of Explorer agents (multi-agent mode only)")
    parser.add_argument("--max_explore_rounds", type=int, default=2,
                       help="Maximum exploration rounds (multi-agent mode only)")
    parser.add_argument("--search_url", type=str, default="http://127.0.0.1:8000/retrieve",
                       help="Search server URL")
    parser.add_argument("--topk", type=int, default=3,
                       help="Number of retrieved documents per query")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    # Run inference
    if args.multi_agent:
        infer_multi_agent(
            args.question,
            model,
            tokenizer,
            device,
            args.search_url,
            args.topk,
            args.num_explorers,
            args.max_explore_rounds
        )
    else:
        infer_single_agent(
            args.question,
            model,
            tokenizer,
            device,
            args.search_url,
            args.topk
        )

