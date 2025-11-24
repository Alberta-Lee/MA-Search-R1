"""
Multi-Agent Inference Script for MA-Search-R1.

This script demonstrates how to use the trained multi-agent system for inference.
"""

import transformers
import torch
import requests
import re


def get_queries_from_plan(text):
    """Extract queries from planner's plan."""
    pattern = re.compile(r"<plan>(.*?)</plan>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        queries = [q.strip() for q in re.split(r'[,;\n]', matches[-1]) if q.strip()]
        return queries
    return []


def search(query: str, search_url: str, topk: int = 3):
    """Call search API."""
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
    
    return _passages2string(results[0]) if results else ""


def infer_multi_agent(question: str, model_id: str, search_url: str = "http://127.0.0.1:8000/retrieve", topk: int = 3, max_turns: int = 3):
    """
    Run multi-agent inference.
    
    Args:
        question: The question to answer
        model_id: HuggingFace model ID
        search_url: Search API URL
        topk: Number of documents to retrieve
        max_turns: Maximum interaction turns
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare question
    question = question.strip()
    if question[-1] != '?':
        question += '?'
    
    # Multi-agent prompt template
    planner_prompt = f"""You are a Planner agent. Analyze the question and create a search plan.
Generate search queries that will help answer the question. Output your plan in the format:
<plan>query1, query2, query3</plan>

Question: {question}
"""
    
    synthesizer_prompt_template = """You are a Synthesizer agent. Based on the information provided, either:
1. Provide a final answer: <answer>your answer</answer>
2. Request more information: <need_more>explain what information is still needed</need_more>

Question: {question}
Information: {information}

Your response:"""
    
    # Initialize model and tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    print("\n" + "="*80)
    print("MA-Search-R1 Multi-Agent Inference")
    print("="*80)
    print(f"Question: {question}\n")
    
    # Step 1: Planner generates search plan
    print("[Planner] Generating search plan...")
    planner_input = tokenizer(planner_prompt, return_tensors='pt').to(device)
    planner_output = model.generate(
        planner_input['input_ids'],
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    planner_response = tokenizer.decode(planner_output[0][planner_input['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"[Planner] {planner_response}\n")
    
    queries = get_queries_from_plan(planner_response)
    if not queries:
        print("[Warning] No queries extracted from planner. Using question as query.")
        queries = [question]
    
    # Step 2: Researcher executes searches
    print("[Researcher] Executing searches...")
    all_search_results = []
    for query in queries:
        print(f"  Searching: {query}")
        results = search(query, search_url, topk)
        all_search_results.append(results)
        print(f"  Retrieved {len(results.split('Doc'))-1} documents")
    
    combined_information = "\n\n".join(all_search_results)
    print()
    
    # Step 3: Synthesizer synthesizes and decides
    for turn in range(max_turns):
        print(f"[Synthesizer] Turn {turn + 1}...")
        synthesizer_prompt = synthesizer_prompt_template.format(
            question=question,
            information=combined_information
        )
        
        synthesizer_input = tokenizer(synthesizer_prompt, return_tensors='pt').to(device)
        synthesizer_output = model.generate(
            synthesizer_input['input_ids'],
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        synthesizer_response = tokenizer.decode(
            synthesizer_output[0][synthesizer_input['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        print(f"[Synthesizer] {synthesizer_response}\n")
        
        # Check if final answer
        if '<answer>' in synthesizer_response:
            answer_pattern = r'<answer>(.*?)</answer>'
            answer_match = re.search(answer_pattern, synthesizer_response, re.DOTALL)
            if answer_match:
                final_answer = answer_match.group(1).strip()
                print("="*80)
                print(f"Final Answer: {final_answer}")
                print("="*80)
                return final_answer
        
        # Check if needs more information
        if '<need_more>' in synthesizer_response and turn < max_turns - 1:
            need_more_pattern = r'<need_more>(.*?)</need_more>'
            need_more_match = re.search(need_more_pattern, synthesizer_response, re.DOTALL)
            if need_more_match:
                reason = need_more_match.group(1).strip()
                print(f"[Synthesizer] Needs more information: {reason}")
                print("[Planner] Generating additional search queries...")
                
                # Planner generates additional queries
                additional_prompt = f"""Based on the synthesizer's feedback, generate additional search queries.
Previous queries: {', '.join(queries)}
Feedback: {reason}
Generate new queries: <plan>query1, query2</plan>
"""
                additional_input = tokenizer(additional_prompt, return_tensors='pt').to(device)
                additional_output = model.generate(
                    additional_input['input_ids'],
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
                additional_response = tokenizer.decode(
                    additional_output[0][additional_input['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                print(f"[Planner] {additional_response}\n")
                
                new_queries = get_queries_from_plan(additional_response)
                if new_queries:
                    queries.extend(new_queries)
                    
                    # Execute new searches
                    print("[Researcher] Executing additional searches...")
                    for query in new_queries:
                        print(f"  Searching: {query}")
                        results = search(query, search_url, topk)
                        all_search_results.append(results)
                        print(f"  Retrieved {len(results.split('Doc'))-1} documents")
                    
                    combined_information = "\n\n".join(all_search_results)
                    print()
        else:
            break
    
    print("="*80)
    print("Max turns reached. Final response:")
    print(synthesizer_response)
    print("="*80)
    return synthesizer_response


if __name__ == '__main__':
    # Example question
    question = "Mike Barnett negotiated many contracts including which player that went on to become general manager of CSKA Moscow of the Kontinental Hockey League?"
    
    # Model ID (use your trained model)
    model_id = "meta-llama/Llama-3.2-3B"  # Replace with your trained model
    
    # Search API URL
    search_url = "http://127.0.0.1:8000/retrieve"
    
    # Run inference
    infer_multi_agent(question, model_id, search_url)

