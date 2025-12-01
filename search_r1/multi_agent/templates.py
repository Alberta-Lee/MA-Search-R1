"""
Multi-agent prompt templates.
Extends single-agent templates with role-based structure.
"""

from typing import List, Optional


class MultiAgentTemplate:
    """
    Template generator for multi-agent search and reasoning.
    """
    
    @staticmethod
    def get_single_agent_template(question: str) -> str:
        """
        Get single-agent template (original Search-R1 format).
        
        Args:
            question: User question
            
        Returns:
            Formatted prompt string
        """
        return f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
    
    @staticmethod
    def get_multi_agent_template(
        question: str,
        num_explorers: int = 2,
        roles: Optional[List[str]] = None
    ) -> str:
        """
        Get multi-agent template with role-based structure.
        
        Args:
            question: User question
            num_explorers: Number of Explorer agents
            roles: List of role names (default: ["Planner", "Explorer", "Synthesizer"])
            
        Returns:
            Formatted prompt string
        """
        if roles is None:
            roles = ["Planner", "Explorer", "Synthesizer"]
        
        explorer_names = [f"Explorer_{i+1}" for i in range(num_explorers)]
        
        template = f"""You are working in a multi-agent team to answer the following question: {question}

Your team consists of:
- **Planner**: Analyzes the question, breaks it down into sub-questions, and plans the search strategy.
- **Explorer_1, Explorer_2, ...**: Each Explorer independently searches for information related to specific sub-questions.
- **Synthesizer**: Synthesizes all collected information and provides the final answer.

Workflow:
1. **Planner** starts by analyzing the question and breaking it down:
   <agent=Planner>
   <think>Analyze the question and identify what information is needed...</think>
   </agent>

2. **Explorers** search for information (each can search multiple times):
   **CRITICAL**: Each Explorer MUST perform at least one search. After reasoning, if you find you lack some knowledge to answer your assigned sub-question, you MUST call a search engine by <search>query</search> and it will return the top searched results between <information> and </information>. You can search as many times as you want. Do NOT skip searching - searching is mandatory for Explorers.
   
   <agent=Explorer_1>
   <think>Reason about what to search...</think>
   <search>query for sub-question 1</search>
   </agent>
   
   <information agent=Explorer_1>Search results will appear here...</information>
   
   <agent=Explorer_2>
   <think>Reason about what to search...</think>
   <search>query for sub-question 2</search>
   </agent>
   
   <information agent=Explorer_2>Search results will appear here...</information>

3. **Synthesizer** provides the final answer:
   <agent=Synthesizer>
   <think>Reason about the answer based on all collected information...</think>
   <answer>Final answer</answer>
   </agent>

Important:
- Each agent must wrap their output in <agent=RoleName>...</agent> tags
- Use <think>...</think> for reasoning
- **Explorers MUST use <search>query</search> to search for information. If you lack knowledge, you MUST search.**
- Use <answer>...</answer> for final answer (Synthesizer only)
- Search results will be automatically inserted as <information agent=RoleName>...</information>
- After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as you want.

Now begin with the Planner's analysis:
"""
        return template
    
    @staticmethod
    def get_role_instruction(role: str) -> str:
        """
        Get instruction for a specific role.
        
        Args:
            role: Role name (Planner, Explorer_N, Synthesizer)
            
        Returns:
            Role-specific instruction string
        """
        if role == "Planner":
            return """You are the Planner. Your task is to:
1. Analyze the question carefully
2. Break it down into sub-questions if needed
3. Plan the search strategy
4. You do NOT perform searches yourself, but guide the Explorers

**CRITICAL**: You MUST wrap your entire response in <agent=Planner>...</agent> tags.

Format your response EXACTLY as:
<agent=Planner>
<think>Your analysis and planning...</think>
</agent>

Start your response immediately with <agent=Planner>"""
        
        elif role.startswith("Explorer"):
            return f"""You are {role}. Your task is to:
1. Focus on specific sub-questions assigned by the Planner
2. **MUST search for relevant information** - searching is mandatory, not optional
3. After reasoning, if you find you lack some knowledge, you MUST call a search engine by <search>query</search>
4. You can search multiple times if needed

**CRITICAL**: You MUST wrap your entire response in <agent={role}>...</agent> tags AND include at least one <search>query</search>.

Format your response EXACTLY as:
<agent={role}>
<think>Your reasoning about what to search...</think>
<search>your search query</search>
</agent>

Start your response immediately with <agent={role}> and include <search>query</search> inside."""
        
        elif role == "Synthesizer":
            return """You are the Synthesizer. Your task is to:
1. Review all information collected by the Explorers
2. Synthesize the information
3. Provide the final answer

**CRITICAL**: You MUST wrap your entire response in <agent=Synthesizer>...</agent> tags.

Format your response EXACTLY as:
<agent=Synthesizer>
<think>Your reasoning about the answer...</think>
<answer>Final answer</answer>
</agent>

Start your response immediately with <agent=Synthesizer>"""
        
        else:
            return f"Role {role} instruction not defined."

