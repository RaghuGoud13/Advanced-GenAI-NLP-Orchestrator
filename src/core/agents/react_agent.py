from typing import List, Dict, Any, Union, Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

class ReActAgent:
    """
    A ReAct-style agent implementation for multi-step reasoning and tool-calling.
    """
    def __init__(
        self, 
        llm: BaseChatModel, 
        tools: List[BaseTool], 
        prompt: Optional[PromptTemplate] = None,
        memory: Optional[ConversationBufferMemory] = None
    ):
        """
        Initialize the ReActAgent.
        
        Args:
            llm (BaseChatModel): The language model instance.
            tools (List[BaseTool]): List of tools the agent can use.
            prompt (PromptTemplate, optional): Custom prompt template.
            memory (ConversationBufferMemory, optional): Memory to store conversation history.
        """
        self.llm = llm
        self.tools = tools
        self.memory = memory or ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Default ReAct prompt if not provided
        self.prompt = prompt or self._get_default_prompt()
        
        # Create the ReAct agent
        self.agent = create_react_agent(llm, tools, self.prompt)
        
        # Create the executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10
        )

    def _get_default_prompt(self) -> PromptTemplate:
        """
        Retrieves the default ReAct prompt template.
        """
        template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

History: {chat_history}
Question: {input}
Thought: {agent_scratchpad}"""
        return PromptTemplate.from_template(template)

    def run(self, input_text: str) -> str:
        """
        Execute the agent with the given input.
        
        Args:
            input_text (str): The user query.
            
        Returns:
            str: The final answer from the agent.
        """
        try:
            response = self.agent_executor.invoke({"input": input_text})
            return response.get("output", "I was unable to find an answer.")
        except Exception as e:
            return f"Error executing agent: {str(e)}"

    async def arun(self, input_text: str) -> str:
        """
        Asynchronously execute the agent.
        
        Args:
            input_text (str): The user query.
            
        Returns:
            str: The final answer from the agent.
        """
        try:
            response = await self.agent_executor.ainvoke({"input": input_text})
            return response.get("output", "I was unable to find an answer.")
        except Exception as e:
            return f"Error executing agent: {str(e)}"
