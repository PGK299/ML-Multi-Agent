import os
import logging
import google.cloud.logging

from callback_logging import log_query_to_model, log_model_response
from dotenv import load_dotenv

from google.adk import Agent
from google.adk.agents import SequentialAgent, LoopAgent, ParallelAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.langchain_tool import LangchainTool
from google.adk.models import Gemini
from google.genai import types

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from google.adk.tools import exit_loop

# Setup Logging & Env
cloud_logging_client = google.cloud.logging.Client()
cloud_logging_client.setup_logging()

load_dotenv()

model_name = os.getenv("MODEL")
print(f"Using Model: {model_name}")

RETRY_OPTIONS = types.HttpRetryOptions(initial_delay=1, attempts=6)

def append_to_state(
    tool_context: ToolContext, field: str, response: str
) -> dict[str, str]:

    existing_state = tool_context.state.get(field, [])
    tool_context.state[field] = existing_state + [response]
    logging.info(f"[Added to {field}] {response}")
    return {"status": "success"}

def write_file(
    tool_context: ToolContext,
    directory: str,
    filename: str,
    content: str
) -> dict[str, str]:

    target_path = os.path.join(directory, filename)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(content)
    logging.info(f"[File Written] {target_path}")
    return {"status": "success"}

wiki_tool = LangchainTool(tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()))


admirer_agent = Agent(
    name="admirer",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="Researches the positive achievements and historical contributions.",
    instruction="""
    TOPIC:
    { TOPIC? }

    JUDGE_FEEDBACK:
    { JUDGE_FEEDBACK? }

    INSTRUCTIONS:
    You are 'The Admirer'. Your job is to find the positive achievements, successes, and great contributions of the TOPIC.
    - Use your Wikipedia tool to search. 
    - CRITICAL: Add specific keywords to your search query like "{TOPIC} achievements", "{TOPIC} positive impact", or "{TOPIC} successful campaigns" to get the right branch of information.
    - If JUDGE_FEEDBACK requests more specific positive details, search for those specifically.
    - Once you have gathered the facts, use the 'append_to_state' tool to save your findings to the field 'pos_data'.
    - Summarize what you found briefly.
    """,
    generate_content_config=types.GenerateContentConfig(temperature=0.2),
    tools=[wiki_tool, append_to_state],
)


critic_agent = Agent(
    name="critic",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="Researches the controversies, failures, and negative aspects.",
    instruction="""
    TOPIC:
    { TOPIC? }

    JUDGE_FEEDBACK:
    { JUDGE_FEEDBACK? }

    INSTRUCTIONS:
    You are 'The Critic'. Your job is to find the controversies, mistakes, failures, and negative criticisms of the TOPIC.
    - Use your Wikipedia tool to search.
    - CRITICAL: Add specific keywords to your search query like "{TOPIC} controversy", "{TOPIC} failures", "{TOPIC} criticism", or "{TOPIC} human rights violations" to get the right branch of information.
    - If JUDGE_FEEDBACK requests more specific negative details, search for those specifically.
    - Once you have gathered the facts, use the 'append_to_state' tool to save your findings to the field 'neg_data'.
    - Summarize what you found briefly.
    """,
    generate_content_config=types.GenerateContentConfig(temperature=0.2),
    tools=[wiki_tool, append_to_state],
)


investigation_team = ParallelAgent(
    name="investigation_team",
    description="Runs Admirer and Critic in parallel to gather both sides of the history.",
    sub_agents=[
        admirer_agent,
        critic_agent
    ]
)


judge_agent = Agent(
    name="judge",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="Evaluates the gathered evidence for balance and depth.",
    instruction="""
    TOPIC:
    { TOPIC? }

    POSITIVE_EVIDENCE (Admirer):
    { pos_data? }

    NEGATIVE_EVIDENCE (Critic):
    { neg_data? }

    INSTRUCTIONS:
    You are 'The Judge'. Your job is to review the POSITIVE_EVIDENCE and NEGATIVE_EVIDENCE to ensure a fair and balanced trial.
    1. Check if both sides have sufficient detail to write a comprehensive historical report.
    2. Check if one side is severely lacking compared to the other.

    - If the evidence is balanced and complete enough: MUST use the 'exit_loop' tool to end the investigation phase.
    - If the evidence is lacking or unbalanced: Use the 'append_to_state' tool to add specific instructions to 'JUDGE_FEEDBACK' detailing what the Admirer or Critic needs to search for next. DO NOT use 'exit_loop' in this case.
    
    Explain your reasoning before calling any tools.
    """,
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
    generate_content_config=types.GenerateContentConfig(temperature=0),
    tools=[append_to_state, exit_loop]
)


trial_and_review = LoopAgent(
    name="trial_and_review",
    description="Iterates between investigation and judging until evidence is balanced.",
    sub_agents=[
        investigation_team,
        judge_agent
    ],
    max_iterations=4,
)


verdict_writer = Agent(
    name="verdict_writer",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="Writes the final neutral historical report and saves it.",
    instruction="""
    TOPIC:
    { TOPIC? }

    POSITIVE_EVIDENCE:
    { pos_data? }

    NEGATIVE_EVIDENCE:
    { neg_data? }

    INSTRUCTIONS:
    You are the 'Court Reporter'. Write a comprehensive, highly objective, and neutral historical report about the TOPIC.
    - Compare and contrast the positive and negative aspects.
    - Conclude with a balanced summary of their historical impact.
    
    Use the 'write_file' tool to save the report:
    - directory: 'historical_verdicts'
    - filename: '{TOPIC}_verdict.txt' (replace spaces with underscores)
    - content: The full report you just generated.
    """,
    generate_content_config=types.GenerateContentConfig(temperature=0.2),
    tools=[write_file],
)


historical_court_system = SequentialAgent(
    name="historical_court_system",
    description="The main sequential flow from investigation to verdict.",
    sub_agents=[
        trial_and_review,
        verdict_writer
    ],
)


root_agent = Agent(
    name="court_clerk",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="Greets the user and gets the historical topic.",
    instruction="""
    - Greet the user to 'The Historical Court'.
    - Ask them for a historical figure or event they want to put on trial (e.g., Genghis Khan, The Cold War).
    - When they respond, use the 'append_to_state' tool to store their response in the 'TOPIC' state key.
    - Then the system will automatically transfer to the 'historical_court_system'.
    """,
    generate_content_config=types.GenerateContentConfig(temperature=0),
    tools=[append_to_state],
    sub_agents=[historical_court_system],
)