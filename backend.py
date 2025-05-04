from __future__ import annotations

import os
from typing import TypedDict, List, Dict, Any, Tuple
from langgraph.constants import Send
from dotenv import load_dotenv
load_dotenv()  # load variables from .env, if present

# -------------------------------------------------------------------------
# Mandatory API keys (no placeholders, no prompting)
# -------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY – add it to .env or export it.")
if not TAVILY_API_KEY:
    raise RuntimeError("Missing TAVILY_API_KEY – add it to .env or export it.")

# LangChain / LangGraph imports AFTER env vars are set --------------------
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# -------------------------------------------------------------------------
# Analyst personas data models (from notebook)
# -------------------------------------------------------------------------
from pydantic import BaseModel, Field

class Analyst(BaseModel):
    affiliation: str = Field(description="Primary affiliation of the analyst.")
    name: str = Field(description="Name of the analyst.")
    role: str = Field(description="Role of the analyst in the context of the topic.")
    description: str = Field(description="Description of the analyst focus, concerns, and motives.")

    @property
    def persona(self) -> str:  # nicely formatted persona string
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"

class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(description="Comprehensive list of analysts with their roles and affiliations.")

class GenerateAnalystsState(TypedDict):
    topic: str  # research topic
    max_analysts: int  # desired number of analyst personas
    human_analyst_feedback: str  # feedback supplied by the human user
    analysts: List[Analyst]  # generated analyst personas

# -------------------------------------------------------------------------
# LangGraph state definition
# -------------------------------------------------------------------------
class QAState(TypedDict):
    question: str
    search_query: str
    search_results: List[Any]  # Tavily may return str or dict
    answer: str

# -------------------------------------------------------------------------
# Shared singletons – survive Streamlit hot‑reloads
# -------------------------------------------------------------------------
_llm = ChatOpenAI(model="gpt-4o", temperature=0)
_search_tool = TavilySearchResults(max_results=5)

# -------------------------------------------------------------------------
# Node 1 – craft concise web‑search query
# -------------------------------------------------------------------------
search_query_system_prompt = SystemMessage(
    content="""You are an expert research assistant. Rewrite the user's question as ONE concise web‑search query."""
)

def create_search_query(state: QAState) -> QAState:
    q = state["question"]
    messages = [search_query_system_prompt, HumanMessage(content=q)]
    search_query = _llm.invoke(messages).content.strip()
    state["search_query"] = search_query
    return state

# -------------------------------------------------------------------------
# Node 2 – call Tavily
# -------------------------------------------------------------------------

def run_search(state: QAState) -> QAState:
    query = state["search_query"]
    docs = _search_tool.invoke(query)
    state["search_results"] = docs
    return state

# -------------------------------------------------------------------------
# Helper to format search docs into plain text
# -------------------------------------------------------------------------

def _format_doc(d: Any) -> str:
    if isinstance(d, dict):
        url = d.get("url") or d.get("link") or ""
        content = d.get("content") or d.get("snippet") or str(d)
        return f"[{url}] {content}" if url else content
    return str(d)

# -------------------------------------------------------------------------
# Node 3 – draft the final answer
# -------------------------------------------------------------------------
answer_system_prompt = SystemMessage(
    content="""You are a helpful expert.
Answer the question in markdown using the provided web snippets for factual support.
Add a 'Sources' section listing any URLs you referenced."""
)

def generate_answer(state: QAState) -> QAState:
    q = state["question"]
    docs = state.get("search_results", [])
    docs_text = "\n\n".join(_format_doc(d) for d in docs)
    messages = [answer_system_prompt, HumanMessage(content=f"Question: {q}\n\nContext:\n{docs_text}")]
    answer = _llm.invoke(messages).content.strip()
    state["answer"] = answer
    return state

# -------------------------------------------------------------------------
# Build LangGraph DAG (node IDs distinct from state keys)
# -------------------------------------------------------------------------
_graph = StateGraph(QAState)
_graph.add_node("create_query", create_search_query)
_graph.add_node("run_search", run_search)
_graph.add_node("draft_answer", generate_answer)

_graph.add_edge(START, "create_query")
_graph.add_edge("create_query", "run_search")
_graph.add_edge("run_search", "draft_answer")
_graph.add_edge("draft_answer", END)

qa_graph = _graph.compile()

# -------------------------------------------------------------------------
# Analyst‑persona generation subgraph
# -------------------------------------------------------------------------
from langgraph.checkpoint.memory import MemorySaver

analyst_instructions = (
    """You are tasked with creating a set of AI analyst personas.

"
    "1. First, review the research topic:
{topic}

"
    "2. Examine any editorial feedback that has been optionally provided to guide creation of the analysts:

"
    "{human_analyst_feedback}

"
    "3. Determine the most interesting themes based upon the documents and/or feedback above.
"
    "4. Pick the top {max_analysts} themes.
"
    "5. Assign one analyst to each theme."""
)

def _create_analysts(state: GenerateAnalystsState):
    """LLM generates analyst personas (structured)."""
    structured_llm = _llm.with_structured_output(Perspectives)
    sys_msg = analyst_instructions.format(
        topic=state["topic"],
        human_analyst_feedback=state.get("human_analyst_feedback", ""),
        max_analysts=state["max_analysts"],
    )
    perspectives: Perspectives = structured_llm.invoke(
        [SystemMessage(content=sys_msg), HumanMessage(content="Generate the set of analysts.")]
    )
    return {"analysts": perspectives.analysts}

def _human_feedback(state: GenerateAnalystsState):
    """Placeholder node for potential human feedback loop."""
    return {}

def _should_continue(state: GenerateAnalystsState):
    return "_create_analysts" if state.get("human_analyst_feedback") else END

_analyst_builder = StateGraph(GenerateAnalystsState)
_analyst_builder.add_node("_create_analysts", _create_analysts)
_analyst_builder.add_node("_human_feedback", _human_feedback)
_analyst_builder.add_edge(START, "_create_analysts")
_analyst_builder.add_edge("_create_analysts", "_human_feedback")
_analyst_builder.add_conditional_edges("_human_feedback", _should_continue, ["_create_analysts", END])

_analyst_graph = _analyst_builder.compile()

def generate_analyst_personas(topic: str, max_analysts: int = 5, human_analyst_feedback: str = "") -> list[Analyst]:
    """Public helper to obtain analyst personas list."""
    state = _analyst_graph.invoke(
        {
            "topic": topic,
            "max_analysts": max_analysts,
            "human_analyst_feedback": human_analyst_feedback,
        }
    )
    return state.get("analysts", [])

# -------------------------------------------------------------------------
# Interview‑style conversation data models (from notebook)
# -------------------------------------------------------------------------
import operator
from typing import Annotated
from langgraph.graph import MessagesState

class InterviewState(MessagesState):
    """Extended state for an analyst‑led interview loop."""
    max_num_turns: int  # total allowed turns
    context: Annotated[list, operator.add]  # retrieved docs or background
    analyst: Analyst  # the persona conducting the interview
    interview: str  # running transcript
    sections: list  # extracted sections/final summary

class SearchQuery(BaseModel):
    search_query: str | None = Field(None, description="Search query for retrieval.")

# -------------------------------------------------------------------------
# Interview question‑generation utility (from notebook)
# -------------------------------------------------------------------------
question_instructions = (
    """You are an analyst tasked with interviewing an expert to learn about a specific topic.

"
    "Your goal is to boil down to interesting and specific insights related to your topic.

"
    "1. **Interesting**: Insights that people will find surprising or non‑obvious.
"
    "2. **Specific**: Insights that avoid generalities and include concrete examples.

"
    "Here is your topic of focus and set of goals: {goals}

"
    "Begin by introducing yourself using a name that fits your persona, and then ask your question.

"
    "Continue to ask questions to drill down and refine your understanding of the topic.

"
    "When you are satisfied with your understanding, complete the interview with: \"Thank you so much for your help!\"

"
    "Remember to stay in character throughout your response, reflecting the persona and goals provided to you."""
)

def generate_next_question(state: InterviewState):
    """LangGraph node: produce the next interview question from the analyst persona."""
    analyst: Analyst = state["analyst"]
    messages: list = state.get("messages", [])  # prior dialogue

    sys_msg = question_instructions.format(goals=analyst.persona)
    question_msg = _llm.invoke([SystemMessage(content=sys_msg)] + messages)

    # append the new question to the running transcript
    return {"messages": [question_msg]}

def generate_question(state: InterviewState):
    analyst: Analyst = state["analyst"]
    messages = state.get("messages", [])
    sys = question_instructions.format(goals=analyst.persona)
    q_msg = _llm.invoke([SystemMessage(content=sys)] + messages)
    return {"messages": [q_msg]}

# ---------------- Retrieval helpers ----------------
from langchain_community.document_loaders import WikipediaLoader

search_instructions = SystemMessage(content=f"""You will be given a conversation between an analyst and an expert. 

Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.
        
First, analyze the full conversation.

Pay particular attention to the final question posed by the analyst.

Convert this final question into a well-structured web search query""")

def search_web(state: InterviewState):
    
    """ Retrieve docs from web search """

    # Search query
    structured_llm = _llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_instructions]+state['messages'])
    
    # Search
    search_docs = _search_tool.invoke(search_query.search_query)

     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )
    
    return {"context": [formatted_search_docs]} 

def search_wikipedia(state: InterviewState):
    
    """ Retrieve docs from wikipedia """

    # Search query
    structured_llm = _llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_instructions]+state['messages'])
    
    # Search
    search_docs = WikipediaLoader(query=search_query.search_query, 
                                  load_max_docs=2).load()

     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]} 

# ---------------- Answer helper ----------------
answer_instructions = """You are an expert being interviewed by an analyst.

Here is analyst area of focus: {goals}. 
        
You goal is to answer a question posed by the interviewer.

To answer question, use this context:
        
{context}

When answering questions, follow these guidelines:
        
1. Use only the information provided in the context. 
        
2. Do not introduce external information or make assumptions beyond what is explicitly stated in the context.

3. The context contain sources at the topic of each individual document.

4. Include these sources your answer next to any relevant statements. For example, for source # 1 use [1]. 

5. List your sources in order at the bottom of your answer. [1] Source 1, [2] Source 2, etc
        
6. If the source is: <Document source="assistant/docs/llama3_1.pdf" page="7"/>' then just list: 
        
[1] assistant/docs/llama3_1.pdf, page 7 
        
And skip the addition of the brackets as well as the Document source preamble in your citation."""

def generate_answer(state: InterviewState):
    
    """ Node to answer a question """

    # Get state
    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]

    # Answer question
    system_message = answer_instructions.format(goals=analyst.persona, context=context)
    answer = _llm.invoke([SystemMessage(content=system_message)]+messages)
            
    # Name the message as coming from the expert
    answer.name = "expert"
    
    # Append it to state
    return {"messages": [answer]}

# ---------------- Save & route ----------------
from langchain_core.messages import get_buffer_string

def save_interview(state: InterviewState):
    
    """ Save interviews """

    # Get messages
    messages = state["messages"]
    
    # Convert interview to a string
    interview = get_buffer_string(messages)
    
    # Save to interviews key
    return {"interview": interview}

def route_messages(state: InterviewState, 
                   name: str = "expert"):

    """ Route between question and answer """
    
    # Get messages
    messages = state["messages"]
    max_num_turns = state.get('max_num_turns',2)

    # Check the number of expert answers 
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )

    # End if expert has answered more than the max turns
    if num_responses >= max_num_turns:
        return 'save_interview'

    # This router is run after each question - answer pair 
    # Get the last question asked to check if it signals the end of discussion
    last_question = messages[-2]
    
    if "Thank you so much for your help" in last_question.content:
        return 'save_interview'
    return "ask_question"


# ---------------- Section writer ----------------
section_writer_instructions = """You are an expert technical writer. 
            
Your task is to create a short, easily digestible section of a report based on a set of source documents.

1. Analyze the content of the source documents: 
- The name of each source document is at the start of the document, with the <Document tag.
        
2. Create a report structure using markdown formatting:
- Use ## for the section title
- Use ### for sub-section headers
        
3. Write the report following this structure:
a. Title (## header)
b. Summary (### header)
c. Sources (### header)

4. Make your title engaging based upon the focus area of the analyst: 
{focus}

5. For the summary section:
- Set up summary with general background / context related to the focus area of the analyst
- Emphasize what is novel, interesting, or surprising about insights gathered from the interview
- Create a numbered list of source documents, as you use them
- Do not mention the names of interviewers or experts
- Aim for approximately 400 words maximum
- Use numbered sources in your report (e.g., [1], [2]) based on information from source documents
        
6. In the Sources section:
- Include all sources used in your report
- Provide full links to relevant websites or specific document paths
- Separate each source by a newline. Use two spaces at the end of each line to create a newline in Markdown.
- It will look like:

### Sources
[1] Link or Document name
[2] Link or Document name

7. Be sure to combine sources. For example this is not correct:

[3] https://ai.meta.com/blog/meta-llama-3-1/
[4] https://ai.meta.com/blog/meta-llama-3-1/

There should be no redundant sources. It should simply be:

[3] https://ai.meta.com/blog/meta-llama-3-1/
        
8. Final review:
- Ensure the report follows the required structure
- Include no preamble before the title of the report
- Check that all guidelines have been followed"""

def write_section(state: InterviewState):

    """ Node to answer a question """

    # Get state
    interview = state["interview"]
    context = state["context"]
    analyst = state["analyst"]
   
    # Write section using either the gathered source docs from interview (context) or the interview itself (interview)
    system_message = section_writer_instructions.format(focus=analyst.description)
    section = _llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Use this source to write your section: {context}")]) 
                
    # Append it to state
    return {"sections": [section.content]}

# ---------------- Interview graph ----------------
# Add nodes and edges 
interview_builder = StateGraph(InterviewState)
interview_builder.add_node("ask_question", generate_question)
interview_builder.add_node("search_web", search_web)
interview_builder.add_node("search_wikipedia", search_wikipedia)
interview_builder.add_node("answer_question", generate_answer)
interview_builder.add_node("save_interview", save_interview)
interview_builder.add_node("write_section", write_section)

# Flow
interview_builder.add_edge(START, "ask_question")
interview_builder.add_edge("ask_question", "search_web")
interview_builder.add_edge("ask_question", "search_wikipedia")
interview_builder.add_edge("search_web", "answer_question")
interview_builder.add_edge("search_wikipedia", "answer_question")
interview_builder.add_conditional_edges("answer_question", route_messages,['ask_question','save_interview'])
interview_builder.add_edge("save_interview", "write_section")
interview_builder.add_edge("write_section", END)

interview_graph = interview_builder.compile()

# Interview 
memory = MemorySaver()
interview_graph = interview_builder.compile().with_config(run_name="Conduct Interviews")

def conduct_interview(topic: str, analyst: Analyst, max_turns: int = 2):
    state = interview_graph.invoke({
        "messages": [],
        "analyst": analyst,
        "context": [],
        "max_num_turns": max_turns,
        "sections": [],
    })
    return state

# -------------------------------------------------------------------------
# Interview models
# -------------------------------------------------------------------------
class ResearchGraphState(TypedDict):
    """High‑level orchestration state for the full research pipeline."""
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[Analyst]
    sections: Annotated[list, operator.add]
    introduction: str
    content: str
    conclusion: str
    final_report: str
    
def initiate_all_interviews(state: ResearchGraphState):
    """ This is the "map" step where we run each interview sub-graph using Send API """    

    # Check if human feedback
    human_analyst_feedback=state.get('human_analyst_feedback')
    if human_analyst_feedback:
        # Return to _create_analysts
        return "_create_analysts"

    # Otherwise kick off interviews in parallel via Send() API
    else:
        topic = state["topic"]
        return [Send("conduct_interview", {"analyst": analyst,
                                           "messages": [HumanMessage(
                                               content=f"So you said you were writing an article on {topic}?"
                                           )
                                                       ]}) for analyst in state["analysts"]]

report_writer_instructions = """You are a technical writer creating a report on this overall topic: 

{topic}
    
You have a team of analysts. Each analyst has done two things: 

1. They conducted an interview with an expert on a specific sub-topic.
2. They write up their finding into a memo.

Your task: 

1. You will be given a collection of memos from your analysts.
2. Think carefully about the insights from each memo.
3. Consolidate these into a crisp overall summary that ties together the central ideas from all of the memos. 
4. Summarize the central points in each memo into a cohesive single narrative.

To format your report:
 
1. Use markdown formatting. 
2. Include no pre-amble for the report.
3. Use no sub-heading. 
4. Start your report with a single title header: ## Insights
5. Do not mention any analyst names in your report.
6. Preserve any citations in the memos, which will be annotated in brackets, for example [1] or [2].
7. Create a final, consolidated list of sources and add to a Sources section with the `## Sources` header.
8. List your sources in order and do not repeat.

[1] Source 1
[2] Source 2

Here are the memos from your analysts to build your report from: 

{context}"""

def write_report(state: ResearchGraphState):
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    # Summarize the sections into a final report
    system_message = report_writer_instructions.format(topic=topic, context=formatted_str_sections)    
    report = _llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Write a report based upon these memos.")]) 
    return {"content": report.content}

intro_conclusion_instructions = """You are a technical writer finishing a report on {topic}

You will be given all of the sections of the report.

You job is to write a crisp and compelling introduction or conclusion section.

The user will instruct you whether to write the introduction or conclusion.

Include no pre-amble for either section.

Target around 100 words, crisply previewing (for introduction) or recapping (for conclusion) all of the sections of the report.

Use markdown formatting. 

For your introduction, create a compelling title and use the # header for the title.

For your introduction, use ## Introduction as the section header. 

For your conclusion, use ## Conclusion as the section header.

Here are the sections to reflect on for writing: {formatted_str_sections}"""

def write_introduction(state: ResearchGraphState):
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    # Summarize the sections into a final report
    
    instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)    
    intro = _llm.invoke([instructions]+[HumanMessage(content=f"Write the report introduction")]) 
    return {"introduction": intro.content}

def write_conclusion(state: ResearchGraphState):
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    # Summarize the sections into a final report
    
    instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)    
    conclusion = _llm.invoke([instructions]+[HumanMessage(content=f"Write the report conclusion")]) 
    return {"conclusion": conclusion.content}

def finalize_report(state: ResearchGraphState):
    """ The is the "reduce" step where we gather all the sections, combine them, and reflect on them to write the intro/conclusion """
    # Save full final report
    content = state["content"]
    if content.startswith("## Insights"):
        content = content.strip("## Insights")
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except:
            sources = None
    else:
        sources = None

    final_report = state["introduction"] + "\n\n---\n\n" + content + "\n\n---\n\n" + state["conclusion"]
    if sources is not None:
        final_report += "\n\n## Sources\n" + sources
    return {"final_report": final_report}

# Add nodes and edges 
builder = StateGraph(ResearchGraphState)
builder.add_node("_create_analysts", _create_analysts)
builder.add_node("_human_feedback", _human_feedback)
builder.add_node("conduct_interview", interview_builder.compile())
builder.add_node("write_report",write_report)
builder.add_node("write_introduction",write_introduction)
builder.add_node("write_conclusion",write_conclusion)
builder.add_node("finalize_report",finalize_report)

# Logic
builder.add_edge(START, "_create_analysts")
builder.add_edge("_create_analysts", "_human_feedback")
builder.add_conditional_edges("_human_feedback", initiate_all_interviews, ["_create_analysts", "conduct_interview"])
builder.add_edge("conduct_interview", "write_report")
builder.add_edge("conduct_interview", "write_introduction")
builder.add_edge("conduct_interview", "write_conclusion")
builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
builder.add_edge("finalize_report", END)

# Compile
memory = MemorySaver()
graph = builder.compile()

# -------------------------------------------------------------------------
# Public helper used by Streamlit
# -------------------------------------------------------------------------

def get_answer(question: str) -> Tuple[str, List[Any]]:
    """Return (markdown_answer, raw_search_results)."""
    state = qa_graph.invoke({"question": question})
    return state["answer"], state.get("search_results", [])

# -------------------------------------------------------------------------
# CLI usage
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python backend.py \"YOUR QUESTION\"")
        sys.exit(1)
    ans, refs = get_answer(" ".join(sys.argv[1:]))
    print(ans)
    if refs:
        print("\nSources:")
        for d in refs:
            print(_format_doc(d))
