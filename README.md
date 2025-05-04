# LangGraph Research Assistant

An intelligent research assistant built using LangGraph and LangChain that automates the process of conducting research, generating analyst personas, and creating comprehensive reports.

## Features

- **Multi-Agent Research System**: Utilizes LangGraph to create intelligent agents that can conduct research and generate insights
- **Analyst Persona Generation**: Automatically generates expert analyst personas based on research topics
- **Interactive Interview System**: Conducts structured interviews with analyst personas to gather insights
- **Report Generation**: Creates comprehensive research reports with sections, introductions, and conclusions
- **Web and Wikipedia Integration**: Uses Tavily Search and Wikipedia for comprehensive research
- **Memory Management**: Maintains conversation context and memory across interactions

## Requirements

- Python 3.13
- OpenAI API Key
- Tavily API Key
- Required Python packages (see `requirements.txt`)

## Setup

1. Create a `.env` file with your API keys:
```env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from backend import generate_analyst_personas, conduct_interview

# Generate analyst personas
analyst = generate_analyst_personas("LangGraph adoption benefits", 1)[0]

# Conduct interview
state = conduct_interview("LangGraph adoption benefits", analyst, max_turns=1)

# Access generated sections
print(state["sections"][0])
```

### Components

1. **Analyst Persona Generation**
   - Creates expert analyst personas based on research topics
   - Generates multiple perspectives on a given topic
   - Supports human feedback for persona refinement

2. **Interview System**
   - Conducts structured interviews with generated analysts
   - Uses web search and Wikipedia for research
   - Maintains conversation context
   - Generates sections based on interview insights

3. **Report Generation**
   - Creates comprehensive research reports
   - Includes introduction, content sections, and conclusion
   - Maintains consistent formatting
   - Supports multiple analyst perspectives

## Technical Details

- **Framework**: LangGraph with LangChain integration
- **LLM**: OpenAI GPT-4
- **Search**: Tavily Search and Wikipedia integration
- **State Management**: Custom StateGraph implementation
