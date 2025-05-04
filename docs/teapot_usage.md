# Teapot Usage in KAIA

## Overview

Teapot is a lightweight AI assistant model specialized for KAIA setup and configuration. The MultiturnTeapot class provides a chat interface with Retrieval-Augmented Generation (RAG) capabilities over a local FAQ file, making it ideal for helping users navigate KAIA functionality.

## Features

- Multi-turn conversation capabilities
- Retrieval-Augmented Generation (RAG) from local FAQ documents
- Efficient operation on CPU-only systems
- Documentation lookup functionality
- Compact model size (~1GB)

## Requirements

- Python 3.11+
- `teapotai` package (installed automatically with KAIA)
- Minimum 4GB RAM
- No GPU required (CPU-only operation)

## Basic Usage

```python
from killeraiagent.models.transformers.multiturn_teapot import MultiturnTeapot

# Initialize the Teapot chat agent
teapot = MultiturnTeapot()

# Start a conversation
response = teapot.chat("How do I install KAIA?")
print(response)

# Continue the conversation
response = teapot.chat("How can I enable GPU acceleration?")
print(response)

# When finished
teapot.close()
```

## CLI Usage

KAIA provides a convenient CLI wrapper for Teapot:

```bash
# Start a Teapot chat session
kaia-cli teapot-chat

# Ask a specific question
kaia-cli teapot-chat --query "How do I download models?"
```

## Configuration Options

When initializing MultiturnTeapot, you can customize its behavior:

```python
teapot = MultiturnTeapot(
    faq_path="/path/to/custom_faq.jsonl",     # Path to custom FAQ file
    system_prompt="Custom system prompt",      # Custom system prompt
    max_context_tokens=2048,                   # Maximum context window size
    rag_top_k=3                                # Number of FAQ snippets to retrieve
)
```

## Documentation Lookup

Teapot includes a built-in documentation lookup feature. Users can access this by starting their queries with "doc" or "man":

```python
# Look up documentation for llama integration
response = teapot.chat("doc llama")

# Look up setup documentation
response = teapot.chat("man setup")
```

Available documentation topics:
- `llama`: LlamaCPP usage documentation
- `setup`: KAIA setup guide
- `teapot`: Teapot usage documentation (this file)

## Creating Custom FAQ Files

The FAQ file is in JSONL format (JSON Lines), with each line containing a question and answer pair:

```json
{"question": "How do I install KAIA?", "answer": "Run `pip install killeraiagent` to install KAIA."}
{"question": "Where are models stored?", "answer": "Models are stored in the ~/.kaia/models/ directory by default."}
```

Save this file and specify its path when initializing MultiturnTeapot.

## Adding RAG Context

You can add additional context to be included in the RAG process:

```python
teapot = MultiturnTeapot()
teapot.rag_context = """
This is additional context that will be included in each prompt.
Useful for providing temporary information or context for the current session.
"""

response = teapot.chat("What should I do next?")
```

## Performance Considerations

- MultiturnTeapot is designed for efficiency on CPU-only systems
- The model uses approximately 1GB of RAM
- Response generation typically takes 1-3 seconds on modern CPUs
- The context window is limited to 1024 tokens by default

## Integration with KAIA

MultiturnTeapot is integrated into the KAIA platform and can be used as a fallback or assistant model:

```python
from killeraiagent.agents import create_agent

# Create an agent using Teapot as the assistant model
agent = create_agent(primary_model="llama", assistant_model="teapot")

# Interact with the agent
agent.chat("How do I optimize performance for my GPU?")
```

## Extending Teapot

You can extend Teapot's capabilities:

1. **Custom embeddings**: Replace the embedding pipeline by overriding the `_build_embeddings` method
2. **Custom retrieval**: Modify the `_retrieve_faq` method to implement different retrieval strategies
3. **Additional tools**: Extend the `_handle_doc_request` method to support more command types

## Troubleshooting

Common issues and solutions:

1. **Model loading fails**: Ensure the teapotai package is installed (`pip install teapotai`)
2. **FAQ file not found**: Check the path to your FAQ file and ensure it exists
3. **Poor quality responses**: Improve your FAQ file or adjust the system prompt
4. **Slow performance**: Reduce the `max_context_tokens` parameter

For more information on Teapot integration, visit the [TeapotAI documentation](https://docs.teapot.ai).