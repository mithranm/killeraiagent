"""
Chat template handling for llama.cpp models.

This module provides utilities for working with Jinja2-based chat templates
that format messages for different model types.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from jinja2 import Template, Environment, BaseLoader, TemplateError, meta

from killeraiagent.paths import get_data_paths


class ChatTemplateManager:
    """Manager for chat templates used with different models."""
    
    BUILT_IN_TEMPLATES = {
        "chatml": """{% for message in messages %}
{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>'}}
{% endfor %}
<|im_start|>assistant
""",
        
        "llama2": """{% if messages[0]['role'] == 'system' %}
<s>[INST] <<SYS>>
{{ messages[0]['content'] }}
<</SYS>>

{% set loop_messages = messages[1:] %}
{% else %}
{% set loop_messages = messages %}
{% endif %}

{% for message in loop_messages %}
{% if message['role'] == 'user' %}
{% if loop.first %}
<s>[INST] {{ message['content'] }} [/INST]
{% else %}
[INST] {{ message['content'] }} [/INST]
{% endif %}
{% elif message['role'] == 'assistant' %}
 {{ message['content'] }} </s>
{% endif %}
{% endfor %}
""",
        
        "mistral": """{% for message in messages %}
{% if message['role'] == 'user' %}
[INST] {{ message['content'] }} [/INST]
{% elif message['role'] == 'assistant' %}
{{ message['content'] }}
{% elif message['role'] == 'system' %}
<s>[INST] <<SYS>>
{{ message['content'] }}
<</SYS>>
{% endif %}
{% endfor %}
""",
        
        "alpaca": """{% if messages[0]['role'] == 'system' %}
{{ messages[0]['content'] }}

{% endif %}
{% for message in messages %}
{% if message['role'] == 'user' %}
### Instruction:
{{ message['content'] }}

{% elif message['role'] == 'assistant' and not loop.last %}
### Response:
{{ message['content'] }}

{% endif %}
{% endfor %}
### Response:
"""
    }
    
    def __init__(self, templates_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the template manager.
        
        Args:
            templates_dir: Directory containing custom template files
        """
        self.paths = get_data_paths()
        self.templates_dir = Path(templates_dir) if templates_dir else self.paths.templates
        
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        self.template_cache = {}
        
        self._create_builtin_templates()
    
    def _create_builtin_templates(self):
        """Create files for built-in templates if they don't exist."""
        for name, content in self.BUILT_IN_TEMPLATES.items():
            template_path = self.templates_dir / f"{name}.jinja2"
            if not template_path.exists():
                with open(template_path, 'w') as f:
                    f.write(content)
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """
        List all available templates.
        
        Returns:
            List of dictionaries with template information
        """
        templates = []
        
        for name in self.BUILT_IN_TEMPLATES:
            templates.append({
                "name": name,
                "type": "built-in",
                "path": str(self.templates_dir / f"{name}.jinja2")
            })
        
        for file_path in self.templates_dir.glob("*.jinja2"):
            name = file_path.stem
            if name not in self.BUILT_IN_TEMPLATES:
                templates.append({
                    "name": name,
                    "type": "custom",
                    "path": str(file_path)
                })
        
        return templates
    
    def get_template(self, template_name: str) -> Template:
        """
        Get a Jinja2 template by name.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Jinja2 Template object
            
        Raises:
            ValueError: If template not found
        """
        if template_name in self.template_cache:
            return self.template_cache[template_name]
        
        if template_name in self.BUILT_IN_TEMPLATES:
            template_str = self.BUILT_IN_TEMPLATES[template_name]
            template = Template(template_str)
            self.template_cache[template_name] = template
            return template
        
        template_path = self.templates_dir / f"{template_name}.jinja2"
        if not template_path.exists():
            raise ValueError(f"Template not found: {template_name}")
        
        with open(template_path, 'r') as f:
            template_str = f.read()
        
        template = Template(template_str)
        self.template_cache[template_name] = template
        return template
    
    def render_template(self, template_name: str, messages: List[Dict[str, str]]) -> str:
        template = self.get_template(template_name)
        return template.render(messages=messages)
    
    def detect_format(self, model_path: Union[str, Path]) -> str:
        model_name = str(model_path).lower()
        
        if "llama-2" in model_name or "llama2" in model_name:
            return "llama2"
        elif "mistral" in model_name or "mixtral" in model_name:
            return "mistral"
        elif "alpaca" in model_name or "vicuna" in model_name:
            return "alpaca"
        
        return "chatml"
    
    def create_template(self, name: str, content: str) -> str:
        try:
            Template(content)
        except TemplateError as e:
            raise ValueError(f"Invalid template: {e}")
        
        if name in self.BUILT_IN_TEMPLATES:
            raise ValueError(f"Cannot override built-in template: {name}")
        
        template_path = self.templates_dir / f"{name}.jinja2"
        with open(template_path, 'w') as f:
            f.write(content)
        
        if name in self.template_cache:
            del self.template_cache[name]
        
        return str(template_path)
    
    def get_template_content(self, template_name: str) -> str:
        if template_name in self.BUILT_IN_TEMPLATES:
            return self.BUILT_IN_TEMPLATES[template_name]
        
        template_path = self.templates_dir / f"{template_name}.jinja2"
        if not template_path.exists():
            raise ValueError(f"Template not found: {template_name}")
        
        with open(template_path, 'r') as f:
            return f.read()
    
    def analyze_template(self, template_name: str) -> Dict[str, Any]:
        content = self.get_template_content(template_name)
        
        env = Environment(loader=BaseLoader())
        ast = env.parse(content)
        variables = meta.find_undeclared_variables(ast)
        
        return {
            "name": template_name,
            "variables": list(variables),
            "length": len(content),
            "has_messages_loop": ("messages" in variables and "{% for" in content and "messages" in content)
        }


def create_template_manager() -> ChatTemplateManager:
    return ChatTemplateManager()


def format_messages(
    messages: List[Dict[str, str]], 
    template_name_or_content: str,
    is_template_content: bool = False
) -> str:
    if is_template_content:
        template = Template(template_name_or_content)
        return template.render(messages=messages)
    else:
        manager = create_template_manager()
        return manager.render_template(template_name_or_content, messages)
