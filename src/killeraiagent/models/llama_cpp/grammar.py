"""
Grammar handling for structured generation with llama.cpp.

This module provides utilities for working with GBNF grammars to constrain
model outputs to specific formats like JSON, YAML, etc.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from killeraiagent.paths import get_data_paths


class GrammarManager:
    """Manager for GBNF grammars that constrain model outputs."""
    
    def __init__(self, grammars_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the grammar manager.
        
        Args:
            grammars_dir: Directory containing grammar files (.gbnf)
        """
        self.paths = get_data_paths()
        self.grammars_dir = Path(grammars_dir) if grammars_dir else self.paths.grammars
        
        # Ensure the directory exists
        self.grammars_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache of loaded grammars
        self.grammar_cache = {}
    
    def list_grammars(self) -> List[Dict[str, Any]]:
        """
        List all available grammar files.
        
        Returns:
            List of dictionaries with grammar information
        """
        grammars = []
        
        # Search for .gbnf files
        grammar_files = list(self.grammars_dir.glob("**/*.gbnf"))
        
        for file_path in grammar_files:
            info = {
                "name": file_path.stem,
                "path": str(file_path),
                "relative_path": str(file_path.relative_to(self.grammars_dir)),
                "size": file_path.stat().st_size
            }
            
            # Try to determine purpose from first few lines
            try:
                with open(file_path, 'r') as f:
                    first_lines = [next(f) for _ in range(5) if f]
                    for line in first_lines:
                        if line.strip().startswith('#'):
                            info["description"] = line.strip('# \n')
                            break
            except Exception:
                pass
            
            grammars.append(info)
        
        return grammars
    
    def get_grammar_path(self, grammar_name: str) -> Optional[Path]:
        """
        Get the path to a grammar file by name.
        
        Args:
            grammar_name: Name of the grammar (with or without .gbnf extension)
        
        Returns:
            Path to the grammar file or None if not found
        """
        if not grammar_name.endswith('.gbnf'):
            grammar_name += '.gbnf'
        
        exact_path = self.grammars_dir / grammar_name
        if exact_path.exists():
            return exact_path
        
        for file_path in self.grammars_dir.glob("**/*.gbnf"):
            if file_path.name == grammar_name:
                return file_path
        
        return None
    
    def get_or_create_json_grammar(self, schema: Dict[str, Any]) -> Path:
        """
        Generate a GBNF grammar from a JSON schema and save it.
        
        Args:
            schema: JSON schema definition
        
        Returns:
            Path to the generated grammar file
        """
        import hashlib
        schema_str = json.dumps(schema, sort_keys=True)
        schema_hash = hashlib.md5(schema_str.encode()).hexdigest()[:10]
        grammar_name = f"json_schema_{schema_hash}.gbnf"
        
        grammar_path = self.grammars_dir / grammar_name
        if grammar_path.exists():
            return grammar_path
        
        grammar_content = self._generate_json_grammar(schema)
        
        with open(grammar_path, 'w') as f:
            f.write(grammar_content)
        
        return grammar_path
    
    def _generate_json_grammar(self, schema: Dict[str, Any]) -> str:
        """
        Generate GBNF grammar from JSON schema.
        
        Args:
            schema: JSON schema definition
        
        Returns:
            GBNF grammar text
        """
        lines = [
            "# JSON grammar generated from schema",
            "root ::= object",
            "",
            "object ::= \"{\" ws (pair (ws \",\" ws pair)*)? ws \"}\"",
            "pair ::= string ws \":\" ws value",
            "",
            "array ::= \"[\" ws (value (ws \",\" ws value)*)? ws \"]\"",
            "",
            "value ::= object | array | string | number | boolean | null",
            "",
            "string ::= \"\\\"\" ([^\\\"\\\\] | \\\\\\\\ | \\\\\\\")* \"\\\"\"",
            "",
            "number ::= integer | float",
            "integer ::= [\"-\"]? (\"0\" | [1-9] [0-9]*)",
            "float ::= [\"-\"]? (\"0\" | [1-9] [0-9]*) \".\" [0-9]+",
            "",
            "boolean ::= \"true\" | \"false\"",
            "null ::= \"null\"",
            "",
            "# Whitespace handling",
            "ws ::= [ \\t\\n]*"
        ]
        
        if 'properties' in schema:
            property_rules = []            
            for prop_name, prop_schema in schema['properties'].items():
                if 'type' in prop_schema:
                    prop_type = prop_schema['type']
                    if prop_type == 'string':
                        if 'enum' in prop_schema:
                            enum_values = " | ".join([f'\"\\\"{val}\\\"\"' for val in prop_schema['enum']])
                            property_rules.append(f"{prop_name}_value ::= {enum_values}")
                        else:
                            property_rules.append(f"{prop_name}_value ::= string")
                    elif prop_type in ['number', 'integer']:
                        property_rules.append(f"{prop_name}_value ::= number")
                    elif prop_type == 'boolean':
                        property_rules.append(f"{prop_name}_value ::= boolean")
                    elif prop_type == 'array':
                        property_rules.append(f"{prop_name}_value ::= array")
                    elif prop_type == 'object':
                        property_rules.append(f"{prop_name}_value ::= object")
                    else:
                        property_rules.append(f"{prop_name}_value ::= value")
            
            if property_rules:
                lines.append("")
                lines.append("# Property-specific rules")
                lines.extend(property_rules)
        
        return "\n".join(lines)


def create_grammar_manager() -> GrammarManager:
    """Factory function to create a grammar manager."""
    return GrammarManager()
