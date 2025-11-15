#!/usr/bin/env python3
"""
Script to update nn.py to support operations parameter
"""
import re

# Read the file
with open('src/kandinsky/models/nn.py', 'r') as f:
    content = f.read()

# Classes that need updating (exclude RoPE classes as they don't use Linear)
classes_to_update = [
    'TextEmbeddings',
    'VisualEmbeddings',
    'Modulation',
    'MultiheadSelfAttentionEnc',
    'MultiheadSelfAttentionDec',
    'MultiheadCrossAttention',
    'FeedForward',
    'OutLayer'
]

for class_name in classes_to_update:
    # Find the __init__ method signature
    pattern = rf'(class {class_name}\(nn\.Module\):.*?def __init__\(self(?:, [^)]*)?)\):'

    def replacer(match):
        init_sig = match.group(1)
        # Add operations parameter if not already there
        if 'operations' not in init_sig:
            return init_sig + ', operations=None):'
        return match.group(0)

    content = re.sub(pattern, replacer, content, flags=re.DOTALL)

    # Add operations = nn if operations is None after super().__init__()
    pattern = rf'(class {class_name}\(nn\.Module\):.*?def __init__\(.*?\):.*?super\(\).__init__\(\))'

    def add_ops_check(match):
        block = match.group(1)
        if 'if operations is None:' not in block:
            return block + '\n        if operations is None:\n            operations = nn'
        return block

    content = re.sub(pattern, add_ops_check, content, flags=re.DOTALL)

# Global replacements of nn.Linear, nn.LayerNorm with operations.*
# We need to be careful not to replace nn.Module, nn.SiLU, etc.
content = re.sub(r'\bnn\.Linear\b', 'operations.Linear', content)
content = re.sub(r'\bnn\.LayerNorm\b', 'operations.LayerNorm', content)
content = re.sub(r'\bnn\.RMSNorm\b', 'operations.RMSNorm', content)

# Write back
with open('src/kandinsky/models/nn.py', 'w') as f:
    f.write(content)

print("Updated nn.py with operations parameter support")
