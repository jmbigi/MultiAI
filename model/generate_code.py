# model.generate_code.py

import nltk

def generate_python_code(natural_language_instruction):
    """
    Generates Python code from natural language instructions.
    
    Args:
        natural_language_instruction (str): The natural language instruction to generate Python code from.
    
    Returns:
        str: The generated Python code.
    """
    tokens = nltk.word_tokenize(natural_language_instruction)
    code = ""
    for token in tokens:
        if token.lower() == "print":
            code += "print("
        elif token.isdigit():
            code += token
            code += ")"
    return code
