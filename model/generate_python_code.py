import nltk

def generate_python_code(natural_language_instruction):
    """Generates Python code from natural language instructions."""
    tokens = nltk.word_tokenize(natural_language_instruction)
    code = ""
    for token in tokens:
        if token.lower() == "print":
            code += "print("
        elif token.isdigit():
            code += token
            code += ")"
    return code
