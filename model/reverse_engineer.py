import nltk

def reverse_engineer_terminal_output(terminal_output):
    """Reverse-engineers terminal output into natural language instructions and Python code."""
    if terminal_output.isdigit():
        natural_language_instruction = f"print the number {terminal_output} in the terminal"
        python_code = f"print({terminal_output})"
        return natural_language_instruction, python_code
    return None, None
