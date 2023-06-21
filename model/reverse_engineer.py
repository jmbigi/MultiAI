# model.reverse_engineer.py

import nltk

def reverse_engineer_terminal_output(terminal_output):
    """
    Reverse-engineers terminal output into natural language instructions and Python code.
    
    Args:
        terminal_output (str): The terminal output to be reverse-engineered.
    
    Returns:
        tuple or None: A tuple containing the reverse-engineered natural language instruction and
        Python code if the terminal output is a valid number. Otherwise, returns None.
    """
    if terminal_output.isdigit():
        natural_language_instruction = f"print the number {terminal_output} in the terminal"
        python_code = f"print({terminal_output})"
        return natural_language_instruction, python_code
    
    return None, None
