import nltk

def generate_python_code(natural_language_instruction):
    """This function generates Python code from natural language instructions."""
    tokens = nltk.word_tokenize(natural_language_instruction)
    code = ""
    for token in tokens:
        if token.lower() == "print":
            code += "print("
        elif token.isdigit():
            code += token
            code += ")"
    return code

def reverse_engineer_terminal_output(terminal_output):
    """This function reverse-engineers terminal output into natural language instructions and Python code."""
    if terminal_output.isdigit():
        natural_language_instruction = f"print the number {terminal_output} in the terminal"
        python_code = f"print({terminal_output})"
        return natural_language_instruction, python_code
    return None, None

if __name__ == "__main__":
    # nltk.download('punkt')
    natural_language_instruction = "print the number 20 in the terminal"
    python_code = generate_python_code(natural_language_instruction)
    print(f"Natural language instruction: {natural_language_instruction}")
    print(f"Generated Python code: {python_code}")

    terminal_output = "20"
    natural_language_instruction, python_code = reverse_engineer_terminal_output(terminal_output)
    if natural_language_instruction and python_code:
        print(f"Terminal output: {terminal_output}")
        print(f"Reverse-engineered natural language instruction: {natural_language_instruction}")
        print(f"Reverse-engineered Python code: {python_code}")
    else:
        print("Unable to reverse-engineer terminal output")