# MultiAI

**Project Description:**

We're developing an innovative, multidirectional Artificial Intelligence (AI) model, constructed in Python. The unique structure of this system revolves around three interconnected nodes: Natural Language, Python Code, and Terminal Output. These nodes are designed to function bidirectionally, facilitating diverse input-output interactions.

To illustrate, let's consider a natural language instruction such as "print the number 20 in the terminal". The model processes this instruction, generating equivalent Python code (i.e., print(20)), and simultaneously produces '20' in the terminal, showcasing the dynamic flexibility of the nodes.

On the flip side, if the Terminal Output node receives '20' as an input, the model adeptly reverse-engineers the original command. As a result, the Natural Language node fills with "print the number 20 in the terminal", and the Python Code node populates with print(20).

Our overarching objective is to build an AI system that enables seamless integration between human language, programming code, and terminal output, reshaping our interaction with technology.

**Python Code:**

```python
import nltk

def generate_python_code(natural_language_instruction):
    """This function transforms natural language instructions into equivalent Python code."""
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
    """This function converts terminal output back into the original natural language instructions and Python code."""
    if terminal_output.isdigit():
        natural_language_instruction = f"print the number {terminal_output} in the terminal"
        python_code = f"print({terminal_output})"
        return natural_language_instruction, python_code
    return None, None

if __name__ == "__main__":
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
        print("Failed to reverse-engineer terminal output")
```

Your second version of Python code is identical to the first one, so the same corrections apply. If you intended to make different changes or enhancements, they're not apparent in the provided versions.
