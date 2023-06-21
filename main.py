import numpy as np
from model.generate_code import generate_python_code
from model.reverse_engineer import reverse_engineer_terminal_output
from training.train_model import train_model

if __name__ == "__main__":
    # Example usage of the functions
    
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

    # Request user input for each of the nodes
    new_natural_language_instruction = input("Please enter a new natural language instruction: ")
    new_python_code = input("Please enter the corresponding Python code: ")
    new_terminal_output = input("Please enter the corresponding terminal output: ")

    # Train the model using the user inputs
    train_model(new_natural_language_instruction, new_python_code, new_terminal_output)
