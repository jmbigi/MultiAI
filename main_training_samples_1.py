# main_training_samples_1.py

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

    # Generate example training samples
    training_data = [
        ("print the number 10 in the terminal", "print(10)", "10"),
        ("calculate the square root of 16", "sqrt(16)", "4"),
        ("check if a number is even", "is_even(7)", "False"),
        ("convert temperature from Celsius to Fahrenheit", "celsius_to_fahrenheit(25)", "77"),
        # Add more examples here
    ]
    
    # Train the model using the generated training samples
    for data in training_data:
        train_model(*data)
