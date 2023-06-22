# Calculate the square of numbers in a list
def calculate_square(numbers):
    squares = [num**2 for num in numbers]
    return squares

numbers = [2, 4, 6, 8, 10]
squares = calculate_square(numbers)
print("Squares:", squares)
