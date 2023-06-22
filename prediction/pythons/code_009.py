# Calculate the sum of numbers using recursion
def sum_recursive(numbers):
    if len(numbers) == 0:
        return 0
    else:
        return numbers[0] + sum_recursive(numbers[1:])

numbers = [1, 2, 3, 4, 5]
sum_of_numbers = sum_recursive(numbers)
print("Sum of numbers:", sum_of_numbers)
