# Convert temperature from Celsius to Fahrenheit
def find_max(numbers):
    max_num = numbers[0]
    for num in numbers:
        if num > max_num:
            max_num = num
    return max_num

numbers = [12, 45, 67, 23, 9, 55]
max_num = find_max(numbers)
print("Maximum number:", max_num)
