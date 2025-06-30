# 1.Write a python program to compute Central Tendency Measures: Mean, Median,Mode Measure of Dispersion: Variance, Standard Deviation

import statistics

data = [21, 5, 20, 20, 21, 1, 9, 42, 25, 20]

mean = statistics.mean(data)
median = statistics.median(data)
mode = statistics.mode(data)


variance = statistics.variance(data)
std_dev = statistics.stdev(data)


print("\n--- Central Tendency ---")
print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)

print("\n--- Measure of Dispersion ---")
print("Variance:", round(variance, 2))
print("Standard Deviation:", round(std_dev, 2))



# Central Tendency and Dispersion Measures using Formulas

# Input: dynamic list of numbers from user
data = list(map(float, input("Enter numbers separated by spaces: ").split()))

# Function to calculate Mean
def calculate_mean(data):
    return sum(data) / len(data)

# Function to calculate Median
def calculate_median(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2
    else:
        return sorted_data[mid]

# Function to calculate Mode (can return multiple modes)
def calculate_mode(data):
    frequency = {}
    for value in data:
        frequency[value] = frequency.get(value, 0) + 1
    max_freq = max(frequency.values())
    modes = [key for key, val in frequency.items() if val == max_freq]
    return modes

# Function to calculate Variance (Population)
def calculate_variance(data):
    mean = calculate_mean(data)
    return sum((x - mean) ** 2 for x in data) / len(data)

# Function to calculate Standard Deviation
def calculate_std_dev(data):
    variance = calculate_variance(data)
    return variance ** 0.5

# Perform Calculations
mean = calculate_mean(data)
median = calculate_median(data)
mode = calculate_mode(data)
variance = calculate_variance(data)
std_dev = calculate_std_dev(data)

# Display the Results
print("\nResults:")
print("Data:", data)
print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)
print("Variance:", variance)
print("Standard Deviation:", std_dev)
