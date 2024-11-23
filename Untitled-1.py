import numpy as np
import random
import matplotlib.pyplot as plt

# Constants
NUM_ORDERS = 400  # Number of battery orders
NUM_METHODS = 4   # Charging methods: 1-super, 2-fast, 3-normal, 4-slow
NUM_GENERATIONS = 50
POPULATION_SIZE = 30
MUTATION_RATE = 0.1
TIME_HORIZON = 780  # Total time in minutes (e.g., 13 hours from 9:00 to 22:00)

# Example input data
time_of_use_prices = {"off-peak": 0.06, "mid-peak": 0.10, "on-peak": 0.13}
charging_methods = {
    1: {"power": 120, "damage_cost": 8.75, "charging_time": 60},
    2: {"power": 80, "damage_cost": 3.50, "charging_time": 90},
    3: {"power": 60, "damage_cost": 0.70, "charging_time": 120},
    4: {"power": 40, "damage_cost": 0.0, "charging_time": 180},
}

# Generate normally distributed arrival times
mean_time = TIME_HORIZON / 2
std_dev = TIME_HORIZON / 6  # Controls spread of arrivals
arrival_times = np.clip(
    np.random.normal(loc=mean_time, scale=std_dev, size=NUM_ORDERS).astype(int),
    0, TIME_HORIZON - 1
)

# Create orders with normally distributed arrivals
orders = [{"arrival_time": arrival_time, "soc": random.uniform(0.1, 0.35), "soh": random.uniform(0.8, 1.0)}
          for arrival_time in arrival_times]
# orders = [{"arrival_time": random.randint(0, TIME_HORIZON), "soc": random.uniform(0.1, 0.35), "soh": random.uniform(0.8, 1.0)}
#           for _ in range(NUM_ORDERS)]

# Initialize tracking variables for graphs
time_steps = np.arange(TIME_HORIZON)
battery_stock = np.zeros(TIME_HORIZON)
charging_counts = {method: np.zeros(TIME_HORIZON) for method in charging_methods.keys()}
power_usage = np.zeros(TIME_HORIZON)

# Fitness Function
def fitness(solution, orders, charging_methods, time_of_use_prices):
    total_cost = 0
    for i, method in enumerate(solution):
        order = orders[i]
        charger = charging_methods[method]
        # Battery cost
        battery_cost = charger["damage_cost"]
        # Electricity cost
        time_of_day = "on-peak" if order["arrival_time"] % 360 < 120 else "mid-peak"
        electricity_cost = charger["power"] * time_of_use_prices[time_of_day]
        total_cost += battery_cost + electricity_cost
    return -total_cost  # Minimize cost

def track_metrics(solution, orders, charging_methods):
    stock = 50  # Initial stock of batteries
    power_schedule = np.zeros(TIME_HORIZON)

    for i, method in enumerate(solution):
        order = orders[i]
        start_time = order["arrival_time"]
        charger = charging_methods[method]
        duration = charger["charging_time"]
        end_time = min(start_time + duration, TIME_HORIZON - 1)  # Ensure within bounds

        # Update battery stock levels
        battery_stock[start_time] -= 1
        battery_stock[end_time] += 1

        # Count charging methods
        charging_counts[method][start_time:end_time] += 1

        # Record power usage
        power_schedule[start_time:end_time] += charger["power"]

    # Cumulative battery stock over time
    for t in range(1, TIME_HORIZON):
        battery_stock[t] += battery_stock[t - 1]

    return power_schedule


    # Cumulative battery stock over time
    for t in range(1, TIME_HORIZON):
        battery_stock[t] += battery_stock[t - 1]

    return power_schedule

# Initialize Population
def initialize_population(size, num_orders, num_methods):
    return [np.random.randint(1, num_methods + 1, size=num_orders) for _ in range(size)]

# Crossover Operation
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2

# Mutation Operation
def mutate(solution, num_methods, mutation_rate):
    for i in range(len(solution)):
        if random.random() < mutation_rate:
            solution[i] = random.randint(1, num_methods)
    return solution

# Genetic Algorithm
def genetic_algorithm():
    population = initialize_population(POPULATION_SIZE, NUM_ORDERS, NUM_METHODS)
    best_solution = None
    best_fitness = float('-inf')

    for generation in range(NUM_GENERATIONS):
        fitness_values = [fitness(sol, orders, charging_methods, time_of_use_prices) for sol in population]
        sorted_indices = np.argsort(fitness_values)[::-1]
        population = [population[i] for i in sorted_indices]

        # Update best solution
        if fitness_values[sorted_indices[0]] > best_fitness:
            best_fitness = fitness_values[sorted_indices[0]]
            best_solution = population[0]

        # Generate next generation
        next_generation = population[:2]  # Elitism
        while len(next_generation) < POPULATION_SIZE:
            parent1, parent2 = random.choices(population[:10], k=2)
            child1, child2 = crossover(parent1, parent2)
            next_generation.extend([mutate(child1, NUM_METHODS, MUTATION_RATE),
                                    mutate(child2, NUM_METHODS, MUTATION_RATE)])
        population = next_generation

    # Track metrics for the best solution
    global power_usage
    power_usage = track_metrics(best_solution, orders, charging_methods)

    return best_solution, -best_fitness

# Generate random charging method assignment
def random_assignment(num_orders, num_methods):
    return np.random.randint(1, num_methods + 1, size=num_orders)

# Evaluate cost for a given solution
def evaluate_solution(solution, orders, charging_methods, time_of_use_prices):
    total_cost = 0
    for i, method in enumerate(solution):
        order = orders[i]
        charger = charging_methods[method]
        # Battery cost (charging damage)
        battery_cost = charger["damage_cost"]
        # Electricity cost
        time_of_day = "on-peak" if order["arrival_time"] % 360 < 120 else "mid-peak"
        electricity_cost = charger["power"] * time_of_use_prices[time_of_day]
        total_cost += battery_cost + electricity_cost
    return total_cost

# Final cost breakdown after the GA finishes
def calculate_final_costs(solution, orders, charging_methods, time_of_use_prices):
    battery_damage_cost = 0
    electricity_cost = 0
    total_cost = 0

    for i, method in enumerate(solution):
        order = orders[i]
        charger = charging_methods[method]

        # Battery damage cost
        battery_damage_cost += charger["damage_cost"]

        # Electricity cost
        time_of_day = "on-peak" if order["arrival_time"] % 360 < 120 else "mid-peak"
        electricity_cost += charger["power"] * time_of_use_prices[time_of_day]

        # Total cost
        total_cost += charger["damage_cost"] + charger["power"] * time_of_use_prices[time_of_day]

    return battery_damage_cost, electricity_cost, total_cost

# Run GA
best_solution, best_cost = genetic_algorithm()
best_cost2 = evaluate_solution(best_solution, orders, charging_methods, time_of_use_prices)



# Count arrivals per time step
arrival_counts = np.zeros(TIME_HORIZON)
for order in orders:
    arrival_counts[order["arrival_time"]] += 1

# Calculate final costs
battery_damage_cost, electricity_cost, total_cost = calculate_final_costs(best_solution, orders, charging_methods, time_of_use_prices)

# Print the final average costs
print(f"Final Average Battery Damage Cost: ${battery_damage_cost:.2f}")
print(f"Final Average Electricity Cost: ${electricity_cost:.2f}")
print(f"Final Average Total Cost: ${total_cost:.2f}")

# Plot Arrival Times
plt.figure(figsize=(12, 4))
plt.bar(time_steps, arrival_counts, color='skyblue', label='Arrivals')
plt.title("Arrival Time Distribution")
plt.xlabel("Time (minutes)")
plt.ylabel("Number of Arrivals")
plt.legend()
plt.grid()
plt.show()

# Plot Battery Stock vs Time
plt.figure(figsize=(12, 4))
plt.plot(time_steps, battery_stock, label='Battery Stock')
plt.title("Battery Stock vs Time")
plt.xlabel("Time (minutes)")
plt.ylabel("Battery Stock")
plt.legend()
plt.grid()
plt.show()

# Plot Distribution of Charging Methods
plt.figure(figsize=(12, 4))
for method, counts in charging_counts.items():
    plt.plot(time_steps, counts, label=f"Method {method}")
plt.title("Distribution of Charging Methods")
plt.xlabel("Time (minutes)")
plt.ylabel("Number of Charges")
plt.legend()
plt.grid()
plt.show()

# Plot Power vs Time
plt.figure(figsize=(12, 4))
plt.plot(time_steps, power_usage, label='Power Usage')
plt.title("Power Usage vs Time")
plt.xlabel("Time (minutes)")
plt.ylabel("Power (kW)")
plt.legend()
plt.grid()
plt.show()



# Initialize tracking variables for graphs
time_steps = np.arange(TIME_HORIZON)
battery_stock = np.zeros(TIME_HORIZON)
charging_counts = {method: np.zeros(TIME_HORIZON) for method in charging_methods.keys()}
power_usage = np.zeros(TIME_HORIZON)

# Generate and evaluate random solution
random_solution = random_assignment(NUM_ORDERS, NUM_METHODS)
random_cost = evaluate_solution(random_solution, orders, charging_methods, time_of_use_prices)
random_power_usage = track_metrics(random_solution, orders, charging_methods)

print(f"Cost with Random Assignment: ${random_cost:.2f}")

# Plot graphs for Random Assignment
# Battery Stock vs Time
plt.figure(figsize=(12, 4))
plt.plot(time_steps, battery_stock, label='Battery Stock (Random)', color='orange')
plt.title("Battery Stock vs Time (Random Assignment)")
plt.xlabel("Time (minutes)")
plt.ylabel("Battery Stock")
plt.legend()
plt.grid()
plt.show()

# Distribution of Charging Methods
plt.figure(figsize=(12, 4))
for method, counts in charging_counts.items():
    plt.plot(time_steps, counts, label=f"Method {method} (Random)")
plt.title("Distribution of Charging Methods (Random Assignment)")
plt.xlabel("Time (minutes)")
plt.ylabel("Number of Charges")
plt.legend()
plt.grid()
plt.show()

# Plot Power vs Time
plt.figure(figsize=(12, 4))
plt.plot(time_steps, power_usage, label='Power Usage')
plt.title("Power Usage vs Time (Random Assignment)")
plt.xlabel("Time (minutes)")
plt.ylabel("Power (kW)")
plt.legend()
plt.grid()
plt.show()