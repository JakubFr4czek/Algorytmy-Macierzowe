import time
import numpy as np
import matplotlib.pyplot as plt

from floatcounter import FloatCounter
from utility import get_random_matrix_pair_any_size

# Casts all elements in matrix / vector to FloatCounter
def float_matrix_to_floatcounter_matrix(a):
    b = np.zeros(a.shape, dtype=FloatCounter)
    
    if len(a.shape) == 1: # vector
        for i in range(len(a)):
            b[i] = FloatCounter(a[i])
    else: # matrix
        for i in range(len(a)):
            for j in range(len(a[i])):
                b[i][j] = FloatCounter(a[i][j])
    return b

# Clears all file contents
def clear_file(filename):
    with open(filename, "w") as file:
        pass

# Measures function runtime and flops
def measure_FLOPS_and_runtime(filename, measurement_id, function, *args):

        args = list(map(float_matrix_to_floatcounter_matrix, args))

        FloatCounter.reset_counters()

        start = time.time()
            
        function(*args)

        end = time.time()
            
        results = FloatCounter.get_data()

        with open(filename, "a") as file:
            file.write(str(measurement_id) + " " + str(results[0]) + " " + str(results[1]) + " " + str(results[2]) + " " + str(results[3]) + " " + str(end-start) + "\n")
            
'''
def measure_FLOPS_and_runtime(function, n_args, ns, filename):

    with open(filename, "w") as file:
        pass

    for i in ns:
        
        A, B = get_random_matrix_pair_any_size(i)

        A = float_matrix_to_floatcounter_matrix(A)
        B = float_matrix_to_floatcounter_matrix(B)

        FloatCounter.reset_counters()

        if n_args == 1:
    
            start = time.time()
            
            function(A)

            end = time.time()
            
        elif n_args == 2:
            
            start = time.time()
            
            function(A, B)

            end = time.time()
            
        else:
            raise Exception("n_args can be equal to 1 or 2")
            
        results = FloatCounter.get_data()

        with open(filename, "a") as file:
            file.write(str(i) + " " + str(results[0]) + " " + str(results[1]) + " " + str(results[2]) + " " + str(results[3]) + " " + str(end-start) + "\n")

'''
            
def show_FLOPS_plot(filename, algorithm_name):
    data = np.loadtxt(filename)

    n = data[:, 0]
    mul = data[:, 1]
    add = data[:, 2]
    sub = data[:, 3]

    flops = mul + add + sub

    plt.figure(figsize=(10, 6))
    plt.plot(flops, label=algorithm_name)
    
    plt.legend()

    plt.xlabel("Rozmiar macierzy")
    plt.ylabel("Liczba FLOPS'ów")
    plt.title(f"Porównanie liczby FLOPS'ów do rozmiarów macierzy")

    plt.show()

def show_mult_comparison_plot(filenames, algorithms_names):
    if len(filenames) != len(algorithms_names):
        raise Exception("len(filenames) != len(algorithms_names)")
    
    data = []
    
    for filename in filenames:
        data.append(np.loadtxt(filename))

    plt.figure(figsize=(10, 6))
    
    for i, dt in enumerate(data):
        
        n = dt[:, 0]
        mult = dt[:, 1]
        
        plt.step(n, mult, label = algorithms_names[i])

    plt.legend()

    plt.xlabel("Rozmiar macierzy")
    plt.ylabel("Liczba mnożeń")
    plt.title(f"Porównanie liczby mnożeń do rozmiarów macierzy")

    plt.show()
    
def show_FLOPS_comparison_plot(filenames, algorithms_names):
    if len(filenames) != len(algorithms_names):
        raise Exception("len(filenames) != len(algorithms_names)")
    
    data = []
    
    for filename in filenames:
        data.append(np.loadtxt(filename))

    plt.figure(figsize=(10, 6))
    
    for i, dt in enumerate(data):
        
        n = dt[:, 0]
        mult = dt[:, 1]
        add = dt[:, 2]
        sub = dt[:, 3]
        div = dt[:, 4]
        
        plt.step(n, mult + add + sub + div, label = algorithms_names[i])

    plt.legend()

    plt.xlabel("Rozmiar macierzy")
    plt.ylabel("Liczba FLOPS'ów")
    plt.title(f"Porównanie liczby FLOPS'ów do rozmiarów macierzy")

    plt.show()
    
def show_runtime_plot(filename, algorithm_name):
    data = np.loadtxt(filename)
    
    n = data[:, 0]
    runtime = data[:, 5]

    plt.figure(figsize=(10, 6))
    plt.plot(n, runtime, label=f"{algorithm_name} Runtime", color='blue', marker='o')

    plt.xlabel("Rozmiar macierzy")
    plt.ylabel("Czas wykonania")
    plt.title(f"Porównanie czasów wykonania do rozmiarów macierzy dla algorytmu {algorithm_name}")
    plt.legend()

    plt.show()

def fit_curve_to_runtime_plot(filename, algorithm_name, curve_function, function_name):
    data = np.loadtxt(filename)
    
    n = data[:, 0]
    runtime = data[:, 5]

    plt.figure(figsize=(10, 6))


    n_function = np.array(list(map(curve_function, n)))
    scaling_factor = np.mean(runtime / n_function)
    scaled_n_function = n_function * scaling_factor

        
    plt.plot(n, scaled_n_function, label=function_name, color='red')
        
    plt.plot(n, runtime, label=f"{algorithm_name} Runtime", color='blue', marker='o')

    plt.xlabel("Rozmiar macierzy")
    plt.ylabel("Czas wykonania")
    plt.title(f"Porównanie czasów wykonania do rozmiarów macierzy dla algorytmu {algorithm_name}")
    plt.legend()

    plt.show()

def show_runtime_comparison_plot(filenames, algorithms_names):
    if len(filenames) != len(algorithms_names):
        raise Exception("len(filenames) != len(algorithms_names)")
    
    data = []
    
    for filename in filenames:
        data.append(np.loadtxt(filename))

    plt.figure(figsize=(10, 6))
    
    for i, dt in enumerate(data):
        
        n = dt[:, 0]
        runtime = dt[:, 5]
        
        plt.step(n, runtime, label = algorithms_names[i])

    plt.legend()

    plt.xlabel("Rozmiar macierzy")
    plt.ylabel("Czas wykonania")
    plt.title(f"Porównanie czasów wykonania do rozmiarów macierzy")

    plt.show()
    
    
