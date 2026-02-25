import numpy as np
import matplotlib.pyplot as plt
import random


Nст = 16
Nгр = 52
seed = (Nст + Nгр) * 100 

m = 2**32
a = 4 * Nст + 17  
c = Nгр + 3 * Nст 

size = 10000

X = np.zeros(size, dtype=np.uint64)
X[0] = seed

for n in range(1, size):
    X[n] = (a * X[n - 1] + c) % m

# Normalize to [0, 1]
U = X / m


print(f"First 10 LCG: {U[:10]}")


print(f"m = {m}\nmin = {X.min()}\nmax = {X.max()}")

if X.min() >= 0 and X.max() < m:
    print("Xi values are in range [0, m-1]")
else:
    raise Exception("Xi values out of range")


D = np.var(U)

print(f"\nВыборочная дисперсия: {D}")
print(f"Теоретическая дисперсия: {1/12}")

plt.figure(figsize=(8,5))
plt.hist(U, bins=15, density=True)
plt.title("Гистограмма LCG (оценка плотности w(x))")
plt.xlabel("x")
plt.ylabel("w(x)")
plt.grid(True)
plt.show()


U_sorted = np.sort(U)
F = np.arange(1, size + 1) / size

plt.figure(figsize=(8,5))
plt.plot(U_sorted, F, label="F(x)")
plt.plot([0,1], [0,1], 'r--', label="Теоретическая F(x)=x")
plt.title("Функция распределения")
plt.xlabel("x")
plt.ylabel("F(x)")
plt.legend()
plt.grid(True)
plt.show()



random.seed(seed)

U_py = np.array([random.random() for _ in range(size)])
D_py = np.var(U_py)

print("\n--- Python generator ---")
print(f"Выборочная дисперсия: {D_py}")

plt.figure(figsize=(8,5))
plt.hist(U_py, bins=12, density=True, alpha=0.6, label="Python")
plt.hist(U, bins=12, density=True, alpha=0.6, label="LCG")
plt.title("Сравнение гистограмм")
plt.legend()
plt.grid(True)
plt.show()

U_py_sorted = np.sort(U_py)
F_py = np.arange(1, size + 1) / size

plt.figure(figsize=(8,5))
plt.plot(U_sorted, F, label="LCG")
plt.plot(U_py_sorted, F_py, label="Python")
plt.plot([0,1], [0,1], 'r--', label="F(x)=x")
plt.legend()
plt.title("Сравнение функций распределения")
plt.grid(True)
plt.show()
