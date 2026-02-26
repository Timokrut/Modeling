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
print(sum(U)/size)

print(f"m = {m}\nmin = {X.min()}\nmax = {X.max()}")

if X.min() >= 0 and X.max() < m:
    print("Xi values are in range [0, m-1]")
else:
    raise Exception("Xi values out of range")

D = np.var(U)

print(f"\nДисперсия: {D}")
print(f"Теоретическая дисперсия: {1/12}")

cnt = 15

plt.figure(figsize=(8,5))
counts, edges, _ = plt.hist(U, bins=cnt, density=False)
plt.title("Гистограмма LCG (оценка плотности w(x))")
plt.xlabel("x")
plt.ylabel("w(x)")
plt.grid(True)
plt.show()


cum_counts = np.cumsum(counts)
F = cum_counts / size

plt.figure(figsize=(8,5))
plt.step(edges[1:], F, where='post', label="F(x)")
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

print("\nPython generator")
print(f"Дисперсия: {D_py}")

plt.figure(figsize=(8,5))
c1, e1, _ = plt.hist(U_py, bins=cnt, density=False, alpha=0.6, label="Python")
c2, e2, _ = plt.hist(U, bins=cnt, density=False, alpha=0.6, label="LCG")
plt.title("Сравнение гистограмм")
plt.legend()
plt.grid(True)
plt.show()

cum_counts1 = np.cumsum(c1)
cum_counts2 = np.cumsum(c2)
F1 = cum_counts1 / size
F2 = cum_counts2 / size

plt.figure(figsize=(8,5))
plt.step(e2[1:], F2, where='post', label="LCG")
plt.step(e1[1:], F1, where='post', label="Python")
plt.plot([0,1], [0,1], 'r--', label="F(x)=x")
plt.legend()
plt.title("Сравнение функций распределения")
plt.grid(True)
plt.show()
