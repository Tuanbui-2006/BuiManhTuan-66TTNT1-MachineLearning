def grad(x):
    return 2*x
def cost(x):
    return 2**x -2
def myGD1(x0, eta): 
    x = [x0]
    for it in range (100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return (x,it)
print(grad(5))
print(cost(5))
print(myGD1(5,0.1))

