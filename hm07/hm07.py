from micrograd.engine import Value


def f(x, y, z):
    return x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8


x = Value(0.0)
y = Value(0.0)
z = Value(0.0)

learning_rate = 0.1

for i in range(50):

    output = f(x, y, z)

    x.grad = 0.0
    y.grad = 0.0
    z.grad = 0.0

    output.backward()

    x.data -= learning_rate * x.grad
    y.data -= learning_rate * y.grad
    z.data -= learning_rate * z.grad

    print(
        f"Step {i+1}: x={x.data:.4f}, y={y.data:.4f}, z={z.data:.4f}, f(x,y,z)={output.data:.4f}")


print("\n：")
print(f"x = {x.data:.4f}, y = {y.data:.4f}, z = {z.data:.4f}, f(x, y, z) = {f(x, y, z).data:.4f}")
