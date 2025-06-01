import torch


def f(x, y, z):
    return x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8


def optimize_with_torch(lr=0.1, steps=100):
    x = torch.tensor(0.0, requires_grad=True)
    y = torch.tensor(0.0, requires_grad=True)
    z = torch.tensor(0.0, requires_grad=True)

    for step in range(steps):
        if x.grad:
            x.grad.zero_()
        if y.grad:
            y.grad.zero_()
        if z.grad:
            z.grad.zero_()

        loss = f(x, y, z)

        loss.backward()

        with torch.no_grad():
            x -= lr * x.grad
            y -= lr * y.grad
            z -= lr * z.grad

        print(
            f"Step {step+1:3d}: x={x.item():.4f}, y={y.item():.4f}, z={z.item():.4f}, f(x,y,z)={loss.item():.4f}")

    return x.item(), y.item(), z.item(), f(x, y, z).item()


optimize_with_torch()
