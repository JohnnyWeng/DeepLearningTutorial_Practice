import matplotlib.pyplot as plt
import torch

torch.manual_seed(16)  # for fixed model


# Generate random data for a quadratic function
def generate_quadratic_data(num_samples):
    # -5~5 平均產生 x
    x = torch.linspace(-5, 5, num_samples)
    # random: to make the dot more scatterring. 不讓資料太整齊 Try:
    # y = 2 * x**2 + 3 * x + 1
    # y = 2 * x**2 + 3 * x + 1 + torch.randn(num_samples) * 5
    y = 2 * x ** 2 + 3 * x + 1 + torch.randn(num_samples) * 2
    return x, y


# Define a quadratic function model: 微分generate_quadratic_data
def quadratic_model(x, params):
    a, b, c = params
    return a * x ** 2 + b * x + c


# Define the loss function (mean squared error)
def mean_squared_error(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)


def gradient_descent(x, y, learning_rate, num_epochs):
    # Initialize model parameters randomly
    params = torch.randn(3, requires_grad=True)

    for epoch in range(num_epochs):
        # Calculate the model predictions
        y_pred = quadratic_model(x, params)

        # Calculate the loss function
        loss = mean_squared_error(y, y_pred)

        # Calculate gradients and update parameters
        loss.backward()
        params.data -= learning_rate * params.grad.data

        # Clear the gradient
        params.grad.zero_()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    return params


# Generate data
num_samples = 100
x, y = generate_quadratic_data(num_samples)

learning_rate = 0.001
num_epochs = 1000

optimal_params = gradient_descent(x, y, learning_rate, num_epochs)

# Plot the fitted curve
x_test = torch.linspace(-5, 5, 100)
print('x_test = ', x_test)
y_pred = quadratic_model(x_test, optimal_params)

plt.figure(figsize=(8, 6))
plt.scatter(x.numpy(), y.numpy(), label='Real data')
plt.plot(x_test.numpy(), y_pred.detach().numpy(), 'r-', label='Fit curve')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

print(f"Fitted parameters：a={optimal_params[0]:.2f}, b={optimal_params[1]:.2f}, c={optimal_params[2]:.2f}")
