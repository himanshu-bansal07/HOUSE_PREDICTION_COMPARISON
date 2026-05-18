import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import gradio as gr
import os

# Global variables to store models and scaler
GLOBAL_MODELS = {}
GLOBAL_SCALER = None

def load_and_preprocess_data():
    print("Loading California Housing dataset...")
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target.reshape(-1, 1)

    np.random.seed(42)
    indices = np.random.choice(X.shape[0], 5000, replace=False)
    X = X[indices]
    y = y[indices]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
    X_test_scaled = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return cost

def batch_gradient_descent(X, y, learning_rate=0.01, epochs=500):
    m, n = X.shape
    theta = np.zeros((n, 1))
    cost_history = []
    start_time = time.time()
    for _ in range(epochs):
        predictions = X.dot(theta)
        gradients = (1 / m) * X.T.dot(predictions - y)
        theta -= learning_rate * gradients
        cost_history.append(compute_cost(X, y, theta))
    duration = time.time() - start_time
    return theta, cost_history, duration

def stochastic_gradient_descent(X, y, learning_rate=0.01, epochs=50):
    m, n = X.shape
    theta = np.zeros((n, 1))
    cost_history = []
    start_time = time.time()
    for epoch in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        for i in range(m):
            xi = X_shuffled[i:i+1]
            yi = y_shuffled[i:i+1]
            gradients = xi.T.dot(xi.dot(theta) - yi)
            
            # Use simple learning rate decay for stability
            decayed_lr = learning_rate / (1 + 0.001 * (epoch * m + i))
            theta -= decayed_lr * gradients
        cost_history.append(compute_cost(X, y, theta))
    duration = time.time() - start_time
    return theta, cost_history, duration

def mini_batch_gradient_descent(X, y, learning_rate=0.01, epochs=100, batch_size=64):
    m, n = X.shape
    theta = np.zeros((n, 1))
    cost_history = []
    start_time = time.time()
    for _ in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        for i in range(0, m, batch_size):
            xi = X_shuffled[i:i+batch_size]
            yi = y_shuffled[i:i+batch_size]
            gradients = (1 / len(xi)) * xi.T.dot(xi.dot(theta) - yi)
            theta -= learning_rate * gradients
        cost_history.append(compute_cost(X, y, theta))
    duration = time.time() - start_time
    return theta, cost_history, duration

def train_and_plot():
    global GLOBAL_MODELS, GLOBAL_SCALER
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    GLOBAL_SCALER = scaler
    
    print("Training models for comparison...")
    bgd_theta, bgd_cost, bgd_time = batch_gradient_descent(X_train, y_train, learning_rate=0.05, epochs=500)
    sgd_theta, sgd_cost, sgd_time = stochastic_gradient_descent(X_train, y_train, learning_rate=0.01, epochs=50)
    mbgd_theta, mbgd_cost, mbgd_time = mini_batch_gradient_descent(X_train, y_train, learning_rate=0.05, epochs=100)
    
    GLOBAL_MODELS['BGD'] = bgd_theta
    GLOBAL_MODELS['SGD'] = sgd_theta
    GLOBAL_MODELS['MBGD'] = mbgd_theta
    
    print("\n--- Training Results (Mean Squared Error) ---")
    print(f"BGD:  Test MSE = {compute_cost(X_test, y_test, bgd_theta):.4f} | Time = {bgd_time:.4f}s")
    print(f"SGD:  Test MSE = {compute_cost(X_test, y_test, sgd_theta):.4f} | Time = {sgd_time:.4f}s")
    print(f"MBGD: Test MSE = {compute_cost(X_test, y_test, mbgd_theta):.4f} | Time = {mbgd_time:.4f}s")
    print("---------------------------------------------\n")
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(500), bgd_cost, label='Batch GD (BGD)', color='blue')
    plt.plot(range(50), sgd_cost, label='Stochastic GD (SGD)', color='red')
    plt.plot(range(100), mbgd_cost, label='Mini-Batch GD (MBGD)', color='green')
    plt.title('Convergence Comparison of Gradient Descent Techniques')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (Cost)')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.abspath('gd_comparison.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def predict_house_price(med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, lat, lon):
    input_features = np.array([[med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, lat, lon]])
    scaled_features = GLOBAL_SCALER.transform(input_features)
    scaled_features_with_bias = np.c_[np.ones((scaled_features.shape[0], 1)), scaled_features]
    
    predictions = {}
    for name, theta in GLOBAL_MODELS.items():
        pred = scaled_features_with_bias.dot(theta)[0][0]
        actual_price = max(0, pred * 100000)
        predictions[name] = f"${actual_price:,.2f}"
        
    return predictions['BGD'], predictions['SGD'], predictions['MBGD']

# 1. Train models and generate plot
plot_path = train_and_plot()

# 2. Build the Gradio UI
with gr.Blocks() as ui:
    gr.Markdown("# 🏡 Gradient Descent Technique Comparison")
    gr.Markdown("Input housing features below to see how different Gradient Descent techniques predict the final house price differently. The graph shows how they converged during training.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Input Features (House Attributes)")
            med_inc = gr.Slider(0, 15, value=8.3, label="Median Income (in $10k)")
            house_age = gr.Slider(1, 100, value=41, step=1, label="House Age (years)")
            ave_rooms = gr.Slider(1, 15, value=7, step=1, label="Rooms")
            ave_bedrms = gr.Slider(1, 8, value=2, step=1, label="Bedrooms")
            population = gr.Slider(10, 10000, value=322, step=1, label="Population (in neighborhood)")
            ave_occup = gr.Slider(1, 15, value=3, step=1, label="Occupancy (number of people)")
            lat = gr.Slider(32, 42, value=37.88, label="Latitude")
            lon = gr.Slider(-125, -114, value=-122.23, label="Longitude")
            submit_btn = gr.Button("Predict Prices")
            
        with gr.Column():
            gr.Markdown("### Training Convergence Comparison")
            gr.Image(value=plot_path, label="Cost over Epochs")
            
            gr.Markdown("### Model Predictions")
            bgd_out = gr.Textbox(label="Batch GD Prediction")
            sgd_out = gr.Textbox(label="Stochastic GD Prediction")
            mbgd_out = gr.Textbox(label="Mini-Batch GD Prediction")
            
    submit_btn.click(
        fn=predict_house_price,
        inputs=[med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, lat, lon],
        outputs=[bgd_out, sgd_out, mbgd_out]
    )

if __name__ == "__main__":
    print("Launching Gradio interface...")

    port = int(os.environ.get("PORT", 7860))

    ui.launch(
        server_name="0.0.0.0",
        server_port=port
    )
