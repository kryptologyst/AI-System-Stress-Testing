"""
Streamlit demo for AI System Stress Testing.

This module provides an interactive web interface for running stress tests
and visualizing results in real-time.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models import ModelFactory
from src.attacks import AdversarialAttacker
from src.uncertainty import UncertaintyQuantifier
from src.ood import OODDetector
from src.utils import set_seed, get_device


# Page configuration
st.set_page_config(
    page_title="AI System Stress Testing",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🧪 AI System Stress Testing</h1>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="warning-box">
    <h4>⚠️ Important Disclaimer</h4>
    <p>This tool is for <strong>research and educational purposes only</strong>. 
    XAI outputs may be unstable or misleading and should not be used for 
    regulated decisions without human review. Results are not a substitute 
    for human judgment.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("Configuration")

# Data configuration
st.sidebar.subheader("📊 Data Settings")
dataset_type = st.sidebar.selectbox(
    "Dataset Type",
    ["Synthetic Classification", "Iris", "Wine", "Breast Cancer"],
    index=0
)

if dataset_type == "Synthetic Classification":
    n_samples = st.sidebar.slider("Number of Samples", 100, 2000, 1000)
    n_features = st.sidebar.slider("Number of Features", 5, 50, 20)
    n_classes = st.sidebar.slider("Number of Classes", 2, 10, 2)
    noise_level = st.sidebar.slider("Noise Level", 0.0, 0.5, 0.1)
else:
    n_samples = 1000
    n_features = 20
    n_classes = 2
    noise_level = 0.1

# Model configuration
st.sidebar.subheader("🤖 Model Settings")
model_type = st.sidebar.selectbox(
    "Model Type",
    ["Simple MLP", "Robust MLP"],
    index=0
)

epochs = st.sidebar.slider("Training Epochs", 10, 200, 50)
learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")

# Attack configuration
st.sidebar.subheader("⚔️ Attack Settings")
attack_methods = st.sidebar.multiselect(
    "Attack Methods",
    ["FGSM", "PGD"],
    default=["FGSM", "PGD"]
)

max_epsilon = st.sidebar.slider("Maximum Epsilon", 0.01, 0.5, 0.2, format="%.2f")
epsilon_values = np.linspace(0.01, max_epsilon, 5)

# Uncertainty configuration
st.sidebar.subheader("🎯 Uncertainty Settings")
uncertainty_methods = st.sidebar.multiselect(
    "Uncertainty Methods",
    ["Monte Carlo Dropout", "Temperature Scaling"],
    default=["Monte Carlo Dropout"]
)

# OOD configuration
st.sidebar.subheader("🔍 OOD Detection Settings")
ood_methods = st.sidebar.multiselect(
    "OOD Methods",
    ["Energy-based", "Max Softmax", "Entropy"],
    default=["Energy-based"]
)

# Main content
if st.button("🚀 Run Stress Tests", type="primary"):
    
    # Set up progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Initialize
    status_text.text("Initializing...")
    set_seed(42)
    device = get_device()
    
    progress_bar.progress(10)
    
    # Generate data
    status_text.text("Generating data...")
    if dataset_type == "Synthetic Classification":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_redundant=2,
            n_informative=n_features-2,
            noise=noise_level,
            random_state=42
        )
    else:
        # For demo purposes, use synthetic data
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            noise=noise_level,
            random_state=42
        )
    
    # Preprocess data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    progress_bar.progress(20)
    
    # Create model
    status_text.text("Creating model...")
    model_name = "simple_mlp" if model_type == "Simple MLP" else "robust_mlp"
    model = ModelFactory.create_model(model_name, n_features, n_classes)
    model = model.to(device)
    
    progress_bar.progress(30)
    
    # Train model
    status_text.text("Training model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            progress_bar.progress(30 + (epoch / epochs) * 20)
    
    progress_bar.progress(50)
    
    # Evaluate clean performance
    status_text.text("Evaluating clean performance...")
    model.eval()
    with torch.no_grad():
        clean_outputs = model(X_test_tensor)
        clean_preds = torch.argmax(clean_outputs, dim=1)
        clean_accuracy = (clean_preds == y_test_tensor).float().mean().item()
    
    progress_bar.progress(60)
    
    # Run adversarial attacks
    status_text.text("Running adversarial attacks...")
    attacker = AdversarialAttacker(device, {})
    
    attack_results = {}
    for method in attack_methods:
        method_lower = method.lower()
        method_results = {}
        
        for epsilon in epsilon_values:
            if method_lower == "fgsm":
                X_adv = attacker.fgsm_attack(model, X_test_tensor, y_test_tensor, epsilon)
            elif method_lower == "pgd":
                X_adv = attacker.pgd_attack(model, X_test_tensor, y_test_tensor, epsilon)
            
            with torch.no_grad():
                adv_outputs = model(X_adv)
                adv_preds = torch.argmax(adv_outputs, dim=1)
                adv_accuracy = (adv_preds == y_test_tensor).float().mean().item()
                perturbation = torch.norm(X_adv - X_test_tensor, p=2, dim=1).mean().item()
            
            method_results[epsilon] = {
                'accuracy': adv_accuracy,
                'perturbation': perturbation
            }
        
        attack_results[method] = method_results
    
    progress_bar.progress(80)
    
    # Run uncertainty quantification
    status_text.text("Running uncertainty quantification...")
    uncertainty_quantifier = UncertaintyQuantifier(device, {})
    
    uncertainty_results = {}
    for method in uncertainty_methods:
        if method == "Monte Carlo Dropout":
            predictions, uncertainties = uncertainty_quantifier.monte_carlo_dropout(
                model, X_test_tensor, n_samples=50
            )
            uncertainty_results[method] = {
                'predictions': predictions.cpu().numpy(),
                'uncertainties': uncertainties.cpu().numpy()
            }
        elif method == "Temperature Scaling":
            uncertainty_quantifier.temperature_scaling(model, X_test_tensor, y_test_tensor)
            predictions = uncertainty_quantifier.calibrated_predictions(model, X_test_tensor)
            uncertainties = -(predictions * torch.log(predictions + 1e-8)).sum(dim=1)
            uncertainty_results[method] = {
                'predictions': predictions.cpu().numpy(),
                'uncertainties': uncertainties.cpu().numpy()
            }
    
    progress_bar.progress(90)
    
    # Run OOD detection
    status_text.text("Running OOD detection...")
    ood_detector = OODDetector(device, {})
    
    # Create OOD data
    X_ood = X_test_tensor + torch.randn_like(X_test_tensor) * 2.0
    y_ood = y_test_tensor
    
    ood_results = {}
    for method in ood_methods:
        if method == "Energy-based":
            id_scores = ood_detector.energy_based_detection(model, X_test_tensor)
            ood_scores = ood_detector.energy_based_detection(model, X_ood)
        elif method == "Max Softmax":
            id_scores = ood_detector.max_softmax_probability(model, X_test_tensor)
            ood_scores = ood_detector.max_softmax_probability(model, X_ood)
        elif method == "Entropy":
            id_scores = ood_detector.entropy_based_detection(model, X_test_tensor)
            ood_scores = ood_detector.entropy_based_detection(model, X_ood)
        
        ood_results[method] = {
            'id_scores': id_scores.cpu().numpy(),
            'ood_scores': ood_scores.cpu().numpy()
        }
    
    progress_bar.progress(100)
    status_text.text("Complete!")
    
    # Display results
    st.success("Stress testing completed successfully!")
    
    # Results tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "⚔️ Adversarial", "🎯 Uncertainty", "🔍 OOD Detection"])
    
    with tab1:
        st.subheader("Performance Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Clean Accuracy", f"{clean_accuracy:.4f}")
        
        with col2:
            avg_adv_acc = np.mean([results['accuracy'] for method_results in attack_results.values() 
                                 for results in method_results.values()])
            st.metric("Avg Adversarial Accuracy", f"{avg_adv_acc:.4f}")
        
        with col3:
            avg_uncertainty = np.mean([np.mean(results['uncertainties']) for results in uncertainty_results.values()])
            st.metric("Avg Uncertainty", f"{avg_uncertainty:.4f}")
        
        # Data distribution
        st.subheader("Data Distribution")
        fig = px.histogram(
            x=y_test,
            title="Test Set Class Distribution",
            labels={'x': 'Class', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Adversarial Attack Results")
        
        # Robustness curves
        fig = go.Figure()
        
        for method, method_results in attack_results.items():
            epsilons = list(method_results.keys())
            accuracies = [method_results[eps]['accuracy'] for eps in epsilons]
            fig.add_trace(go.Scatter(
                x=epsilons,
                y=accuracies,
                mode='lines+markers',
                name=f'{method} Attack',
                line=dict(width=3)
            ))
        
        fig.update_layout(
            title="Robustness Curves",
            xaxis_title="Epsilon (Attack Strength)",
            yaxis_title="Adversarial Accuracy",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Attack results table
        st.subheader("Attack Results Summary")
        attack_data = []
        for method, method_results in attack_results.items():
            for epsilon, results in method_results.items():
                attack_data.append({
                    'Method': method,
                    'Epsilon': epsilon,
                    'Accuracy': results['accuracy'],
                    'Perturbation': results['perturbation']
                })
        
        attack_df = pd.DataFrame(attack_data)
        st.dataframe(attack_df, use_container_width=True)
    
    with tab3:
        st.subheader("Uncertainty Quantification Results")
        
        for method, results in uncertainty_results.items():
            st.subheader(f"{method} Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Uncertainty distribution
                fig = px.histogram(
                    x=results['uncertainties'],
                    title=f"{method} - Uncertainty Distribution",
                    labels={'x': 'Uncertainty', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Accuracy vs Uncertainty
                max_probs = results['predictions'].max(axis=1)
                correct_predictions = (torch.argmax(torch.FloatTensor(results['predictions']), dim=1) == y_test_tensor).float().numpy()
                
                fig = px.scatter(
                    x=max_probs,
                    y=results['uncertainties'],
                    color=correct_predictions,
                    title=f"{method} - Accuracy vs Uncertainty",
                    labels={'x': 'Max Probability', 'y': 'Uncertainty'},
                    color_discrete_map={0: 'red', 1: 'green'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Out-of-Distribution Detection Results")
        
        for method, results in ood_results.items():
            st.subheader(f"{method} OOD Detection")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Score distributions
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=results['id_scores'],
                    name='In-Distribution',
                    opacity=0.7,
                    nbinsx=30
                ))
                fig.add_trace(go.Histogram(
                    x=results['ood_scores'],
                    name='Out-of-Distribution',
                    opacity=0.7,
                    nbinsx=30
                ))
                
                fig.update_layout(
                    title=f"{method} - Score Distributions",
                    xaxis_title="Detection Score",
                    yaxis_title="Count",
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Score separation
                id_mean = np.mean(results['id_scores'])
                ood_mean = np.mean(results['ood_scores'])
                separation = ood_mean - id_mean
                
                fig = go.Figure(data=[
                    go.Bar(x=['ID Mean', 'OOD Mean', 'Separation'], 
                          y=[id_mean, ood_mean, separation])
                ])
                
                fig.update_layout(
                    title=f"{method} - Score Statistics",
                    yaxis_title="Score Value",
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>AI System Stress Testing Demo | For research and educational purposes only</p>
    <p>⚠️ Results may be unstable and should not be used for regulated decisions</p>
</div>
""", unsafe_allow_html=True)
