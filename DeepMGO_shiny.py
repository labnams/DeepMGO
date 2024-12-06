# Standard library imports
import os
import sys
import random
import pickle
import sklearn
from datetime import datetime
from pathlib import Path
from io import StringIO, BytesIO
import nest_asyncio

# Third-party imports
import numpy as np
import pandas as pd
import tensorflow as tf
from padelpy import from_smiles

# Visualization
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Deep Learning
import keras
from keras.models import Model, model_from_json
from keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten,
    Dense, Dropout, Reshape, Concatenate
)
from keras.optimizers import Adam

# Web Framework
import shiny
from shiny import App, render, reactive, ui

# Set random seed
random.seed(42)

# Constants and paths
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "Model")

# File paths
PADEL_COLUMNS_PATH = os.path.join(DATA_DIR, "descriptors", "padel_columns.csv")
SCALER_PATH = os.path.join(BASE_DIR, "scaler", "minmax_scaler.pkl")
FEATURE_SELECTOR_PATH = os.path.join(DATA_DIR, "descriptors", "feature_indices.csv")
MODEL_NAME = "DeepMGO_fs90"

# Load PaDEL columns
PADEL_COLUMNS = pd.read_csv(PADEL_COLUMNS_PATH).columns.tolist()[1:]  # Skip 'Name' column

# Load model and preprocessing components
def load_model_components():
    # Load MinMaxScaler
    scaler = pickle.load(open(SCALER_PATH, 'rb'))
    scaler.clip = False
    
    # Load feature indices
    feature_index = pd.read_csv(FEATURE_SELECTOR_PATH)
    selected_features = feature_index["Index"].tolist()[:-1]
    
    # Load model architecture with custom objects
    custom_objects = {
        'Input': Input,
        'Conv1D': Conv1D,
        'MaxPooling1D': MaxPooling1D,
        'Flatten': Flatten,
        'Dense': Dense,
        'Dropout': Dropout,
        'Reshape': Reshape,
        'Concatenate': Concatenate,
        'Model': Model
    }
    
    # Load model architecture
    json_file = open(os.path.join(MODEL_DIR, f"{MODEL_NAME}.json"), "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json, custom_objects=custom_objects)
    
    # Load weights
    model.load_weights(os.path.join(MODEL_DIR, f"{MODEL_NAME}.h5"))
    
    # Compile model
    def root_mean_squared_error(y_true, y_pred):
        return keras.backend.sqrt(keras.backend.mean(keras.backend.square(y_pred - y_true)))
    
    optimizer = keras.optimizers.Adam()
    model.compile(loss=root_mean_squared_error,
                 optimizer=optimizer,
                 metrics=['mse', 'mae'])
    
    return scaler, selected_features, model

# Function to process descriptors
def process_descriptors(desc_dict, padel_columns, selected_features, scaler):
    # Create DataFrame with all columns
    processed_dict = {}
    for col in padel_columns:
        value = desc_dict.get(col, '')
        try:
            processed_dict[col] = float(value) if value != '' else 0.0
        except ValueError:
            processed_dict[col] = 0.0
    
    # Convert to DataFrame to preserve column order
    df = pd.DataFrame([processed_dict])
    
    # Apply scaler transformation
    scaled_data = scaler.transform(df)
    
    # Select features
    return scaled_data[0, selected_features]

# Safe descriptor calculation
def safe_get_descriptors(smiles, padel_columns, selected_features, scaler):
    try:
        descriptors = from_smiles(smiles)
        return process_descriptors(descriptors, padel_columns, selected_features, scaler), None
    except Exception as e:
        return None, f"Error calculating descriptors: {str(e)}"

# SMILES validation
def validate_smiles(smiles):
    try:
        # Basic SMILES syntax validation
        if not smiles or len(smiles.strip()) == 0:
            return False, "Empty SMILES string"
            
        # Check for basic SMILES characters and patterns
        valid_chars = set('CNOPSFIBrclh[]()=-#@+.%0123456789\\/')
        if not all(c in valid_chars for c in smiles):
            return False, "Invalid characters in SMILES string"
            
        # Check for balanced parentheses and brackets
        if smiles.count('(') != smiles.count(')'):
            return False, "Unbalanced parentheses"
        if smiles.count('[') != smiles.count(']'):
            return False, "Unbalanced brackets"
            
        return True, None
    except Exception as e:
        return False, f"Error parsing SMILES: {str(e)}"

# UI definition
app_ui = ui.page_fluid(
    ui.div(
        ui.h1("DeepMGO Web Tool", style="color: #2c3e50; text-align: center; padding: 20px;"),
        ui.p("Predict MGO affinity using deep learning model: DeepMGO", 
             style="text-align: center; color: #7f8c8d;"),
        style="background-color: #ecf0f1; margin-bottom: 30px;"
    ),
    
    ui.div(
        ui.div(
            ui.input_text_area(
                "smiles",
                "Enter SMILES strings (one per line):",
                value="",
                width="100%",
                rows=5
            ),
            ui.input_text_area(
                "concentrations",
                "Enter concentrations (µM, one per line):",
                value="",
                width="100%",
                rows=5
            ),
            ui.div(
                ui.input_action_button(
                    "load_example",
                    "Load Example",
                    class_="btn-secondary"
                ),
                ui.input_action_button(
                    "clear_inputs",
                    "Clear",
                    class_="btn-secondary"
                ),
                ui.input_action_button(
                    "submit",
                    "Predict",
                    class_="btn-primary"
                ),
            ),
            ui.output_ui("error_message"),
            ui.row(
                ui.column(
                    6,
                    ui.div(
                        ui.output_ui("table_message"),
                        ui.output_data_frame("results_table"),
                        ui.download_button("download_csv", "Download CSV",
                            style="margin-top: 10px;"),
                        style="padding: 20px; background-color: #f8f9fa; border-radius: 5px;"
                    )
                ),
                ui.column(
                    6,
                    ui.div(
                        ui.output_ui("plot_message"),
                        ui.output_ui("results_plot"),
                        ui.download_button("download_svg", "Download SVG",
                            style="margin-top: 10px;"),
                        style="padding: 20px; background-color: #f8f9fa; border-radius: 5px;"
                    )
                )
            ),
            style="padding: 30px;"
        ),
        style="background-color: white; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1);"
    )
)

def server(input, output, session):
    # Load model components
    scaler, selected_features, model = load_model_components()
    
    # Store states
    has_error = reactive.Value(False)
    error_text = reactive.Value("")
    results_df = reactive.Value(pd.DataFrame())
    
    # Example data
    example_smiles = """CCO                # Ethanol
CC(=O)O            # Acetic acid
C1=CC=CC=C1        # Benzene
C1=CC=C2C(=C1)C(=CN2)CC(C(=O)O)N    # Tryptophan
C1=CC2=C(C=C1O)C(=CN2)CCN            # Serotonin"""

    example_concentrations = """1000
1000
1000
1000
1000"""

    @reactive.Effect
    @reactive.event(input.load_example)
    def load_example_data():
        ui.update_text_area("smiles", value=example_smiles)
        ui.update_text_area("concentrations", value=example_concentrations)

    @reactive.Effect
    @reactive.event(input.clear_inputs)
    def clear_input_data():
        ui.update_text_area("smiles", value="")
        ui.update_text_area("concentrations", value="")

    @output
    @render.ui
    def error_message():
        if has_error.get():
            return ui.div(
                ui.div(
                    ui.h4("Error:"),
                    ui.pre(error_text.get()),
                    style="background-color: #ffebee; padding: 20px; border-radius: 5px;"
                )
            )
        return ui.div()

    def process_single_smiles(smiles, concentration):
        # Calculate descriptors
        descriptors, error = safe_get_descriptors(smiles, PADEL_COLUMNS, selected_features, scaler)
        if error:
            return None, error
        
        # Reshape for CNN
        x = descriptors.reshape(1, -1, 1)
        xc = np.array([[[float(concentration)]]])
        
        # Predict
        prediction = model.predict([x, xc])[0][0]
        
        return prediction, None

    @reactive.Effect
    @reactive.event(input.submit)
    def update_results():
        # Show progress notification
        with ui.Progress(min=0, max=1) as p:
            p.set(message="Processing SMILES", detail="Initializing...", value=0)
            
            has_error.set(False)
            error_text.set("")
            
            if not input.smiles() or not input.concentrations():
                has_error.set(True)
                error_text.set("Please enter both SMILES strings and concentrations")
                return
            
            smiles_lines = [line.split('#')[0].strip() for line in input.smiles().split('\n') if line.strip()]
            concentration_lines = [line.strip() for line in input.concentrations().split('\n') if line.strip()]
            
            if len(smiles_lines) != len(concentration_lines):
                has_error.set(True)
                error_text.set("Number of SMILES strings and concentrations must match")
                return
            
            results = []
            total_smiles = len(smiles_lines)
            
            for idx, (smiles, concentration) in enumerate(zip(smiles_lines, concentration_lines)):
                # Update progress
                progress = (idx + 1) / total_smiles
                p.set(
                    value=progress,
                    message=f"Processing SMILES {idx + 1}/{total_smiles}",
                    detail=f"Processing: {smiles[:30]}..."
                )
                
                # Existing validation and processing code
                is_valid, error = validate_smiles(smiles)
                if not is_valid:
                    has_error.set(True)
                    error_text.set(f"Invalid SMILES: {smiles}\nError: {error}")
                    return
                
                try:
                    conc = float(concentration)
                    if conc <= 0:
                        raise ValueError("Concentration must be positive")
                except ValueError as e:
                    has_error.set(True)
                    error_text.set(f"Invalid concentration for {smiles}: {str(e)}")
                    return
                    
                prediction, error = process_single_smiles(smiles, concentration)
                if error:
                    has_error.set(True)
                    error_text.set(f"Error processing {smiles}: {error}")
                    return
                    
                activity_status = "Active" if prediction > -0.186 else "Inactive"
                results.append({
                    'SMILES': smiles,
                    'Concentration (µM)': float(concentration),
                    'Predicted Value (Z)': round(prediction, 4),
                    'Activity Status': activity_status
                })
            
            # Final progress update
            p.set(message="Finalizing results...", value=1)
            
            if results:
                df = pd.DataFrame(results)
                df = df.sort_values('Predicted Value (Z)', ascending=False)
                df.insert(0, 'Rank', range(1, len(df) + 1))
                results_df.set(df)

    @output
    @render.data_frame
    def results_table():
        df = results_df.get()
        if df.empty:
            return pd.DataFrame(columns=['Rank', 'SMILES', 'Concentration (µM)', 'Predicted Value (Z)', 'Activity Status'])
        return render.DataGrid(df)

    @output
    @render.ui
    def results_plot():
        df = results_df.get()
        if df.empty:
            # Create empty Plotly figure when no data
            fig = go.Figure()
            fig.add_annotation(
                text="No data to display",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
        else:
            try:
                # Create a Plotly figure with two traces for Active and Inactive
                active_mask = df['Predicted Value (Z)'] > -0.186
                
                fig = go.Figure()
                
                # Add Active compounds
                fig.add_trace(go.Bar(
                    x=df[active_mask]['Rank'],
                    y=df[active_mask]['Predicted Value (Z)'],
                    marker_color='#2ecc71',
                    name='Active'
                ))
                
                # Add Inactive compounds
                fig.add_trace(go.Bar(
                    x=df[~active_mask]['Rank'],
                    y=df[~active_mask]['Predicted Value (Z)'],
                    marker_color='#e74c3c',
                    name='Inactive'
                ))
                
                fig.update_layout(
                    title='Prediction Results',
                    xaxis_title='Rank',
                    yaxis_title='Predicted Value (Z)',
                    xaxis_tickangle=0,
                    template='plotly_white',
                    height=500,
                    margin=dict(t=50, l=50, r=50, b=100),
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="right",
                        x=0.99
                    )
                )
            except Exception as e:
                print(f"Error generating plot: {e}")
                # Return empty figure with error message
                fig = go.Figure()
                fig.add_annotation(
                    text=f"Error generating plot: {str(e)}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False
                )
        
        # Convert Plotly figure to div
        plot_div = fig.to_html(
            full_html=False,
            include_plotlyjs='cdn',
            config={'responsive': True}
        )
        return ui.HTML(plot_div)

    @output
    @render.download(filename="prediction_results.csv")
    def download_csv():
        df = results_df.get()
        if df.empty:
            return BytesIO(b"No data available")
        
        # Convert DataFrame to CSV in memory
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8')
        csv_buffer.seek(0)
        return csv_buffer

    @output
    @render.download(filename="prediction_plot.svg")
    def download_svg():
        df = results_df.get()
        if df.empty:
            fig = plt.figure()
            plt.title('No data to display')
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Separate active and inactive compounds
            active_mask = df['Predicted Value (Z)'] > -0.186
            
            # Plot active compounds
            ax.bar(df[active_mask]['Rank'], 
                   df[active_mask]['Predicted Value (Z)'], 
                   color='#2ecc71', 
                   label='Active')
            
            # Plot inactive compounds
            ax.bar(df[~active_mask]['Rank'], 
                   df[~active_mask]['Predicted Value (Z)'], 
                   color='#e74c3c', 
                   label='Inactive')
            
            ax.axhline(0, color='gray', linestyle='--')
            ax.set_title('Prediction Results')
            ax.set_xlabel('Rank')
            ax.set_ylabel('Predicted Value (Z)')
            ax.legend()
            plt.tight_layout()

        # Save to BytesIO buffer
        buf = BytesIO()
        plt.savefig(buf, format='svg')
        plt.close(fig)
        buf.seek(0)
        return buf

    @output
    @render.ui
    def table_message():
        if results_df.get().empty:
            return ui.div(
                ui.p("Please enter SMILES strings and concentrations, then click Predict to see results here.",
                     style="text-align: center; color: #666; font-style: italic;")
            )
        return ui.div()

    @output
    @render.ui
    def plot_message():
        if results_df.get().empty:
            return ui.div(
                ui.p("Prediction results will be visualized here after clicking Predict.",
                     style="text-align: center; color: #666; font-style: italic;")
            )
        return ui.div()

# Run the application
app = App(app_ui, server)

if __name__ == "__main__":
    nest_asyncio.apply()
    app.run()
