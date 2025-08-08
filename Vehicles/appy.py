import streamlit as st
import pandas as pd
import plotly.express as px

# Load data
car_data = pd.read_csv("Vehicles/vehicles_us_preprocessed.csv")  

st.title("🚗 Vehicle Data Preprocessing and Visualization")
st.write("Explore and visualize your preprocessed vehicle dataset interactively.")

# Show data types
with st.expander("📊 Data Types"):
    st.write(car_data.dtypes)

# Show basic statistics
with st.expander("📈 Basic Statistics"):
    st.write(car_data.describe())

# --- Interactive histogram ---
st.header("🔍 Explore Column Distributions")

# Column selector
numeric_cols = car_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
selected_col = st.selectbox("Select a numeric column to plot:", numeric_cols)

# Bin slider
nbins = st.slider("Number of bins:", min_value=10, max_value=100, value=50, step=5)

# Toggle plot display
show_plot = st.checkbox("Show Histogram", value=True)

if show_plot and selected_col:
    fig = px.histogram(car_data, x=selected_col, nbins=nbins, title=f"Distribution of {selected_col.capitalize()}")
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Check the box above to display the histogram.")

# --- Interactive scatter plot ---
st.header("📊 Scatter Plot of Two Numeric Columns")
# Column selectors for scatter plot
x_col = st.selectbox("Select X-axis column:", numeric_cols, index=0)
y_col = st.selectbox("Select Y-axis column:", numeric_cols, index=1)

# Toggle plot display
show_scatter = st.checkbox("Show Scatter Plot", value=True)

if show_scatter:
    fig = px.scatter(car_data, x=x_col, y=y_col, title=f"Scatter Plot of {y_col.capitalize()} vs {x_col.capitalize()}")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Check the box above to display the scatter plot.")

# --- Interactive correlation heatmap ---
st.header("🔗 Correlation Heatmap")

# Toggle heatmap display
#show_heatmap = st.checkbox("Show Correlation Heatmap", value=True)

#if show_heatmap:
    # Select numeric columns only
    #numeric_data = car_data.select_dtypes(include=['int64', 'float64', 'int32', 'uint8'])

    # Remove columns that might be misleading or not useful in correlation
    #columns_to_exclude = ['model_year', 'is_4wd', 'days_listed']  # adjust as needed
    #numeric_data = numeric_data.drop(columns=[col for col in columns_to_exclude if col in numeric_data.columns])

    # Compute and round correlation matrix
    #corr_matrix = numeric_data.corr().round(2)

    # Create heatmap
    #fig = px.imshow(
        #corr_matrix,
        #text_auto=True,
        #title="Correlation Heatmap",
        #aspect="auto"
    #)

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Check the box above to display the correlation heatmap.")
