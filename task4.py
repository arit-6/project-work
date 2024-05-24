import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib


kmeans = joblib.load("Task1 & 2/kmeans.pkl")
scaler = joblib.load("Task1 & 2/scaler.pkl")
output_task2 = pd.read_csv(r'C:\Users\Arit\Desktop\New folder\Task1 & 2\task2.csv')
output_task3 = pd.read_csv(r"C:\Users\Arit\Desktop\New folder\Task 3\output.csv")

# Function to identify cluster for the example data point
def identify_cluster(data_point):
    # Standardize the data point
    data_point_scaled = scaler.transform(data_point.reshape(1, -1))
    
    # Predict cluster label for the data point
    cluster_label = kmeans.predict(data_point_scaled)[0]
    
    return cluster_label

# Function to explain why the data point belongs to the cluster
def explain_cluster(data_point):
    # Get cluster label for the data point
    cluster_label = identify_cluster(data_point)
    
    # Get centroid of the cluster
    centroid = kmeans.cluster_centers_[cluster_label]
    
    # Compare data point with centroid
    similarity = pd.Series(data_point).corr(pd.Series(centroid))
    
    return f"The data point belongs to Cluster {cluster_label}. Similarity with centroid: {similarity:.2f}"

# Main function
def main():
    st.title("TASK 4")
    
    st.write("### Task 1 Output")
    # Button to generate random numbers
    if st.button("Generate Random Numbers"):
        # Generate random numbers within the specified range
        random_numbers = np.random.randint(-95, -40, size=18)
        st.write("Randomly generated numbers:", ", ".join(map(str, random_numbers)))
        st.write("Copy and paste these numbers into the input field below to see the results.")

    # Input fields for user input
    input_data_point = st.text_input("Enter your data point (comma-separated values):")
    if input_data_point:
        data_point = np.array([int(x.strip()) for x in input_data_point.split(",")])

        # Identify cluster for the input data point
        cluster_label = identify_cluster(data_point)

        # Explain why the data point belongs to the identified cluster
        explanation = explain_cluster(data_point)

        # Display cluster label and explanation
        st.write(f"Data point belongs to Cluster {cluster_label}.")
        st.write("Explanation:")
        st.write(explanation)


    #Output for task 2
    st.write("### Task 2 Output")
    st.write("I chose Random Forest Algorithm for this task because Random Forest offers high accuracy, robustness to overfitting, handles missing values, provides feature importance, manages non-linear data, parallelizes training efficiently, is versatile across tasks, and less sensitive to noise, making it a preferred choice for various machine learning tasks")
    st.write("#### Accuracy for my model is 98%")
    st.dataframe(output_task2)

    #Output for Task 3
    st.write("### Task 3 Output")
    st.dataframe(output_task3)


if __name__ == "__main__":
    main()
