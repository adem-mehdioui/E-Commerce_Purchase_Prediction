import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import joblib

def main():
    
    st.title('E-commerce Dataset Visualization')

        # Add some text
    st.write('Objectif : Analyse des tendances de vente des produits sur notre site E-commerce')

# Data preprocessing steps
    preprocessing_steps = [
        
        "Count number of Missing values",
        "Replace Nan and missing values",
        "Check if there are duplicate values then removing them",
        "divide category_code into 4 sub_categories",
        "convert column event_time to date and hour",
        "Saving the cleaned dataframe"

    ]

    # Display header
    st.header("Pré-traitement de données collectées")

    # Display bullet points of preprocessing steps
    st.write("Les étapes de pré-traitement des données collectées:")
    st.write("- " + "\n- ".join(preprocessing_steps))



    
    # Load the dataset
    df = pd.read_csv('Cleaned_data.csv', sep=",")
    pd.set_option("display.max_column",None)

    # Assuming df is your DataFrame
    df.drop('dot_count', axis=1, inplace=True)

    #convert category_id from int to string , because The value is larger than the maximum supported integer values in number columns (2^53).
   # Assuming df is your DataFrame and 'column_name' is the column you want to convert
    df['category_id'] = df['category_id'].astype(str)
    df['user_id'] = df['user_id'].astype(str)
    

    # Display the dataset - first 50 rows -
    st.header('Cleaned Dataset')
    st.write(df.head(50))



    #let's created visualizations


    # Convert 'Date' column to datetime if it's not already
    df['Date'] = pd.to_datetime(df['Date'])

    # Extract unique months from the 'Date' column
    df['Month'] = df['Date'].dt.to_period('M')

    

    # 1. Event Type Distribution

    st.header('1. Event Type Distribution over time')
   
    unique_months = df['Month'].unique()
    # Create a dictionary to map month names to month integers
    month_names = {str(month): month for month in unique_months}

    # Add a slider for month selection
    selected_month = st.select_slider(
        "Select Month",
        options=list(month_names.keys())
    )

# Filter the DataFrame based on the selected month
    selected_month_int = month_names[selected_month]
    filtered_df = df[df['Date'].dt.to_period('M') == selected_month_int]

    event_type_count = filtered_df['event_type'].value_counts()
    fig = px.pie(values=event_type_count, names=event_type_count.index, title=f'Event Type Distribution for {selected_month}')
    st.plotly_chart(fig)




    # 2. Total Sales Line Chart
    st.header('2- Total Sales')

    total_sales_monthly = df.groupby('Month')['price'].sum().reset_index()
    total_sales_monthly['Month'] = total_sales_monthly['Month'].astype(str)  # Convert Period objects to strings
    fig2 = px.line(total_sales_monthly, x='Month', y='price'    )
    st.plotly_chart(fig2)



    # 3. Most Purchased Products by Category (Bar Chart)
    st.header('3- Most Purchased Products by Sub-Category')

    most_purchased_products = filtered_df.groupby(['cat_1', 'product_id']).size().reset_index(name='count')
    most_purchased_products = most_purchased_products.sort_values(by='count', ascending=False)
    most_purchased_products = most_purchased_products.groupby('cat_1').head(1)  # Get the most purchased product in each category

    #Exclude No Category from the visualization
    filtered_most_purchased_products = most_purchased_products[most_purchased_products['cat_1'] != 'No Category']

    fig3 = px.bar(filtered_most_purchased_products, x='cat_1', y='count', color='cat_1', title='Most Purchased Products by Sub-Category')
    st.plotly_chart(fig3)



     # 2. Total Sales Line Chart
    st.header('4- Total View Events Over Time')

    # Filter the dataset to include only 'view' events
    df_views = df[df['event_type'] == 'view']

    # Convert 'event_time' to datetime
    df_views['event_time'] = pd.to_datetime(df_views['event_time'])

    # Group by date and count the number of 'view' events
    view_events_over_time = df_views.groupby(df_views['event_time'].dt.date).size().reset_index(name='view_count')

    # Plot the results
    fig = px.line(view_events_over_time, x='event_time', y='view_count', labels={'event_time': 'Date', 'view_count': 'Number of Views'}, title='User Activity: Exploring View Events Over Time')
    st.plotly_chart(fig)




# Prediction Section
    st.header('Purchase Prediction')

    

    # Load the trained model
    try:
        model = joblib.load('decision_tree_model.joblib')

        # Input user_id and product_id for prediction
        user_id = st.text_input('Enter user ID')
        product_id = st.text_input('Enter product ID')

        if st.button('Predict'):
            if user_id and product_id:
                # Ensure the columns exist in the DataFrame
                if 'p_views' in df.columns and 'p_carts' in df.columns and 'p_purchases' in df.columns:
                    user_data = df[(df['user_id'] == user_id) & (df['product_id'] == product_id)][['p_views', 'p_carts', 'p_purchases']]
                    if not user_data.empty:
                        prediction = model.predict(user_data)
                        probability = model.predict_proba(user_data)[:, 1]
                        result = 'Purchase' if prediction[0] == 1 else 'No Purchase'
                        st.write(f"Prediction for user ID {user_id} and product ID {product_id}: {result}")
                        st.write(f"Probability: {probability[0]:.2f}")
                    else:
                        st.write(f"No data available for user ID {user_id} and product ID {product_id}")
                else:
                    st.write("The user will not buy the selected product.")
            else:
                st.write("Please enter both user ID and product ID")
    except FileNotFoundError:
        st.write("Model not found. Please ensure the model has been trained and saved correctly.")

    st.header("Conclusion :")
    st.write("- Les résultats montrent que le modèle a une performance décente avec un AUC de 0.75, ce qui indique une bonne capacité de distinction entre les classes")
    st.write("- Précision de la classe 0 (non-achat) est meilleure que le rappel, ce qui signifie que lorsque le modèle prédit un non-achat, il est souvent correct, mais il manque de nombreux vrais non-achats")
    st.write("- Rappel de la classe 1 (achat) est élevé, ce qui signifie que le modèle identifie correctement la plupart des achats, mais la précision est plus faible, indiquant un certain nombre de faux positifs.")







        
# Add JPG image
st.image('hetic.jpg', width=200)  # Adjust the width as needed

# Add custom CSS to position the image fixed in the top 
st.markdown(
    """
    <style>
    .fixed-image {
        position: fixed;
        top: 0;
        left: 0;
        z-index: -1;
        padding: 10px; /* Optional: Add padding to adjust the position */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create columns to place images and text horizontally
col1, col2, col3 = st.columns(3)

with col1:
    st.image('adem.jpg', caption='Data Analyst / Data Scientist', use_column_width=True)
    st.write('MEHDIOUI Mohamed Adem')

with col2:
    st.image('soulayman.jpg', caption='Data Analyst / Data Scientist', use_column_width=True)
    st.write('EL GUASMI Soulaymane')

with col3:
    st.image('yassine.jpg', caption='Coordinateur de Projet', use_column_width=True)
    st.write('CHARIT Mohamed Yassine')





if __name__ == '__main__':
    main()





