import gzip
import shutil

import  streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Delhi House Price Prediction",
    page_icon="🏠",
    layout="wide"  # Use the full page width
)


# Add custom HTML and CSS to position the image
st.markdown(
    """
    <style>
    .custom-image {
        position: fixed;
        top: 10px;  /* Adjust vertical position */
        right: 10px;  /* Adjust horizontal position */
        width: 10px;  /* Adjust image width */
        height: 50px;  /* Maintain aspect ratio */
        z-index: 1000;  /* Ensure it's above other elements */
    }
    </style>
    """
    , unsafe_allow_html=True
)

# Load and display the image using st.image()
image_path = 'realestate_logo.png'  # Replace with the path to your image file

# Add a spacer to push the image to the top right corner
col1, col2, col3 = st.columns([15, 1, 1])  # Adjust the ratio as needed
with col1:
    pass  # Empty column to push image to the right
with col2:
    st.image(image_path, width=120)  # Display image in the center column
with col3:
    pass  # Empty column to push image to the left

# Set header
st.markdown("<h1 style='text-align: center; color: black;'>Welcome to Delhi House Price Prediction</h1>", unsafe_allow_html=True)

# Set description
st.markdown("""
    <div style='text-align: center;'>
        <p style='font-size: 24px;'>Find the best estimated market value of houses in Delhi with advanced machine learning algorithms.</p>
    </div>
""", unsafe_allow_html=True)


# CSS to inject contained in a string
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .reportview-container .main .block-container{padding-top: 0rem;}
        </style>
"""

# Inject CSS with Markdown
st.markdown(hide_menu_style, unsafe_allow_html=True)

# Set a larger sidebar width and different background color
st.markdown(
    """
    <style>
    .css-18e3th9 {
        padding: 1rem 1rem 10rem;  # Adjust padding to fit your needs
        background-color: #f0f2f6;  # Change the background color
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar header
st.sidebar.markdown("<h2 style='text-align: center; color: black;'>Input Parameters</h2>", unsafe_allow_html=True)

# Continue with your sidebar inputs...
# For example, use selectboxes with a uniform width and style
property_type = st.sidebar.selectbox('Property Type', ['flat', 'house'], index=0,
                                     format_func=lambda x: 'flat' if x == 'flat' else 'house')


# Apply custom CSS to the button if needed
st.markdown(
    """
    <style>
    .stButton>button {
        font-size: 1.5em;
        width: 100%;
        border-radius: 5px;
        color: white;
        background-color: #0083B8;  # Change button color
        height: 3em;
    }
    .stButton>button:hover {
        background-color: #005f73;  # Change hover color
    }
    </style>
""", unsafe_allow_html=True)


#property_type	sector	bedRoom	bathroom	balcony	agePossession	built_up_area	servant room	store room	furnishing_type	luxury_category	floor_category


zipped_pipeline_file_path = 'pipeline_1_zip.pkl.gz'
unzipped_pipeline_file_path = 'pipeline.pkl'

# Uncompress the file
with gzip.open(zipped_pipeline_file_path, 'rb') as zipped_file:
    with open(unzipped_pipeline_file_path, 'wb') as original_file:
        shutil.copyfileobj(zipped_file, original_file)


with open('df_1.pkl','rb') as file:
    df = pickle.load(file)

with open('pipeline.pkl','rb') as file:
    pipeline = pickle.load(file)

# Add a header or explanation about the app
st.markdown("""
    This tool predicts the price of houses in Delhi based on various features
    such as property type, location, number of bedrooms, and more.
""")

# Divide the selections into two columns
col1, col2 = st.columns(2)

# Sector (Housing Society)
housing_society = col1.selectbox('Housing Society', sorted(df['sector'].unique().tolist()))

# Number of Bedrooms
bedrooms = col1.selectbox('Number of Bedrooms', sorted(df['bedRoom'].unique().tolist()))

# Number of Bathrooms
bathroom = col1.selectbox('Number of Bathrooms', sorted(df['bathroom'].unique().tolist()))

# Balconies
balcony = col1.selectbox('Balconies', sorted(df['balcony'].unique().tolist()))

# Property Age
property_age = col1.selectbox('Property Age', sorted(df['agePossession'].unique().tolist()))

# Built Up Area
built_up_area = col2.number_input('Built Up Area (sq ft)',step = 1)

# Maid Room
Maid_Room = col2.selectbox('Maid Room', [0, 1])

# Store Room
storeroom = col2.selectbox('Store Room', [0, 1])

# Furnishing Type
furnishing_type = col2.selectbox('Furnishing Type', sorted(df['furnishing_type'].unique().tolist()))

# Luxury Category
luxury_category = col2.selectbox('Luxury Category', sorted(df['luxury_category'].unique().tolist()))

# Floor Category
floor_category = col1.selectbox('Floor Category', sorted(df['floor_category'].unique().tolist()))


if st.button('Predict'):
    data = [[property_type,housing_society,bedrooms,bathroom, balcony,property_age,built_up_area,Maid_Room,storeroom,furnishing_type,luxury_category,floor_category]]
    columns = ['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
               'agePossession', 'built_up_area', 'servant room', 'store room',
               'furnishing_type', 'luxury_category', 'floor_category']

    # Convert to DataFrame
    one_df = pd.DataFrame(data, columns=columns)

    #erleast.dataframe(one_df)

    Price = round(np.expm1(pipeline.predict(one_df))[0],2)

    # Format the output using CSS and add text
    st.markdown(
        f"<p style='font-size: 24px; color: #007BFF; font-family: Arial, sans-serif; font-weight: bold; text-shadow: "
        f"1px 1px 2px rgba(0, 0, 0, 0.5);'>Estimated Price: INR {Price} Cr</p>",
        unsafe_allow_html=True
    )



lat_long = pd.read_csv("properties lat price sqft.csv")

lat_long_group = lat_long.groupby('Sector').mean()[['Avg_price_per_sqft','Longitude','Latitude']]

fig = px.scatter_mapbox(lat_long_group, lat = "Latitude", lon = "Longitude", color = "Avg_price_per_sqft",
                        color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10,
                        mapbox_style = "carto-positron", width = 1500, height = 500 )
st.plotly_chart(fig, use_container_width=True)


