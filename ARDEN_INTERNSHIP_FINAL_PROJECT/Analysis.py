import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

st.set_page_config(page_title="Netflix Data Analysis", layout="wide")
# Load the dataset
st.title("Netflix Data Analysis")
def load_data():
    df = df = pd.read_csv(os.path.join(os.path.dirname(ARDEN_INTERNSHIP_FINAL_PROJECT), "netflix1.csv"))
    return df
df = load_data()

#df = pd.read_csv("netflix1.csv")
#Ovserving the data and Cleaning it
print(df.head())
print(df.columns)
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.nunique())
df.drop_duplicates(inplace=True)
df['country'].fillna('Unknown', inplace=True)
df['date_added'].fillna(method='ffill', inplace=True)
print(df.tail())

print(df.describe())
df['type'] = df['type'].str.strip()
df['country'] = df['country'].str.strip()
df['rating'] = df['rating'].str.strip().str.upper()
print(df.info())
d=pd.DataFrame(df)
print(d)
# Extracting year and month from date_added
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
df['year_added']=df['date_added'].dt.year
df['month_added']=df['date_added'].dt.month
# String convertions and stripping whitespace
df['date_added'] = df['date_added'].dt.strftime('%Y-%m-%d')
df['country']=df['country'].astype(str).str.strip()
df['rating']=df['rating'].astype(str).str.strip()
df['type']=df['type'].astype(str).str.strip()

#Genere counts
df.rename(columns={'listed_in':'genre'}, inplace=True)
df['genre']=df['genre'].astype(str).str.strip()
genre_counts = df['genre'].str.split(',').explode().str.strip().value_counts()
column_name = 'genre'
filter_list = sorted(df[column_name].dropna().unique().tolist())
print(genre_counts)

#year wise counts of shows
year_counts=df['year_added'].value_counts().sort_index()
print(year_counts)

#country wise counts of shows
country_counts=df['country'].value_counts().sort_index()
print(country_counts)

#rating wise counts of shows
rating_counts=df['rating'].value_counts().sort_index()
print(rating_counts)

#Duration counts
df['duration'] = df['duration'].astype(str).str.strip()
df[['duration_value','duration_unit']]=df['duration'].str.extract(r'(\d+)\s*(\D+)')
df['duration_value'] = pd.to_numeric(df['duration_value'], errors='coerce')
duration_counts = df['duration_unit'].value_counts()
print(duration_counts)

#Average duration of shows
avag_duration = df.groupby('type')['duration_value'].mean().reset_index()
print(avag_duration)




  # it present the frontend part#



   

st.subheader('Dataset summary')

st.write(df.describe())
st.sidebar.header("Filter options")
type=st.sidebar.multiselect("select type of content",
                          options=df["type"].unique(),
                          default=df["type"].unique()
                          )

genre=st.sidebar.markdown("### Filter by Genre (Alphabetical with Search)")
selected_options = st.sidebar.multiselect(
    "Select one or more genres:",
    options=filter_list,
    default=None
)
# Filter the DataFrame based on user selections
if selected_options:
    filtered_df = df[df[column_name].isin(selected_options)]
elif type:
    filtered_df = df[df["type"].isin(type)]    
else:
    filtered_df = df

# Display the DataFrame in Streamlit
st.subheader("Filtered DataFrame")
st.dataframe(filtered_df)

#Visualizations
st.header("Visualizations")
#BarChart
def barChart(genre_counts):
    top_genres = genre_counts.head(10)
    fig,ax=plt.subplots(figsize=(8, 10),facecolor="#7EACED")
    ax.set_facecolor("#87C1E8") 
    sns.barplot(x=top_genres.index, y=top_genres.values, palette='viridis',ax=ax)
    ax.set_title('Genre Counts')
    ax.set_xlabel('Genre')
    ax.set_ylabel('Count')
    plt.xticks(rotation=90,ha='center')
    plt.tight_layout()
    st.subheader('Bar Chart of Top 10 Genres')
    st.write("This bar chart shows the top 10 genres available on Netflix. The x-axis represents the genre, and the y-axis represents the count of shows in each genre.")
    st.pyplot(fig)
barChart(genre_counts) 


#heatmap
def heatmap():
    country_genre=df[['country', 'genre']].dropna()
    country_genre=country_genre.assign(genre=country_genre['genre'].str.split(',')).explode('genre')
    country_genre['genre'] = country_genre['genre'].str.strip()
    counties_counts=country_genre['country'].value_counts().sort_index()
    genre_counts=country_genre['genre'].value_counts().sort_index()

    #top 10 countries and genres
    top_countries = counties_counts.head(10).index
    top_genres = genre_counts.head(10).index
    country_genre = country_genre[country_genre['country'].isin(top_countries) & country_genre['genre'].isin(top_genres)]
    # Create a pivot table
    pivot_table = country_genre.pivot_table(index='country', columns='genre', aggfunc='size', fill_value=0)
    # Create a heatmap

    fig,ax=plt.subplots(figsize=(9, 6) )
    sns.heatmap(pivot_table, cmap='YlGnBu', annot=True, fmt='d', linewidths=.5,cbar=False, ax=ax)
    ax.set_title('Heatmap of Country vs Genre')
    ax.set_xlabel('Genre')
    ax.set_ylabel('Country')
    st.subheader('Heatmap of Country vs Genre(TOP 10)')
    st.write("This heatmap shows the distribution of genres across different countries. The values represent the count of shows in each genre for each country.")
    st.pyplot(fig)
    
    
heatmap()
#Timeline plot
def plot_timeline():
    # Count how many shows were added per year
    content_year = df['year_added'].value_counts().sort_index()
    content_year.index = pd.to_datetime(content_year.index, format='%Y')
    content_year = content_year.sort_index()  

    # Create a DataFrame for Plotly
    timeline_df = pd.DataFrame({
        'Year': content_year.index,
        'Number of Shows Added': content_year.values
    })

    # Create interactive line plot
    fig = px.line(
        timeline_df,
        x='Year',
        y='Number of Shows Added',
        markers=True,
        title='Timeline of Content Added to Netflix',
        hover_data={'Year': True, 'Number of Shows Added': True}
    )

    fig.update_traces(line=dict(color='red'), marker=dict(size=8))
    fig.update_layout(
        plot_bgcolor='#9FB3DF',
        paper_bgcolor='#4E6688',
        font=dict(color='white'),
        xaxis_title='Year',
        yaxis_title='Number of Shows Added',
        hovermode='x unified'
    )

    st.subheader('Timeline of Content Added to Netflix')
    st.write("This timeline shows the number of shows added to Netflix each year. Hover over the points to see detailed info.")
    st.plotly_chart(fig, use_container_width=True)
plot_timeline()  
st.write("This is a simple Netflix data analysis app. It allows users to explore Netflix content by genre, type, and other attributes. The app includes visualizations such as bar charts, heatmaps, and timelines to provide insights into the data.")
st.write("You can filter the data by genre and type using the sidebar options. The app also provides a summary of the dataset, including basic statistics and information about missing values.")
st.subheader("About the Developer")
st.write("Name: Arghajit Tudu \nCollege:Scottish Church College Dept.: Computer Science")
