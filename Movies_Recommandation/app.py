import numpy as np
import pandas as pd

from PIL import Image
import streamlit as st

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, linear_kernel, manhattan_distances, polynomial_kernel

#open_embedded dataset
movies = pd.read_csv("movies_embedded_V2.csv")
movies["embedded"] = movies["embedded"].apply(lambda x : [float(x) for x in x.replace("[", "").replace ("]", "").split()])

#function similarities
model = SentenceTransformer('all-MiniLM-L6-v2')
def embedded_text(df, text, metrica):

    embed_mat = np.array([x for x in df["embedded"]])

    embedding = model.encode(text)

    m = np.array([embedding]) * len(df)
    sim_mat = metrica(m, embed_mat)

    df['sim_score'] = sim_mat[0]

    similarity_results = df.sort_values('sim_score', ascending = False)

    return similarity_results


#st.title("Movies Recommendation")
st.markdown("<h1 style='text-align: center; color: blue;'>Movies Recommendation</h1>", unsafe_allow_html=True)

img = Image.open("img/cinema.jpg")
st.image(img)

metric_select = st.selectbox("Selection of the similarity metric", ["Cosine Similarity", "Euclidean Distances", "Linear Kernel", "Manhattan Distances", "Polynomial Kernel"])

text_input = st.text_area("Choose the movie", height = 100)

if st.button("Recommandation"):
    
    if metric_select == "Cosine Similarity":
        similares = embedded_text(movies, text_input, cosine_similarity) #session.state
        sim = similares.head(10)
        for title, directeur, plot in zip(sim["title"],sim["directors"],sim["plot"]):
            st.write(f"**{title}** ({directeur[2:-2]})")
            #st.write(directeur[2:-2])
            st.write(f"*{plot}*")
            st.write("##")
            
    if metric_select == "Euclidean Distances":
        similares = embedded_text(movies, text_input, euclidean_distances) #session.state
        sim = similares.head(10)
        for title, directeur, plot in zip(sim["title"],sim["directors"],sim["plot"]):
            st.write(f"**{title}** ({directeur[2:-2]})")
            #st.write(directeur[2:-2])
            st.write(f"*{plot}*")
            st.write("##")
            
    if metric_select == "Haversine Distances":
        similares = embedded_text(movies, text_input, linear_kernel) #session.state
        sim = similares.head(10)
        for title, directeur, plot in zip(sim["title"],sim["directors"],sim["plot"]):
            st.write(f"**{title}** ({directeur[2:-2]})")
            #st.write(directeur[2:-2])
            st.write(f"*{plot}*")
            st.write("##")
            
    if metric_select == "Manhattan Distances":
        similares = embedded_text(movies, text_input, manhattan_distances) #session.state
        sim = similares.head(10)
        for title, directeur, plot in zip(sim["title"],sim["directors"],sim["plot"]):
            st.write(f"**{title}** ({directeur[2:-2]})")
            #st.write(directeur[2:-2])
            st.write(f"*{plot}*")
            st.write("##")
            
    if metric_select == "Polynomial Kernel":
        similares = embedded_text(movies, text_input, polynomial_kernel) #session.state
        sim = similares.head(10)
        for title, directeur, plot in zip(sim["title"],sim["directors"],sim["plot"]):
            st.write(f"**{title}** ({directeur[2:-2]})")
            #st.write(directeur[2:-2])
            st.write(f"*{plot}*")
            st.write("##")