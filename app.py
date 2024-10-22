import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util


courses_df = pd.read_csv('analytics_vidhya_courses.csv')


model = SentenceTransformer('all-MiniLM-L6-v2')


course_titles = courses_df['Title'].tolist()
course_embeddings = model.encode(course_titles, convert_to_tensor=True)

st.title("Smart Search for Free Courses on Analytics Vidhya")


query = st.text_input("Enter keywords to search for courses:")

if query:
    
    query_embedding = model.encode(query, convert_to_tensor=True)

   
    similarities = util.pytorch_cos_sim(query_embedding, course_embeddings)[0]
    top_results = similarities.argsort(descending=True)[:10]  

 
    filtered_courses = []
    for idx in top_results:
      
        idx = int(idx)
        
        title = courses_df.iloc[idx]['Title']
        lessons = courses_df.iloc[idx]['Lessons']
        price = courses_df.iloc[idx]['Price']
        link = courses_df.iloc[idx]['Link']

        
        if query.lower() in title.lower():
            filtered_courses.append({
                'Title': title,
                'Lessons': lessons,
                'Price': price,
                'Link': link
            })

    
    if filtered_courses:
        st.write(f"**Top Results for '{query}':**")
        for course in filtered_courses:
            st.write(f"**Title:** {course['Title']}")
            st.write(f"**Lessons:** {course['Lessons']}")
            st.write(f"**Price:** {course['Price']}")
            st.write(f"[Enroll Here](https://courses.analyticsvidhya.com{course['Link']})")
            st.write("---")
    else:
        st.write("No courses found matching your search query.")
else:
    st.write("Please enter a search query.")
