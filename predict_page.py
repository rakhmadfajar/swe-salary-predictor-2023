import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

dec_tree = data['model']
le_cy = data['le_cy']
le_ed = data['le_ed']

kurs = 15860

def show_predict_page():
    st.title("Prediksi Gaji Software Engineer")
    st.write("Projek ini dapat menghasilkan estimasi gaji sesuai dengan informasi yang didapatkan. Metode prediksi menggunakan algoritma Decision Tree")
    st.write("Data gaji berasal dari Stack Overflow Developer Survey 2023")
    st.divider()

    countries = (
                'United States of America',
                'Germany',
                'United Kingdom of Great Britain and Northen Ireland',
                'Canada',
                'India',
                'France',
                'Netherlands',         
            )

    education = (
                'Bachelor degree',
                'Master degree',
                'Post graduate',
            )

    country = st.selectbox("Asal Negara", countries)
    edu = st.selectbox("Tingkat Pendidikan", education)

    experience = st.slider("Tahun Pengalaman", 0, 50, 3)

    ok = st.button("Hitung Prediksi Gaji")

    if ok:
        Xa = np.array([[country, edu, experience]])
        Xa[:, 0] = le_cy.transform(Xa[:, 0])
        Xa[:, 1] = le_ed.transform(Xa[:, 1])
        Xa = Xa.astype(float)
        
        
        salary = dec_tree.predict(Xa)[0] * kurs
        
        
        
        st.subheader("Estimasi Gaji per Tahun: Rp{:,.2f}".format(salary))
        st.subheader("Estimasi Gaji per Bulan: Rp{:,.2f}".format((salary)/12))

        st.write("Estimasi $1 = Rp15.860,00")