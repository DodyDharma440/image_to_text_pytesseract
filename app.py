import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image

def preprocess_image(image, method):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    if method == "Grayscale":
        return gray
    elif method == "Thresholding":
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        return thresh
    elif method == "Adaptive Thresholding":
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        return thresh
    return img

def main():
    st.title("OCR Sederhana dengan Tesseract")
    st.write("Unggah gambar untuk mengenali dan mengekstrak teks secara otomatis menggunakan teknologi OCR (Optical Character Recognition) dengan Tesseract.")
    
    uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "png", "jpeg"])
    method = st.selectbox("Pilih metode preprocessing", ["None", "Grayscale", "Thresholding", "Adaptive Thresholding"])
    
    if uploaded_file is not None:
        with st.expander("Preview Gambar", expanded=True):
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang diunggah", use_container_width=True)
            
            processed_img = preprocess_image(image, method)
            st.image(processed_img, caption="Gambar setelah preprocessing", use_container_width=True, clamp=True)
        
        if st.button("Ekstrak Teks", use_container_width=True, type="primary"):
            text = pytesseract.image_to_string(processed_img, config="--psm 6")
            st.subheader("Hasil OCR:")
            st.code(text, language="markdown", height=400)
            st.info(f"Jumlah karakter: {len(text)}")
            
            st.download_button(
                label="Download text",
                data=text,
                file_name="Hasil OCR.txt",
                on_click="ignore",
                type="primary",
                icon=":material/download:",
                use_container_width=True
            )
                
            
if __name__ == "__main__":
    main()
