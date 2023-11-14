import streamlit as st
import cv2
import struct
import bitstring
import numpy  as np
from PIL import Image
import dct.zigzag as zz
import dct.data_embedding as stego
import dct.image_preparation   as img


def decode(image):
    stego_pil = Image.open(image)

    # Konversi gambar PIL ke array NumPy
    stego_pil = np.array(stego_pil)

    stego_pil = cv2.cvtColor(stego_pil, cv2.COLOR_RGB2BGR)

    stego_image_f32 = np.float32(stego_pil)
    stego_image_YCC = img.YCC_Image(cv2.cvtColor(stego_image_f32, cv2.COLOR_BGR2YCrCb))

    # FORWARD DCT STAGE
    dct_blocks = [cv2.dct(block) for block in stego_image_YCC.channels[0]]  # Only care about Luminance layer

    # QUANTIZATION STAGE
    dct_quants = [np.around(np.divide(item, img.JPEG_STD_LUM_QUANT_TABLE)) for item in dct_blocks]

    # Sort DCT coefficients by frequency
    sorted_coefficients = [zz.zigzag(block) for block in dct_quants]

    # DATA EXTRACTION STAGE
    recovered_data = stego.extract_encoded_data_from_DCT(sorted_coefficients)

    # Determine length of secret message
    data_len = int(recovered_data.read('uint:32') / 8)

    # Extract secret message from DCT coefficients
    extracted_data = bytes()
    for _ in range(data_len): extracted_data += struct.pack('>B', recovered_data.read('uint:8'))

    extracted_message = extracted_data.decode('ascii')
    # Print secret message back to the user
    print(extracted_message)

    return extracted_message

st.set_page_config(
    page_title="Decoder",
    page_icon="👋",
)

st.write("# Welcome to Streamlit! 👋")

# Masukan gambar
image = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

# Tombol
button_clicked = st.button("Proses")

# Aksi ketika tombol ditekan
if button_clicked:
  # Lakukan sesuatu dengan gambar dan teks
  if image is not None:
    st.image(image, caption="Gambar yang dipilih", use_column_width=True)

    with st.spinner('Extracting Message...'):
        message = decode(image)
        st.success(f'Pesan tersembunyi = {message}')