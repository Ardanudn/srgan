import streamlit as st
import cv2
import struct
import bitstring
import numpy  as np
import os
from PIL import Image
import dct.zigzag as zz
import dct.image_preparation as img
import dct.data_embedding as stego



def encode(image,message):
    NUM_CHANNELS = 3
    STEGO_IMAGE_FILEPATH  = "./results/stego/stego_image.png"
    SECRET_MESSAGE_STRING = message

    image_pil = Image.open(image)

    # Konversi gambar PIL ke array NumPy
    image_pil = np.array(image_pil)

    image_pil = cv2.cvtColor(image_pil, cv2.COLOR_RGB2BGR)

    # raw_cover_image = cv2.imread(COVER_IMAGE_FILEPATH, flags=cv2.IMREAD_COLOR)
    height, width   = image_pil.shape[:2]
    # Force Image Dimensions to be 8x8 compliant
    while(height % 8): height += 1 # Rows
    while(width  % 8): width  += 1 # Cols
    valid_dim = (width, height)
    padded_image    = cv2.resize(image_pil, valid_dim)
    cover_image_f32 = np.float32(padded_image)
    cover_image_YCC = img.YCC_Image(cv2.cvtColor(cover_image_f32, cv2.COLOR_BGR2YCrCb))

    # Placeholder for holding stego image data
    stego_image = np.empty_like(cover_image_f32)

    for chan_index in range(NUM_CHANNELS):
        # FORWARD DCT STAGE
        dct_blocks = [cv2.dct(block) for block in cover_image_YCC.channels[chan_index]]

        # QUANTIZATION STAGE
        dct_quants = [np.around(np.divide(item, img.JPEG_STD_LUM_QUANT_TABLE)) for item in dct_blocks]

        # Sort DCT coefficients by frequency
        sorted_coefficients = [zz.zigzag(block) for block in dct_quants]

        # Embed data in Luminance layer
        if (chan_index == 0):
            # DATA INSERTION STAGE
            secret_data = ""
            for char in SECRET_MESSAGE_STRING.encode('ascii'): secret_data += bitstring.pack('uint:8', char)
            embedded_dct_blocks   = stego.embed_encoded_data_into_DCT(secret_data, sorted_coefficients)
            desorted_coefficients = [zz.inverse_zigzag(block, vmax=8,hmax=8) for block in embedded_dct_blocks]
        else:
            # Reorder coefficients to how they originally were
            desorted_coefficients = [zz.inverse_zigzag(block, vmax=8,hmax=8) for block in sorted_coefficients]

        # DEQUANTIZATION STAGE
        dct_dequants = [np.multiply(data, img.JPEG_STD_LUM_QUANT_TABLE) for data in desorted_coefficients]

        # Inverse DCT Stage
        idct_blocks = [cv2.idct(block) for block in dct_dequants]

        # Rebuild full image channel
        stego_image[:,:,chan_index] = np.asarray(img.stitch_8x8_blocks_back_together(cover_image_YCC.width, idct_blocks))
    #-------------------------------------------------------------------------------------------------------------------#

    # Convert back to RGB (BGR) Colorspace
    stego_image_BGR = cv2.cvtColor(stego_image, cv2.COLOR_YCR_CB2BGR)

    # Clamp Pixel Values to [0 - 255]
    final_stego_image = np.uint8(np.clip(stego_image_BGR, 0, 255))

    # Write stego image
    cv2.imwrite(STEGO_IMAGE_FILEPATH, final_stego_image)

    stego_image_RGB = cv2.cvtColor(final_stego_image, cv2.COLOR_BGR2RGB)

    return stego_image_RGB


st.set_page_config(
    page_title="Encoder",
    page_icon="👋",
)

st.write("# Encoder Message to Image")

# Masukan gambar
image = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

# Masukan teks
text_input = st.text_input("Masukkan teks")

# Tombol
button_clicked = st.button("Proses")

# Aksi ketika tombol ditekan
if button_clicked:
  # Lakukan sesuatu dengan gambar dan teks
  if image and text_input is not None:
    

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Gambar yang dipilih", use_column_width=True)
    with col2:
        with st.spinner('Embedding Message...'):
            stego_image = encode(image,text_input)
            st.image(stego_image, caption="Model prediction")

            STEGO_IMAGE_FILEPATH  = "./results/stego/stego_image.png"

            with open(STEGO_IMAGE_FILEPATH, 'rb') as file:
                st.download_button(
                    label="Download Stego Image",
                    data=file,
                    file_name=os.path.basename(STEGO_IMAGE_FILEPATH),
                    mime="image/png")

    
