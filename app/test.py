# --- Imports ---
import streamlit as st          # Streamlit: quick web apps from Python scripts
from PIL import Image           # Pillow (PIL): for opening and manipulating images
import io                       # io.BytesIO: in-memory bytes buffer for downloads

# --- Page config (optional) ---
# Sets the page title (browser tab) and centers the content layout.
st.set_page_config(page_title="Image Preview + Metadata", layout="centered")

# --- Title and intro text ---
st.title("📸 Image Preview & Metadata")   # Big header on the page
st.write("Upload an image to preview it, view metadata, toggle grayscale, and download.")
# st.write can render plain text, Markdown, numbers, DataFrames, etc.

# --- File uploader widget ---
# Shows a file picker in the UI. `type` restricts to certain file extensions.
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Only run the image pipeline if the user selected a file.
if uploaded_file is not None:
    # --- Read the uploaded image with Pillow ---
    # `uploaded_file` is a file-like object. PIL can open it directly.
    image = Image.open(uploaded_file)

    # --- Preview section ---
    st.subheader("Preview")  # Smaller section heading
    # Display the image. `use_container_width=True` makes it responsive to the page width.
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # --- Metadata section ---
    st.subheader("Metadata")
    img_format = image.format or "Unknown"   # e.g., 'JPEG', 'PNG' (may be None for some sources)
    width, height = image.size               # (width, height) tuple in pixels
    st.write(f"- **Format:** {img_format}")  # Markdown works in st.write
    st.write(f"- **Size (pixels):** {width} × {height}")
    try:
        mode = image.mode                    # e.g., 'RGB', 'RGBA', 'L' (grayscale)
        st.write(f"- **Mode:** {mode}")
    except Exception:
        # Some images can fail to report a mode; hide errors from users.
        pass

    # --- Transform (grayscale toggle) ---
    st.subheader("Transform")
    # A simple checkbox returns True/False. When True, convert to grayscale.
    if st.checkbox("Convert to grayscale"):
        # 'L' mode in PIL = 8-bit grayscale
        gray = image.convert("L")
        st.image(gray, caption="Grayscale Preview", use_container_width=True)
        display_image = gray                 # this will be the version we let users download
    else:
        display_image = image                # otherwise, keep the original

    # --- Prepare the download (in-memory buffer) ---
    buf = io.BytesIO()                       # create a bytes buffer
    # If original had no format, default to PNG so save() knows what to write.
    save_format = image.format if image.format else "PNG"
    display_image.save(buf, format=save_format)  # write the image into the buffer
    byte_im = buf.getvalue()                 # get raw bytes for the download button

    # --- Download button ---
    # Creates a button that lets users download the currently displayed image.
    st.download_button(
        label="Download displayed image",                 # button text
        data=byte_im,                                     # bytes to download
        file_name=f"downloaded_image.{save_format.lower()}",  # suggested file name
        mime=f"image/{save_format.lower()}",              # content type for the browser
    )

    # Friendly confirmation to the user
    st.success("You can preview, transform, and download the image.")
else:
    # Shown before any file is uploaded
    st.info("No image uploaded yet — use the uploader above.")
