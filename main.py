from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import uvicorn
from io import BytesIO
from PIL import Image


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("cnn_model.h5")

classes=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def preprocess_image(image: Image.Image, target_size=(32, 32)):
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        processed_image = preprocess_image(image)

        print("Processed image shape:", processed_image.shape)

        #uploaded_img=Image.fromarray((processed_image[0] * 255).astype(np.uint8))
        #uploaded_img.save('C:\\Users\\hlp06\\Documents\\AiML learning\\object classification cifar10\\main.pyinput.jpg')

        predictions = model.predict(processed_image)
        print("Predictions:", predictions)
        predicted_class = classes[np.argmax(predictions)]
        print("Classes:",predicted_class)

        return JSONResponse({"prediction": predicted_class, "confidence": float(np.max(predictions))})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
