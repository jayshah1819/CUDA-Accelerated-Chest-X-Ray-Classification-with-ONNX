# CUDA-Accelerated-Chest-X-Ray-Classification-with-ONNX!



## About the Project

In this project, I built a **high-performance pipeline for chest X-ray analysis**. The goal was to speed up image preprocessing and model inference using **GPU acceleration** and **ONNX**, making it efficient enough to handle large datasets quickly.

Basically, I wanted to see how we can take medical images, prepare them efficiently, and run predictions fast — all while keeping things reproducible and clear.

---

## What I Did

1. **GPU-Powered Preprocessing**

   * Converted chest X-ray images to **grayscale** using **CUDA/Numba**, running directly on the GPU.
   * Normalized pixel values so the images are ready for the model.
   * Saved these preprocessed images as `.npy` files for easy reuse.

2. **ONNX Model Inference**

   * Took the preprocessed images, replicated channels to match DenseNet-121’s expected input, and resized them to **224×224**.
   * Ran predictions using **ONNXRuntime**, which is faster than standard PyTorch inference.
   * Recorded the **predicted class, confidence, and inference time** for each image.

3. **Performance Comparison**

   * Measured inference times for both **PyTorch and ONNX**.
   * Created bar charts to clearly show the speed improvement with ONNX.

---

## Project Structure

```
project-root/
│
├─ dataset/
│   └─ chest_xray/      # Original X-ray images
│
├─ preprocessed/
│   └─ *.npy            # GPU-preprocessed images
│
├─ notebooks/
│   └─ preprocessing_and_onnx_inference.ipynb
│
├─ models/
│   └─ densenet121.onnx
│
└─ README.md
```

---

## How to Use

1. **Preprocessing Images**

```python
gray_img = preprocess_image_gpu(img)
np.save("preprocessed/NORMAL/IM-0001.npy", batch_img)
```

2. **Running ONNX Inference**

```python
import onnxruntime as ort
ort_session = ort.InferenceSession("models/densenet121.onnx")
outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: batch_resized})
```

3. **Measuring Inference Time**

```python
start = time.time()
outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: batch_resized})
end = time.time()
inference_time_ms = (end - start) * 1000
```

---

## Tools and Libraries I Used

* **NumPy** – For array handling and batching.
* **OpenCV** – To resize and manipulate images.
* **Numba** – For CUDA-powered preprocessing.
* **ONNXRuntime** – For fast GPU inference.
* **PyTorch** – DenseNet-121 baseline comparison.
* **Matplotlib & Pandas** – For visualizations and tables.

---

## Results

* Preprocessing on GPU was **fast and efficient**.
* ONNX inference ran significantly **faster than PyTorch**, helping speed up predictions.
* The pipeline is **reproducible** and works well with larger datasets.

---

## Next Steps

* Implement **TensorRT acceleration** for even faster inference.
* Add **batch processing** to handle multiple images at once on HPC clusters.
* Explore **fine-tuning DenseNet** on pediatric X-ray datasets for better medical accuracy.

---

