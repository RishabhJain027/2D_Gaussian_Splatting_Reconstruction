import os
import sys
import subprocess
import threading
import time
import gradio as gr
import torch
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GS_REPO_DIR = os.path.join(BASE_DIR, "2d-gaussian-splatting")
PYTHON_EXE = sys.executable

def check_system_status():
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "No CUDA Device Detected"
    return f"""### 🚀 System Status & Hardware
- **PyTorch Version**: `{torch.__version__}`
- **CUDA Available**: `{"✅ Yes" if cuda_avail else "❌ No"}`
- **GPU Accelerator**: `{device_name}`
- **Active Python**: `{PYTHON_EXE}`
- **2DGS Source Directory**: `{GS_REPO_DIR}`
"""

def run_training_stream(dataset_path, output_path, iterations):
    if not dataset_path or not os.path.exists(dataset_path):
        yield f"❌ Error: Dataset path '{dataset_path}' does not exist on disk."
        return

    cmd = [
        PYTHON_EXE,
        os.path.join(GS_REPO_DIR, "train.py"),
        "-s", dataset_path,
        "-m", output_path or "output/model",
        "--iterations", str(int(iterations))
    ]
    
    yield f"🚀 Launching 2D Gaussian Splatting Training:\nCommand: {' '.join(cmd)}\n\n"
    
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=GS_REPO_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in proc.stdout:
            yield line
        proc.wait()
        if proc.returncode == 0:
            yield "\n✅ Training completed successfully!"
        else:
            yield f"\n⚠️ Process finished with return code {proc.returncode}"
    except Exception as e:
        yield f"\n❌ Execution error: {str(e)}"

def run_rendering_stream(dataset_path, model_path, mesh_resolution, unbounded):
    if not dataset_path or not os.path.exists(dataset_path):
        yield f"❌ Error: Dataset path '{dataset_path}' does not exist."
        return
        
    cmd = [
        PYTHON_EXE,
        os.path.join(GS_REPO_DIR, "render.py"),
        "-s", dataset_path,
        "-m", model_path or "output/model",
        "--skip_train",
        "--skip_test",
        "--mesh_res", str(int(mesh_resolution))
    ]
    if unbounded:
        cmd.append("--unbounded")

    yield f"🎬 Launching 2D Gaussian Splatting Rendering & Mesh Extraction:\nCommand: {' '.join(cmd)}\n\n"

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=GS_REPO_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in proc.stdout:
            yield line
        proc.wait()
        if proc.returncode == 0:
            yield "\n✅ Rendering & Mesh Extraction completed!"
        else:
            yield f"\n⚠️ Process finished with return code {proc.returncode}"
    except Exception as e:
        yield f"\n❌ Execution error: {str(e)}"

def create_ui():
    with gr.Blocks(title="2D Gaussian Splatting: Radiance Field Reconstruction", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🌟 2D Gaussian Splatting: Geometrically Accurate Radiance Field Reconstruction
            ### Interactive Web Interface & Control Suite
            *Based on LearnOpenCV & 2D Gaussian Splatting (Huang et al., SIGGRAPH 2024)*
            """
        )

        with gr.Tab("📊 System & Overview"):
            status_md = gr.Markdown(check_system_status())
            refresh_btn = gr.Button("🔄 Refresh System Status")
            refresh_btn.click(fn=check_system_status, outputs=status_md)

            gr.Markdown(
                """
                ---
                ### 📌 Overview & Architecture
                2D Gaussian Splatting (2DGS) utilizes oriented 2D planar Gaussian disks (surfels) rather than 3D ellipsoids:
                - **Accurate Surface Geometry**: Eliminates ray-intersection ambiguity and multi-view artifacts.
                - **Consistent Normals**: Enables high-quality depth distortion constraints and normal supervision.
                - **Mesh Extraction**: Supports direct surface extraction at arbitrary resolutions via Poisson or Marching Tetrahedra.
                """
            )

        with gr.Tab("🏋️ Training"):
            gr.Markdown("### Configure & Run Training Pipeline")
            with gr.Row():
                train_dataset_input = gr.Textbox(
                    label="Dataset Path (COLMAP format / NeRF Synthetic)",
                    placeholder="e.g. C:/datasets/flowers or ./sample_scene",
                    scale=2
                )
                train_output_input = gr.Textbox(
                    label="Output Model Directory",
                    value="output/flowers",
                    scale=1
                )
            train_iterations = gr.Slider(
                label="Iterations",
                minimum=1000,
                maximum=30000,
                value=7000,
                step=1000
            )
            train_btn = gr.Button("🚀 Start Training", variant="primary")
            train_logs = gr.Textbox(label="Live Training Console Logs", lines=15, max_lines=25)
            train_btn.click(
                fn=run_training_stream,
                inputs=[train_dataset_input, train_output_input, train_iterations],
                outputs=train_logs
            )

        with gr.Tab("🎨 Rendering & Mesh Extraction"):
            gr.Markdown("### Render Radiance Fields & Extract 3D Surface Mesh")
            with gr.Row():
                render_dataset_input = gr.Textbox(
                    label="Dataset Path",
                    placeholder="e.g. C:/datasets/flowers",
                    scale=2
                )
                render_model_input = gr.Textbox(
                    label="Trained Model Directory",
                    value="output/flowers",
                    scale=1
                )
            with gr.Row():
                mesh_res_slider = gr.Slider(
                    label="Mesh Resolution",
                    minimum=256,
                    maximum=2048,
                    value=1024,
                    step=128
                )
                unbounded_checkbox = gr.Checkbox(label="Unbounded Scene Reconstruction", value=False)
            render_btn = gr.Button("🎬 Start Rendering & Mesh Extraction", variant="primary")
            render_logs = gr.Textbox(label="Live Rendering Console Logs", lines=15, max_lines=25)
            render_btn.click(
                fn=run_rendering_stream,
                inputs=[render_dataset_input, render_model_input, mesh_res_slider, unbounded_checkbox],
                outputs=render_logs
            )

        with gr.Tab("📂 3D Model & Mesh Viewer"):
            gr.Markdown("### Interactive 3D Surface / Point Cloud Viewer (PLY / OBJ)")
            model_3d = gr.Model3D(label="3D Model Viewer")
            file_upload = gr.File(label="Upload .ply or .obj Mesh File", file_types=[".ply", ".obj"])
            file_upload.change(fn=lambda f: f.name if f else None, inputs=file_upload, outputs=model_3d)

    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
