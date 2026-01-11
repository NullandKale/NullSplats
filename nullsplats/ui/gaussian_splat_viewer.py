"""OpenGL Gaussian Splatting viewer widget for 3D visualization.

This module provides a proper Gaussian splatting renderer that displays
Gaussians as textured quads with shader-based falloff, embedded in Tkinter.
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
from typing import Callable, List, Optional
import logging
import ctypes
import math
from pathlib import Path

from OpenGL.GL import *
from OpenGL.GLU import *
from pyopengltk import OpenGLFrame
OPENGL_AVAILABLE = True

import torch
TORCH_AVAILABLE = True
# Check if CUDA is available for GPU sorting
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    TORCH_DEVICE = torch.device('cuda')
else:
    TORCH_DEVICE = torch.device('cpu')

logger = logging.getLogger(__name__)

from nullsplats.ui.gaussian_splat_camera import Camera

class GaussianSplatViewer(OpenGLFrame if OPENGL_AVAILABLE else tk.Frame):
    """OpenGL-based Gaussian Splatting viewer for 3D visualization.
    
    Renders Gaussians as instanced quads with proper covariance projection,
    Gaussian falloff, and alpha blending. Embedded directly in Tkinter.
    """
    
    def __init__(self, parent: tk.Widget, width: int = 800, height: int = 600):
        """Initialize the Gaussian splat viewer.
        
        Args:
            parent: Parent Tkinter widget
            width: Viewer width in pixels
            height: Viewer height in pixels
        """
        self.width = width
        self.height = height
        
        # Gaussian data
        self.means = None           # (N, 3) - Gaussian centers
        self.scales = None          # (N, 3) - Log variances (log σ²)
        self.rotations = None       # (N, 4) - Quaternions (x,y,z,w)
        self.opacities = None       # (N, 1) - Logit opacities
        self.sh_dc = None           # (N, 3) - DC band of spherical harmonics
        self.num_gaussians = 0
        
        # Scene bounds for auto-centering
        self.scene_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.scene_bounds_min = None
        self.scene_bounds_max = None
        self.scene_size = 1.0
        
        # Shader program and buffers
        self.shader_program = None
        self.quad_vao = None
        self.quad_vao_onscreen = None
        self.quad_vao_offscreen = None
        self.quad_vbo = None
        self.instance_vbo = None     # Per-instance attributes (onscreen default)
        self.instance_vbo_onscreen = None
        self.instance_vbo_offscreen = None
        
        # Camera with proper look-at system
        self.camera = Camera()
        
        # Render settings
        self.point_scale = 1.0       # Global scale multiplier
        self.background_color = (0.0, 0.0, 0.0, 1.0)
        self.scale_bias = np.zeros(3, dtype=np.float32)  # Per-axis log-scale bias
        self.opacity_bias = 0.0      # Additional logistic bias for opacity
        
        # Debug visualization modes
        self.debug_mode = False       # Enable debug rendering
        self.debug_flat_color = False # Render as flat colored quads without Gaussian falloff
        
        # Mouse state
        self._last_x = 0
        self._last_y = 0
        self._mouse_button = None
        
        # Pending data upload
        self._pending_gaussians = None
        self._needs_data_upload = False
        self._warned_no_gaussians = False

        # Depth sorting - initially sort every frame to ensure visibility
        self._sorted_indices = None
        self._mark_depth_sort_needed()
        self._needs_depth_sort_onscreen = True
        self._needs_depth_sort_offscreen = True
        self._frame_count = 0
        self._sort_back_to_front = False
        self._invert_onscreen_sort = True
        self._invert_offscreen_sort = True
        self._last_onscreen_sort_view = None
        self._frame_listeners: List[Callable[[], None]] = []
        self._suspend_frame_listeners = False
        self._rendering_offscreen = False
        self._preserve_viewport = False
        self._flip_y = True
        self._view_offset_x = 0.0
        self._projection_shift_x = 0.0
        self._invert_view_y = False
        self._offscreen_sort_view = None
        self._fov_deg = 45.0
        
        if not OPENGL_AVAILABLE:
            super().__init__(parent)
            self._show_error()
        else:
            super().__init__(parent, width=width, height=height)
            try:
                self.configure(takefocus=True)
            except Exception:
                pass
            self.bind("<Button-1>", self._on_left_mouse_down)
            self.bind("<Button-3>", self._on_right_mouse_down)
            self.bind("<B1-Motion>", self._on_left_mouse_drag)
            self.bind("<B3-Motion>", self._on_right_mouse_drag)
            self.bind("<MouseWheel>", self._on_mouse_wheel)
            self.bind("<ButtonRelease-1>", self._on_mouse_up)
            self.bind("<ButtonRelease-3>", self._on_mouse_up)
            self.bind("<Enter>", lambda _event: self.focus_set())
            # Keep the render loop paused until data or explicit start is requested.
            self.animate = 0
    
    def _show_error(self):
        """Show error when PyOpenGL not available."""
        error_label = ttk.Label(
            self,
            text="PyOpenGL not available.\n\nInstall with:\npip install PyOpenGL pyopengltk",
            font=("TkDefaultFont", 11),
            foreground="red",
            justify=tk.CENTER
        )
        error_label.pack(expand=True)
    
    def initgl(self):
        """Initialize OpenGL context and shaders."""
        if not OPENGL_AVAILABLE:
            return
        
        try:
            logger.info("Initializing Gaussian Splat Viewer OpenGL...")
            logger.info(f"OpenGL Version: {glGetString(GL_VERSION).decode()}")
            logger.info(f"OpenGL Renderer: {glGetString(GL_RENDERER).decode()}")
            
            # Set background color
            glClearColor(*self.background_color)
            
            # Depth testing is disabled for correct alpha compositing (sorted splats).
            glDisable(GL_DEPTH_TEST)
            
            # Enable blending for alpha transparency (standard alpha blending)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glBlendEquation(GL_FUNC_ADD)
            
            # Disable face culling (we want to see both sides)
            glDisable(GL_CULL_FACE)
            
            # Enable point sprite (if supported)
            try:
                glEnable(GL_POINT_SPRITE)
                glEnable(GL_PROGRAM_POINT_SIZE)
            except:
                pass  # May not be available in all OpenGL versions
            
            # Create shader program
            self._create_shader_program()
            logger.info("Gaussian splat shaders created")
            
            # Create quad geometry (unit quad that will be instanced)
            self._create_quad_geometry()
            
            # Check for OpenGL errors during initialization
            err = glGetError()
            if err != GL_NO_ERROR:
                logger.error(f"OpenGL error during initialization: {err}")
            else:
                logger.info("Gaussian Splat Viewer OpenGL initialized successfully")
        except Exception as e:
            logger.exception("Failed to initialize Gaussian Splat Viewer OpenGL")
    
    def _create_shader_program(self):
        """Load and compile shaders for Gaussian splatting."""
        # Get shader directory
        shader_dir = Path(__file__).parent / "shaders"
        
        # Load vertex shader
        vert_path = shader_dir / "gaussian_splat.vert"
        if not vert_path.exists():
            raise FileNotFoundError(f"Vertex shader not found: {vert_path}")
        
        with open(vert_path, 'r') as f:
            vertex_shader_source = f.read()
        
        # Load fragment shader
        frag_path = shader_dir / "gaussian_splat.frag"
        if not frag_path.exists():
            raise FileNotFoundError(f"Fragment shader not found: {frag_path}")
        
        with open(frag_path, 'r') as f:
            fragment_shader_source = f.read()
        
        # Compile vertex shader
        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex_shader, vertex_shader_source)
        glCompileShader(vertex_shader)
        
        if not glGetShaderiv(vertex_shader, GL_COMPILE_STATUS):
            error = glGetShaderInfoLog(vertex_shader).decode()
            logger.error(f"Vertex shader compilation failed: {error}")
            raise RuntimeError(f"Vertex shader error: {error}")
        
        # Compile fragment shader
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment_shader, fragment_shader_source)
        glCompileShader(fragment_shader)
        
        if not glGetShaderiv(fragment_shader, GL_COMPILE_STATUS):
            error = glGetShaderInfoLog(fragment_shader).decode()
            logger.error(f"Fragment shader compilation failed: {error}")
            raise RuntimeError(f"Fragment shader error: {error}")
        
        # Link program
        self.shader_program = glCreateProgram()
        glAttachShader(self.shader_program, vertex_shader)
        glAttachShader(self.shader_program, fragment_shader)
        glLinkProgram(self.shader_program)
        
        if not glGetProgramiv(self.shader_program, GL_LINK_STATUS):
            error = glGetProgramInfoLog(self.shader_program).decode()
            logger.error(f"Shader program linking failed: {error}")
            raise RuntimeError(f"Shader linking error: {error}")
        
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)
        
        logger.info("Gaussian splat shaders compiled successfully")

    def _setup_instance_vao(self, vao: int, instance_vbo: int) -> None:
        """Bind quad + instance attributes into a VAO."""
        glBindVertexArray(vao)

        # Quad vertices (location 0).
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vbo)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        # Per-instance attributes.
        glBindBuffer(GL_ARRAY_BUFFER, instance_vbo)
        stride = (3 + 3 + 4 + 1 + 3) * 4

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glVertexAttribDivisor(1, 1)

        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * 4))
        glVertexAttribDivisor(2, 1)

        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(6 * 4))
        glVertexAttribDivisor(3, 1)

        glEnableVertexAttribArray(4)
        glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(10 * 4))
        glVertexAttribDivisor(4, 1)

        glEnableVertexAttribArray(5)
        glVertexAttribPointer(5, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(11 * 4))
        glVertexAttribDivisor(5, 1)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def _active_instance_vbo(self):
        if self._rendering_offscreen and self.instance_vbo_offscreen is not None:
            return self.instance_vbo_offscreen
        if self.instance_vbo_onscreen is not None:
            return self.instance_vbo_onscreen
        return self.instance_vbo

    def _active_vao(self):
        if self._rendering_offscreen and self.quad_vao_offscreen is not None:
            return self.quad_vao_offscreen
        if self.quad_vao_onscreen is not None:
            return self.quad_vao_onscreen
        return self.quad_vao

    def _mark_depth_sort_needed(self) -> None:
        self._needs_depth_sort = True
        self._needs_depth_sort_onscreen = True
        self._needs_depth_sort_offscreen = True
    
    def _create_quad_geometry(self):
        """Create unit quad geometry for instanced rendering."""
        # Quad vertices (2D, will be positioned by vertex shader)
        # Two triangles forming a quad: (-1,-1) to (1,1)
        quad_vertices = np.array([
            # Triangle 1
            -1.0, -1.0,
             1.0, -1.0,
             1.0,  1.0,
            # Triangle 2
            -1.0, -1.0,
             1.0,  1.0,
            -1.0,  1.0,
        ], dtype=np.float32)
        
        # Create VBO for quad vertices
        self.quad_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vbo)
        glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)

        # Create per-instance buffers for onscreen/offscreen paths.
        self.instance_vbo_onscreen = glGenBuffers(1)
        self.quad_vao_onscreen = glGenVertexArrays(1)
        self._setup_instance_vao(self.quad_vao_onscreen, self.instance_vbo_onscreen)

        self.instance_vbo_offscreen = glGenBuffers(1)
        self.quad_vao_offscreen = glGenVertexArrays(1)
        self._setup_instance_vao(self.quad_vao_offscreen, self.instance_vbo_offscreen)

        # Default to onscreen buffers for legacy code paths.
        self.instance_vbo = self.instance_vbo_onscreen
        self.quad_vao = self.quad_vao_onscreen
        
        logger.info("Quad geometry created for instanced rendering")
    
    def set_gaussians(self, means: np.ndarray, scales: np.ndarray,
                     rotations: np.ndarray, opacities: np.ndarray,
                     sh_dc: np.ndarray, *, preserve_camera: bool = False):
        """Set Gaussian data for rendering.
        
        Args:
            means: (N, 3) - Gaussian centers
            scales: (N, 3) - Log variances (log σ²)
            rotations: (N, 4) - Quaternions (x, y, z, w)
            opacities: (N,) or (N, 1) - Logit opacities
            sh_dc: (N, 3) - DC band of spherical harmonics (RGB)
        """
        if not OPENGL_AVAILABLE:
            return
        
        # Validate inputs
        n = means.shape[0]
        if scales.shape[0] != n or rotations.shape[0] != n:
            raise ValueError("All Gaussian parameters must have same count")
        
        # Ensure correct shapes
        means = np.ascontiguousarray(means, dtype=np.float32)
        scales = np.ascontiguousarray(scales, dtype=np.float32)
        rotations = np.ascontiguousarray(rotations, dtype=np.float32)
        
        if opacities.ndim == 1:
            opacities = opacities[:, np.newaxis]
        opacities = np.ascontiguousarray(opacities, dtype=np.float32)
        
        sh_dc = np.ascontiguousarray(sh_dc, dtype=np.float32)
        
        # Calculate scene bounds for auto-centering
        self.scene_bounds_min = means.min(axis=0)
        self.scene_bounds_max = means.max(axis=0)
        self.scene_center = (self.scene_bounds_min + self.scene_bounds_max) / 2.0
        
        scene_size = self.scene_bounds_max - self.scene_bounds_min
        self.scene_size = float(np.max(scene_size))
        if not np.isfinite(self.scene_size) or self.scene_size < 1e-3:
            # Avoid degenerate bounds that collapse the camera/projection.
            self.scene_size = 1.0
            self.scene_bounds_min = self.scene_center - 0.5
            self.scene_bounds_max = self.scene_center + 0.5
        
        cam_x = self.scene_center[0]
        cam_y = self.scene_center[1]
        cam_z = self.scene_bounds_min[2] - max(self.scene_size * 0.5, 0.5)  # In front of scene with minimum offset
        
        if not preserve_camera:
            # CAMERA OUTSIDE SCENE - looking IN at center
            # Place camera at MIN Z bound (in front), looking at center
            self.camera.set_position_direct(cam_x, cam_y, cam_z)
            self.camera.set_target_direct(self.scene_center[0], self.scene_center[1], self.scene_center[2])
        
        logger.info("=" * 80)
        logger.info(f"SCENE INFO:")
        logger.info(f"  Gaussians: {n:,}")
        logger.info(f"  Bounds: X[{self.scene_bounds_min[0]:.1f} to {self.scene_bounds_max[0]:.1f}]")
        logger.info(f"          Y[{self.scene_bounds_min[1]:.1f} to {self.scene_bounds_max[1]:.1f}]")
        logger.info(f"          Z[{self.scene_bounds_min[2]:.1f} to {self.scene_bounds_max[2]:.1f}]")
        logger.info(f"  Center: ({self.scene_center[0]:.1f}, {self.scene_center[1]:.1f}, {self.scene_center[2]:.1f})")
        logger.info(f"  Size: {self.scene_size:.1f}")
        logger.info("CAMERA: In front of scene (at min Z - offset), looking at center")
        if not preserve_camera:
            logger.info(f"  Position: ({cam_x:.1f}, {cam_y:.1f}, {cam_z:.1f})")
            logger.info(f"  Target: ({self.scene_center[0]:.1f}, {self.scene_center[1]:.1f}, {self.scene_center[2]:.1f})")
        logger.info("=" * 80)
        
        # Store for later use
        self._pending_gaussians = {
            'means': means,
            'scales': scales,
            'rotations': rotations,
            'opacities': opacities,
            'sh_dc': sh_dc
        }
        self._needs_data_upload = True
        self._mark_depth_sort_needed()
        self._frame_count = 0  # Reset frame counter
        
        logger.info(f"Gaussian data staged for upload")
    
    def _upload_gaussian_data(self):
        """Upload Gaussian data to GPU."""
        if self._pending_gaussians is None:
            return
        
        try:
            means = self._pending_gaussians['means']
            scales = self._pending_gaussians['scales']
            rotations = self._pending_gaussians['rotations']
            opacities = self._pending_gaussians['opacities']
            sh_dc = self._pending_gaussians['sh_dc']
            
            n = means.shape[0]

            # Interleave data: [mean(3), scale(3), rot(4), opacity(1), sh_dc(3)]
            data = np.empty((n, 14), dtype=np.float32)
            data[:, 0:3] = means
            data[:, 3:6] = scales
            data[:, 6:10] = rotations
            data[:, 10:11] = opacities
            data[:, 11:14] = sh_dc
            
            # Flatten
            data_flat = data.flatten()
            
            # Upload to GPU (keep onscreen/offscreen buffers in sync).
            target_vbos = [self.instance_vbo_onscreen, self.instance_vbo_offscreen]
            if not any(target_vbos):
                target_vbos = [self.instance_vbo]
            for vbo in target_vbos:
                if vbo is None:
                    continue
                glBindBuffer(GL_ARRAY_BUFFER, vbo)
                glBufferData(GL_ARRAY_BUFFER, data_flat.nbytes, data_flat, GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            
            self.num_gaussians = n
            self._warned_no_gaussians = False
            self.means = means
            self.scales = scales
            self.rotations = rotations
            self.opacities = opacities
            self.sh_dc = sh_dc
            
            self._pending_gaussians = None
            
            try:
                exp_scales = np.exp(0.5 * scales)
                alphas = 1.0 / (1.0 + np.exp(-opacities.reshape(-1)))
            except Exception:
                exp_scales = np.exp(scales, dtype=np.float64)
                alphas = 1.0 / (1.0 + np.exp(-opacities.reshape(-1)))
            scale_stats = (float(exp_scales.min()), float(exp_scales.max()), float(exp_scales.mean()))
            opacity_stats = (float(alphas.min()), float(alphas.max()))
            color_stats = (float(sh_dc.min()), float(sh_dc.max()))
            logger.info(
                "Uploaded %d Gaussians to GPU | scale_exp[min,mean,max]=[%.3g,%.3g,%.3g] "
                "| opacity[alpha_min,max]=[%.3g,%.3g] | sh_dc[min,max]=[%.3g,%.3g]",
                n,
                scale_stats[0],
                scale_stats[1],
                scale_stats[2],
                opacity_stats[0],
                opacity_stats[1],
                color_stats[0],
                color_stats[1],
            )
            if scale_stats[0] < 1e-4:
                logger.warning("Upload warning: min exp(scale) is %.3g, splats may collapse", scale_stats[0])
            if opacity_stats[0] < 1e-3 or opacity_stats[1] > 0.999:
                logger.warning(
                    "Upload warning: opacity distribution is skewed [alpha_min=%.3g, alpha_max=%.3g]",
                    opacity_stats[0],
                    opacity_stats[1],
                )
        except Exception as e:
            logger.exception("Failed to upload Gaussian data")
            raise
    
    def _sort_gaussians_gpu(self, means: np.ndarray, view_matrix: np.ndarray) -> np.ndarray:
        """Sort Gaussians on GPU using PyTorch (CUDA-accelerated) by camera depth.
        
        Args:
            means: (N, 3) array of Gaussian centers
            view_matrix: (4, 4) numpy array representing the current view transform
            
        Returns:
            (N,) array of sorted indices (back to front for alpha blending)
        """
        # Move data to GPU
        means_gpu = torch.from_numpy(means).to(TORCH_DEVICE)
        view_gpu = torch.from_numpy(view_matrix[:3, :4]).to(TORCH_DEVICE)

        # Compute camera coordinates using rotation + translation (view multiplies world -> camera)
        rot = view_gpu[:, :3]
        trans = view_gpu[:, 3]
        cam_coords = torch.matmul(means_gpu, rot.T) + trans
        depths = cam_coords[:, 2]

        # Sort by depth (ascending for front-to-back, descending for back-to-front)
        descending = self._effective_sort_back_to_front()
        sorted_indices = torch.argsort(depths, descending=descending)

        return sorted_indices.cpu().numpy().astype(np.int32)
    
    def _depth_sort_gaussians(self, view_matrix: Optional[np.ndarray] = None):
        """Sort Gaussians by depth for proper alpha blending.
        
        Uses GPU-accelerated sorting via PyTorch when available (CUDA),
        falls back to CPU NumPy sorting otherwise.
        """
        if self.means is None:
            return
        
        try:
            import time
            
            if view_matrix is None:
                view_matrix = self.camera.get_view_matrix()
            
            # Choose sorting method based on availability
            sort_method = "CPU NumPy"
            start_time = time.perf_counter()
            
            if TORCH_AVAILABLE and CUDA_AVAILABLE:
                # GPU-accelerated sorting using PyTorch CUDA
                sort_method = "GPU PyTorch (CUDA)"
                sorted_indices = self._sort_gaussians_gpu(self.means, view_matrix)
            elif TORCH_AVAILABLE:
                # PyTorch available but no CUDA - still use PyTorch on CPU
                # (may be faster than NumPy for large arrays due to optimizations)
                sort_method = "CPU PyTorch"
                sorted_indices = self._sort_gaussians_gpu(self.means, view_matrix)
            else:
                # Fallback to NumPy CPU sorting
                # Compute distances from camera (simpler than full transform)
                # This is approximate but much faster than full matrix multiply
                rot = view_matrix[:3, :3]
                trans = view_matrix[:3, 3]
                cam_coords = self.means @ rot.T + trans
                depths = cam_coords[:, 2]
                sorted_indices = np.argsort(depths)
                if self._effective_sort_back_to_front():
                    sorted_indices = sorted_indices[::-1]
                sorted_indices = sorted_indices.astype(np.int32)
            
            sort_time = time.perf_counter() - start_time
            
            # Re-upload data in sorted order
            if self._pending_gaussians is None:
                # Create sorted data
                sorted_data = np.empty((self.num_gaussians, 14), dtype=np.float32)
                sorted_data[:, 0:3] = self.means[sorted_indices]
                sorted_data[:, 3:6] = self.scales[sorted_indices]
                sorted_data[:, 6:10] = self.rotations[sorted_indices]
                sorted_data[:, 10:11] = self.opacities[sorted_indices].reshape(-1, 1)
                sorted_data[:, 11:14] = self.sh_dc[sorted_indices]
                
                # Upload to GPU
                data_flat = sorted_data.flatten()
                target_vbo = self._active_instance_vbo() or self.instance_vbo
                if target_vbo is not None:
                    glBindBuffer(GL_ARRAY_BUFFER, target_vbo)
                    glBufferData(GL_ARRAY_BUFFER, data_flat.nbytes, data_flat, GL_DYNAMIC_DRAW)
                    glBindBuffer(GL_ARRAY_BUFFER, 0)
            
            if self._rendering_offscreen:
                self._needs_depth_sort_offscreen = False
            else:
                self._needs_depth_sort_onscreen = False
                self._needs_depth_sort = False
            
            # Log performance information
            logger.debug(f"Sorted {self.num_gaussians:,} Gaussians using {sort_method} in {sort_time*1000:.2f}ms ({1/sort_time:.1f} sorts/sec)")
            
            # Log first-time sorting method
            if not hasattr(self, '_sort_method_logged'):
                self._sort_method_logged = True
                if TORCH_AVAILABLE and CUDA_AVAILABLE:
                    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown"
                    logger.info(f"=== GPU SORTING ENABLED ===")
                    logger.info(f"Using CUDA device: {gpu_name}")
                    logger.info(f"Expected speedup: 50-100x faster than CPU for large datasets")
                elif TORCH_AVAILABLE:
                    logger.info(f"=== CPU PyTorch SORTING ===")
                    logger.info(f"PyTorch available but no CUDA detected")
                    logger.info(f"Using PyTorch CPU backend (may be faster than NumPy)")
                else:
                    logger.info(f"=== CPU NumPy SORTING (FALLBACK) ===")
                    logger.info(f"PyTorch not available - using NumPy CPU sorting")
                    logger.info(f"For better performance, install PyTorch with CUDA support")
        except Exception as e:
            logger.exception("Failed to sort Gaussians by depth")
    
    def redraw(self):
        """Render the Gaussian splat."""
        if not OPENGL_AVAILABLE or self.shader_program is None:
            return
        
        # Validate shader program is still valid (not deleted)
        try:
            if not glIsProgram(self.shader_program):
                logger.warning("Shader program no longer valid, skipping render")
                self.shader_program = None
                return
        except:
            # OpenGL context may not be available
            return
        
        try:
            # Set background color
            glClearColor(*self.background_color)
            
            # Upload pending data if needed
            if self._needs_data_upload:
                self._upload_gaussian_data()
                self._needs_data_upload = False
            
            # Validate widget dimensions before sorting (needed for projection)
            if self.width <= 0 or self.height <= 0:
                return
            widget_aspect = self.width / self.height
            near_plane = max(self.scene_size * 0.01, 0.01)
            far_plane = max(self.scene_size * 10.0, near_plane * 10.0)
            projection = self._perspective(self._fov_deg, widget_aspect, near_plane, far_plane)
            if self._flip_y:
                projection[1, 1] *= -1.0
            if self._projection_shift_x:
                projection = projection.copy()
                projection[0, 2] += self._projection_shift_x
            view = self.camera.get_view_matrix()
            if self._view_offset_x:
                view = view.copy()
                view[0, 3] += self._view_offset_x
            if self._invert_view_y:
                view = view.copy()
                view[1, :] *= -1.0

            sort_view = view
            if self._rendering_offscreen and self._offscreen_sort_view is not None:
                sort_view = self._offscreen_sort_view
            needs_sort = self._needs_depth_sort_offscreen if self._rendering_offscreen else self._needs_depth_sort_onscreen

            # Depth sort (always for first few frames, then throttle)
            if not self._rendering_offscreen:
                self._frame_count += 1
            if needs_sort:
                if self._rendering_offscreen:
                    if sort_view is not None:
                        self._depth_sort_gaussians(sort_view)
                else:
                    if self._frame_count <= 30:
                        if sort_view is not None:
                            self._depth_sort_gaussians(sort_view)
                    else:
                        if not hasattr(self, '_frames_since_sort'):
                            self._frames_since_sort = 0
                        self._frames_since_sort += 1

                        if self._frames_since_sort >= 10:
                            if sort_view is not None:
                                self._depth_sort_gaussians(sort_view)
                            self._frames_since_sort = 0

            # DEBUG: Log first render
            if not hasattr(self, '_debug_logged'):
                self._debug_logged = True
                logger.info(f"=== RENDER CONFIGURATION ===")
                logger.info(f"Rendering {self.num_gaussians:,} Gaussians")
                logger.info(f"Viewport: {self.width}x{self.height}")
                logger.info(f"Near plane: {near_plane:.3f}, Far plane: {far_plane:.3f}")
                logger.info(f"Camera position: ({self.camera.position[0]:.3f}, {self.camera.position[1]:.3f}, {self.camera.position[2]:.3f})")
                logger.info(f"Camera target: ({self.camera.target[0]:.3f}, {self.camera.target[1]:.3f}, {self.camera.target[2]:.3f})")
                logger.info(f"Point scale: {self.point_scale}")
                logger.info(f"Background: {self.background_color}")
                if self.means is not None:
                    logger.info(f"First 3 Gaussian means:")
                    for i in range(min(3, len(self.means))):
                        logger.info(f"  [{i}]: ({self.means[i, 0]:.3f}, {self.means[i, 1]:.3f}, {self.means[i, 2]:.3f})")
                logger.info(f"View matrix:\n{view}")
                logger.info(f"Projection matrix:\n{projection}")
                rot = view[:3, :3]
                orthonormal = rot.T @ rot
                if not np.allclose(orthonormal, np.eye(3), atol=1e-4):
                    logger.warning("View rotation not orthonormal: diag=%s", np.diag(orthonormal))

            # Clear buffers
            # Keep viewport in sync with the Tk widget dimensions so the render fills the space.
            int_width = max(1, int(self.width))
            int_height = max(1, int(self.height))
            if not getattr(self, "_preserve_viewport", False):
                glViewport(0, 0, int_width, int_height)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Skip rendering if no data
            if self.num_gaussians == 0:
                if not self._warned_no_gaussians:
                    logger.debug("No Gaussians to render (waiting for data)")
                    self._warned_no_gaussians = True
                return

            # Use shader program
            glUseProgram(self.shader_program)
            
            # Set uniforms with error checking
            proj_loc = glGetUniformLocation(self.shader_program, "projection")
            view_loc = glGetUniformLocation(self.shader_program, "view")
            point_scale_loc = glGetUniformLocation(self.shader_program, "point_scale")
            scale_bias_loc = glGetUniformLocation(self.shader_program, "scale_bias")
            opacity_bias_loc = glGetUniformLocation(self.shader_program, "opacity_bias")
            viewport_size_loc = glGetUniformLocation(self.shader_program, "viewport_size")
            
            # Verify critical uniforms (only log errors once)
            if not hasattr(self, '_uniform_errors_logged'):
                self._uniform_errors_logged = True
                if proj_loc == -1:
                    logger.error("Failed to get 'projection' uniform location")
                if view_loc == -1:
                    logger.error("Failed to get 'view' uniform location")
                if point_scale_loc == -1:
                    logger.warning("'point_scale' uniform not found (may be optimized out if unused)")
                if scale_bias_loc == -1:
                    logger.warning("'scale_bias' uniform not found (may be optimized out if unused)")
                if opacity_bias_loc == -1:
                    logger.warning("'opacity_bias' uniform not found (may be optimized out if unused)")
                if viewport_size_loc == -1:
                    logger.error("Failed to get 'viewport_size' uniform location")
            
            # Set uniform values
            glUniformMatrix4fv(proj_loc, 1, GL_TRUE, projection)
            glUniformMatrix4fv(view_loc, 1, GL_TRUE, view)
            
            if not hasattr(self, "_uniform_bias_logged"):
                logger.debug(
                    "Uniform biases: scale_bias=%s opacity_bias=%.3f",
                    self.scale_bias.tolist(),
                    self.opacity_bias,
                )
                self._uniform_bias_logged = True
            # Apply debug scale multiplier if in debug mode
            debug_scale = self.point_scale * (5.0 if self.debug_mode else 1.0)
            glUniform1f(point_scale_loc, debug_scale)
            if scale_bias_loc != -1:
                glUniform3fv(scale_bias_loc, 1, self.scale_bias)
            if opacity_bias_loc != -1:
                glUniform1f(opacity_bias_loc, self.opacity_bias)
            
            # Pass viewport size
            viewport_size = np.array([float(int_width), float(int_height)], dtype=np.float32)
            glUniform2fv(viewport_size_loc, 1, viewport_size)
            
            # Ensure correct OpenGL state before drawing
            active_vao = self._active_vao()
            glBindVertexArray(active_vao)
            
            # Check OpenGL state
            if self._frame_count == 1:
                logger.info("=== OPENGL STATE CHECK ===")
                logger.info(f"Depth test enabled: {glIsEnabled(GL_DEPTH_TEST)}")
                logger.info(f"Blend enabled: {glIsEnabled(GL_BLEND)}")
                logger.info(f"VAO bound: {active_vao}")
                logger.info(f"Drawing {self.num_gaussians:,} instances")
            
            # Draw instanced quads
            glDrawArraysInstanced(GL_TRIANGLES, 0, 6, self.num_gaussians)
            glBindVertexArray(0)
            
            # Unbind program
            glUseProgram(0)

            # Notify any external listeners that a frame was rendered.
            if self._frame_listeners and not self._suspend_frame_listeners:
                for listener in list(self._frame_listeners):
                    try:
                        listener()
                    except Exception:
                        logger.debug("Frame listener failed", exc_info=True)
            
            # DEBUG: Check if anything was actually drawn (first frame only)
            if not hasattr(self, '_draw_logged'):
                self._draw_logged = True
                err = glGetError()
                if err != GL_NO_ERROR:
                    logger.error(f"OpenGL error after draw: {err} (0x{err:04x})")
                else:
                    logger.info("glDrawArraysInstanced completed without GL errors")
            
        except Exception as e:
            logger.exception("Error during Gaussian splat rendering")

    def add_frame_listener(self, callback: Callable[[], None]) -> None:
        """Register a callback to be invoked after each redraw."""
        if not callable(callback):
            return
        self._frame_listeners.append(callback)

    def render_once(self) -> None:
        """Render a single onscreen frame without toggling the animate loop."""
        if not OPENGL_AVAILABLE:
            return
        try:
            make_current = getattr(self, "tkMakeCurrent", None)
            swap = getattr(self, "tkSwapBuffers", None)
            if callable(make_current):
                make_current()
            try:
                from OpenGL.GL import (
                    glBindFramebuffer,
                    glDisable,
                    glViewport,
                    GL_FRAMEBUFFER,
                    GL_SCISSOR_TEST,
                )

                glBindFramebuffer(GL_FRAMEBUFFER, 0)
                glDisable(GL_SCISSOR_TEST)
                glViewport(0, 0, max(1, int(self.width)), max(1, int(self.height)))
            except Exception:
                pass
            self.redraw()
            if callable(swap):
                swap()
        except Exception:
            logger.debug("render_once failed", exc_info=True)

    def render_offscreen(
        self,
        width: int,
        height: int,
        camera_pose: Optional[tuple[np.ndarray, np.ndarray]] = None,
        *,
        flip_y: bool = False,
        view_offset_x: float = 0.0,
        projection_shift_x: float = 0.0,
        invert_view_y: bool = False,
    ) -> bool:
        """Render a single frame into the currently bound framebuffer.

        width/height control the viewport size; the caller is responsible for binding
        the desired framebuffer and restoring GL state.
        """
        if not OPENGL_AVAILABLE or self.shader_program is None:
            return False
        if self._rendering_offscreen:
            return False
        self._rendering_offscreen = True
        prev_preserve = self._preserve_viewport
        prev_width, prev_height = self.width, self.height
        prev_pos = self.camera.position.copy()
        prev_target = self.camera.target.copy()
        prev_up = self.camera.up.copy()
        prev_flip = self._flip_y
        prev_view_offset = self._view_offset_x
        prev_proj_shift = self._projection_shift_x
        prev_invert_view = self._invert_view_y
        prev_sort_view = self._offscreen_sort_view
        try:
            self._preserve_viewport = True
            self._flip_y = bool(flip_y)
            self._view_offset_x = float(view_offset_x)
            self._projection_shift_x = float(projection_shift_x)
            self._invert_view_y = bool(invert_view_y)
            self.width = width
            self.height = height
            if camera_pose is not None:
                pos, target = camera_pose
                self.camera.set_position_direct(pos[0], pos[1], pos[2])
                self.camera.set_target_direct(target[0], target[1], target[2])
            try:
                self._offscreen_sort_view = self.camera.get_view_matrix().copy()
            except Exception:
                self._offscreen_sort_view = None
            self._suspend_frame_listeners = True
            try:
                make_current = getattr(self, "tkMakeCurrent", None)
                if callable(make_current):
                    make_current()
            except Exception:
                pass
            self.redraw()
            try:
                from OpenGL.GL import glFlush

                glFlush()
            except Exception:
                pass
            return True
        finally:
            self._suspend_frame_listeners = False
            self.width = prev_width
            self.height = prev_height
            self.camera.position = prev_pos
            self.camera.target = prev_target
            self.camera.up = prev_up
            self._preserve_viewport = prev_preserve
            self._flip_y = prev_flip
            self._view_offset_x = prev_view_offset
            self._projection_shift_x = prev_proj_shift
            self._invert_view_y = prev_invert_view
            self._offscreen_sort_view = prev_sort_view
            self._rendering_offscreen = False
    
    def _perspective(self, fov: float, aspect: float, near: float, far: float) -> np.ndarray:
        """Create perspective projection matrix."""
        f = 1.0 / math.tan(math.radians(fov) / 2.0)
        m = np.zeros((4, 4), dtype=np.float32)
        m[0, 0] = f / aspect
        m[1, 1] = f
        m[2, 2] = (far + near) / (near - far)
        m[2, 3] = (2 * far * near) / (near - far)
        m[3, 2] = -1.0
        return m
    
    
    # Mouse interaction handlers
    def _on_left_mouse_down(self, event):
        """Handle left mouse button press (rotation)."""
        try:
            self.focus_set()
        except Exception:
            pass
        self._last_x = event.x
        self._last_y = event.y
        self._mouse_button = 'left'
    
    def _on_right_mouse_down(self, event):
        """Handle right mouse button press (panning)."""
        try:
            self.focus_set()
        except Exception:
            pass
        self._last_x = event.x
        self._last_y = event.y
        self._mouse_button = 'right'
    
    def _on_left_mouse_drag(self, event):
        """Handle left mouse drag for rotation - orbit around target."""
        if self._mouse_button == 'left':
            dx = event.x - self._last_x
            dy = event.y - self._last_y
            
            to_camera = self.camera.position - self.camera.target
            distance = np.linalg.norm(to_camera)
            if distance < 1e-5:
                return

            sensitivity = 0.005
            yaw = math.atan2(to_camera[0], to_camera[2])
            pitch = math.asin(np.clip(to_camera[1] / distance, -0.999, 0.999))

            yaw -= dx * sensitivity
            pitch = np.clip(pitch + dy * sensitivity, -math.pi / 2 + 0.01, math.pi / 2 - 0.01)

            cos_pitch = math.cos(pitch)
            sin_pitch = math.sin(pitch)
            sin_yaw = math.sin(yaw)
            cos_yaw = math.cos(yaw)

            new_offset = np.array([
                distance * cos_pitch * sin_yaw,
                distance * sin_pitch,
                distance * cos_pitch * cos_yaw,
            ], dtype=np.float32)

            self.camera.position = self.camera.target + new_offset
            
            self._last_x = event.x
            self._last_y = event.y
            self._mark_depth_sort_needed()
            self._request_onscreen_redraw()
    
    def _on_right_mouse_drag(self, event):
        """Handle right mouse drag for panning."""
        if self._mouse_button == 'right':
            dx = event.x - self._last_x
            dy = event.y - self._last_y
            
            pan_speed = self.scene_size * 0.001
            view = self.camera.get_view_matrix()
            right = view[0, :3]
            up = view[1, :3]

            offset = right * dx * pan_speed - up * dy * pan_speed
            self.camera.target += offset
            self.camera.position += offset
            
            self._last_x = event.x
            self._last_y = event.y
            self._mark_depth_sort_needed()
            logger.debug("Pan change -> camera pos=%s target=%s", tuple(self.camera.position.tolist()), tuple(self.camera.target.tolist()))
            self._request_onscreen_redraw()
    
    def _on_mouse_up(self, event):
        """Handle mouse button release."""
        self._mouse_button = None
    
    def _on_mouse_wheel(self, event):
        """Handle mouse wheel for zoom - move camera toward/away from target."""
        # Get direction from camera to target
        to_target = self.camera.target - self.camera.position
        distance = np.linalg.norm(to_target)
        direction = to_target / (distance + 1e-8)
        
        # Zoom in/out
        if event.delta > 0:
            # Zoom in
            self.camera.position += direction * distance * 0.1
        else:
            # Zoom out
            self.camera.position -= direction * distance * 0.1
        
        self._mark_depth_sort_needed()
        logger.debug("Zoom change -> camera pos=%s target=%s", tuple(self.camera.position.tolist()), tuple(self.camera.target.tolist()))
        self._request_onscreen_redraw()

    def _request_onscreen_redraw(self) -> None:
        if getattr(self, "_rendering_offscreen", False):
            return
        try:
            if hasattr(self, "render_once"):
                self.render_once()
        except Exception:
            logger.debug("Onscreen redraw failed", exc_info=True)
    
    # Settings
    def set_point_scale(self, scale: float):
        """Set global point scale multiplier."""
        if not OPENGL_AVAILABLE:
            return
        self.point_scale = max(0.1, min(10.0, scale))

    def set_scale_bias(self, bias):
        """Apply per-axis log-scale bias before exponentiation."""
        if not OPENGL_AVAILABLE:
            return
        arr = np.ascontiguousarray(bias, dtype=np.float32)
        if arr.shape != (3,):
            raise ValueError("scale_bias must be an iterable of three floats")
        self.scale_bias = arr
        self._mark_depth_sort_needed()

    def set_opacity_bias(self, bias: float):
        """Adjust the sigmoid bias applied to each Gaussian's opacity."""
        if not OPENGL_AVAILABLE:
            return
        self.opacity_bias = float(bias)
        self._mark_depth_sort_needed()

    def set_sort_back_to_front(self, value: bool):
        """Toggle whether Gaussians are sorted back-to-front or front-to-back."""
        if not OPENGL_AVAILABLE:
            return
        self._sort_back_to_front = value
        self._mark_depth_sort_needed()

    def set_fov_deg(self, value: float) -> None:
        if not OPENGL_AVAILABLE:
            return
        self._fov_deg = float(value)

    def request_depth_sort(self):
        """Force the viewer to re-sort Gaussians on the next redraw."""
        if not OPENGL_AVAILABLE:
            return
        self._mark_depth_sort_needed()

    def _effective_sort_back_to_front(self) -> bool:
        effective = self._sort_back_to_front
        if self._rendering_offscreen and self._invert_offscreen_sort:
            effective = not effective
        elif not self._rendering_offscreen and self._invert_onscreen_sort:
            effective = not effective
        return effective
    
    def set_debug_mode(self, enabled: bool):
        """Enable debug rendering mode (larger splats for visibility)."""
        if not OPENGL_AVAILABLE:
            return
        self.debug_mode = enabled
        logger.info(f"Debug mode: {'ENABLED' if enabled else 'DISABLED'}")
        if enabled:
            logger.info("Debug mode increases splat size by 5x for visibility")
    
    def set_debug_flat_color(self, enabled: bool):
        """Enable flat color rendering (no Gaussian falloff)."""
        if not OPENGL_AVAILABLE:
            return
        self.debug_flat_color = enabled
        logger.info(f"Flat color debug: {'ENABLED' if enabled else 'DISABLED'}")
    
    def set_background_color(self, color: str):
        """Set background color."""
        if not OPENGL_AVAILABLE:
            return
        
        color_map = {
            'black': (0.0, 0.0, 0.0, 1.0),
            'white': (1.0, 1.0, 1.0, 1.0),
            'gray': (0.5, 0.5, 0.5, 1.0)
        }
        self.background_color = color_map.get(color, (0.0, 0.0, 0.0, 1.0))
    
    def _stop_rendering_safe(self):
        """Stop the render loop to reduce GPU usage."""
        if not OPENGL_AVAILABLE:
            return
        # Stop scheduling future redraws and cancel any pending callback to avoid
        # running GL code after the widget is unmapped (can crash on Windows).
        self.animate = 0
        try:
            if getattr(self, "cb", None):
                self.after_cancel(self.cb)
                self.cb = None
        except Exception as exc:
            logger.debug("Failed to cancel render callback: %s", exc)
        logger.info("Gaussian splat viewer rendering stopped")

    def stop_rendering(self):
        self._stop_rendering_safe()

    def _start_rendering_safe(self):
        """Start the render loop."""
        if not OPENGL_AVAILABLE:
            return
        self.animate = 1
        logger.info("Gaussian splat viewer rendering started")
        # If the widget is visible, kick off the display loop immediately.
        # Otherwise, retry shortly until Tk has mapped the widget.
        def _kickoff() -> None:
            if self.animate == 0:
                return
            if not self.winfo_ismapped():
                self.after(50, _kickoff)
                return
            if not getattr(self, "context_created", False):
                try:
                    self.tkCreateContext()
                    self.initgl()
                    self.context_created = True
                except Exception as exc:
                    logger.exception("Failed to create GL context during start_rendering")
                    self.animate = 0
                    return
            try:
                self._display()
            except Exception as exc:
                logger.exception("Display loop failed to start")
                self.animate = 0

        if not getattr(self, "cb", None):
            _kickoff()

    def start_rendering(self):
        self._start_rendering_safe()

    def _display(self):  # type: ignore[override]
        """Wrap base display loop to trap GL errors instead of crashing."""
        try:
            super()._display()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Display loop crashed; continuing")
            try:
                if self.animate > 0 and getattr(self, "cb", None) is None:
                    self.cb = self.after(50, self._display)
            except Exception:
                pass
    
    def set_camera_pose(self, position: np.ndarray, rotation: Optional[np.ndarray] = None, target: Optional[np.ndarray] = None):
        """Set camera to specific position with optional rotation or target.
        
        Args:
            position: (3,) array of camera position in world coordinates
            rotation: (3, 3) rotation matrix from world-to-camera (optional)
            target: (3,) array of camera target (optional, used if rotation not provided)
        """
        if not OPENGL_AVAILABLE:
            return
        
        self.camera.set_position_direct(position[0], position[1], position[2])
        
        if rotation is not None:
            # Rotation matrix is world-to-camera (R in [R|t] extrinsic)
            # Camera-to-world rotation is the transpose
            R_T = rotation.T
            
            # Camera looks along +Z in camera space for COLMAP-like data.
            forward_cam = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            forward_world = R_T @ forward_cam
            
            # Target is position + forward direction
            target_pos = position + forward_world
            final_target = target_pos if target is None else target
            self.camera.set_target_direct(final_target[0], final_target[1], final_target[2])
            
            # Keep a stable world-up to avoid inverted flips when targets are provided.
            self.camera.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            
        elif target is not None:
            self.camera.set_target_direct(target[0], target[1], target[2])
            logger.info(f"Camera set with target - position: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}), target: ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})")
        else:
            # Look at scene center by default
            self.camera.set_target_direct(
                self.scene_center[0],
                self.scene_center[1],
                self.scene_center[2]
            )
            logger.info(f"Camera set to scene center - position: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")
        
        # Trigger depth sort for new viewpoint
        self._mark_depth_sort_needed()
    
    
    
    
    
    
    
    
    

    
    def destroy(self):
        """Override destroy to ensure proper cleanup."""
        logger.info("Destroying GaussianSplatViewer widget")
        try:
            # Clear OpenGL resources first
            self.clear()
        except Exception as e:
            logger.warning(f"Error during destroy cleanup: {e}")
        finally:
            # Call parent destroy
            super().destroy()
    
    def clear(self):
        """Clear the viewer and release resources."""
        if OPENGL_AVAILABLE:
            try:
                # Stop rendering and drop any pending uploads; keep GL resources intact
                self.animate = 0
                try:
                    if getattr(self, "cb", None):
                        self.after_cancel(self.cb)
                        self.cb = None
                except Exception:
                    pass
                self._pending_gaussians = None
                self._needs_data_upload = False
                self._needs_depth_sort = False
                self._needs_depth_sort_onscreen = False
                self._needs_depth_sort_offscreen = False
                self.num_gaussians = 0
            except Exception as e:
                logger.warning(f"Error during Gaussian splat cleanup: {e}")
